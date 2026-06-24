"""moe_align: sort (token,expert) pairs by expert and pad each expert's run to a
BLOCK_M multiple, so a grouped GEMM block reads ONE expert's weights once and applies
to up to BLOCK_M tokens. Foundation for group-by-expert (amortizes the expert read at
batch). Returns fixed-upper-bound tensors (graph-friendly sizing)."""
import os
import torch
import triton
import triton.language as tl


@triton.jit
def _place_kernel(flat_ptr, eoff_ptr, counter_ptr, stok_ptr, sslot_ptr, valid_ptr,
                  M, k, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M
    e = tl.load(flat_ptr + offs, mask=mask, other=0)
    # atomic per-expert counter -> each pair's rank within its expert (collisions serialize -> distinct old vals)
    rank = tl.atomic_add(counter_ptr + e, 1, mask=mask)
    dest = tl.load(eoff_ptr + e, mask=mask, other=0) + rank
    tl.store(stok_ptr + dest, (offs // k).to(tl.int32), mask=mask)
    tl.store(sslot_ptr + dest, (offs % k).to(tl.int32), mask=mask)
    tl.store(valid_ptr + dest, 1, mask=mask)


def moe_align_fast(topk_ids, E, BM):
    """Graph-safe O(M) moe_align: count-per-expert (scatter_add) + cumsum on [E] ONLY + a Triton
    atomic-counter placement -- replaces the O(M*E) one_hot+cumsum. Identical outputs."""
    dev = topk_ids.device
    N, k = topk_ids.shape
    flat = topk_ids.reshape(-1).to(torch.int32)
    M = flat.numel()
    cnt = torch.zeros(E, dtype=torch.int32, device=dev)
    cnt.scatter_add_(0, flat.to(torch.int64), torch.ones(M, dtype=torch.int32, device=dev))
    padded = ((cnt + BM - 1) // BM) * BM
    eoff = (torch.cumsum(padded, 0) - padded).to(torch.int32)
    Pmax = M + min(M, E) * BM
    nblk = Pmax // BM
    sorted_tok = torch.zeros(Pmax, dtype=torch.int32, device=dev)
    sorted_slot = torch.zeros(Pmax, dtype=torch.int32, device=dev)
    valid = torch.zeros(Pmax, dtype=torch.bool, device=dev)
    counter = torch.zeros(E, dtype=torch.int32, device=dev)
    BLOCK = 256
    _place_kernel[(triton.cdiv(M, BLOCK),)](flat, eoff, counter, sorted_tok, sorted_slot, valid, M, k, BLOCK=BLOCK)
    block_start = (eoff // BM).contiguous()
    blk = torch.arange(nblk, device=dev)
    block_e = (torch.searchsorted(block_start, blk, right=True) - 1).clamp_(min=0).int()
    return sorted_tok, sorted_slot, valid, block_e, nblk


def moe_align(topk_ids, E, BM):
    if os.environ.get("VQ2_FAST_ALIGN", "1") == "1":   # default ON (shipped: 31x exact, quality-neutral)
        return moe_align_fast(topk_ids, E, BM)
    return _moe_align_torch(topk_ids, E, BM)


def _moe_align_torch(topk_ids, E, BM):
    """CUDA-graph-safe: uses ONLY one_hot/cumsum/gather/scatter/searchsorted -- NO
    bincount/argsort/.item() (those host-sync and break graph capture)."""
    dev = topk_ids.device
    N, k = topk_ids.shape
    flat = topk_ids.reshape(-1).to(torch.int64)      # [M] expert per pair
    M = flat.numel()
    oh = torch.nn.functional.one_hot(flat, E).to(torch.int32)   # [M, E]  (replaces bincount/argsort)
    cnt = oh.sum(0)                                   # [E] tokens per expert
    padded = ((cnt + BM - 1) // BM) * BM
    eoff = torch.cumsum(padded, 0) - padded           # block-region start per expert
    cum = torch.cumsum(oh, 0)                         # [M,E] running count per expert
    rank = cum.gather(1, flat[:, None]).squeeze(1) - 1            # [M] rank within expert
    dest = eoff.gather(0, flat) + rank                # [M] destination in padded layout
    # Tight, fixed (graph-safe) upper bound: active experts <= min(pairs M, E), each padded by < BM.
    # Old M + E*BM is loose for small M (spec verify): E*BM=16384 dominates -> ~268MB/layer intermediates
    # -> 43-layer cudagraph pool overflow (illegal address). min(M,E) keeps prefill identical (min=E) but
    # shrinks the verify ~4-8x. sum(padded) <= M + min(M,E)*(BM-1) < this bound, so dest stays in range.
    Pmax = M + min(M, E) * BM
    nblk = Pmax // BM
    ar = torch.arange(M, device=dev)
    sorted_tok = torch.zeros(Pmax, dtype=torch.int32, device=dev)
    sorted_slot = torch.zeros(Pmax, dtype=torch.int32, device=dev)
    valid = torch.zeros(Pmax, dtype=torch.bool, device=dev)
    sorted_tok.scatter_(0, dest, (ar // k).int())
    sorted_slot.scatter_(0, dest, (ar % k).int())
    valid.scatter_(0, dest, torch.ones(M, dtype=torch.bool, device=dev))
    block_start = (eoff // BM).contiguous()
    blk = torch.arange(nblk, device=dev)
    block_e = (torch.searchsorted(block_start, blk, right=True) - 1).clamp_(min=0).int()
    return sorted_tok, sorted_slot, valid, block_e, nblk


def _test():
    torch.manual_seed(0); E, BM = 8, 4
    ids = torch.topk(torch.randn(5, E), 3, 1).indices  # N=5, k=3, distinct experts
    st, ss, val, be, nb = moe_align(ids, E, BM)
    # check: every (token,expert) pair appears exactly once in a block of its expert
    ok = True
    seen = set()
    for b in range(nb):
        e = int(be[b])
        for i in range(BM):
            p = b * BM + i
            if val[p]:
                tok = int(st[p]); slot = int(ss[p])
                assert int(ids[tok, slot]) == e, f"block {b} expert {e} but pair routes to {int(ids[tok,slot])}"
                seen.add((tok, slot))
    assert len(seen) == ids.numel(), f"{len(seen)} vs {ids.numel()}"
    print(f"moe_align OK: {ids.numel()} pairs grouped into {nb} blocks (BM={BM}); each block one expert, all pairs covered once")


if __name__ == "__main__":
    _test()
