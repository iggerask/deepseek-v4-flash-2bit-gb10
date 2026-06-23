"""moe_align: sort (token,expert) pairs by expert and pad each expert's run to a
BLOCK_M multiple, so a grouped GEMM block reads ONE expert's weights once and applies
to up to BLOCK_M tokens. Foundation for group-by-expert (amortizes the expert read at
batch). Returns fixed-upper-bound tensors (graph-friendly sizing)."""
import torch


def moe_align(topk_ids, E, BM):
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
