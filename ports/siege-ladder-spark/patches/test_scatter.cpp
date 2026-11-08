#include "llama-mmap.h"
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <string>
#include <chrono>
int main(int argc, char ** argv) {
    std::string paths[2] = { argv[1], argv[2] };
    std::vector<std::vector<uint8_t>> ref(2);
    for (int f = 0; f < 2; ++f) { FILE * fp = fopen(paths[f].c_str(), "rb"); fseek(fp, 0, SEEK_END); ref[f].resize(ftell(fp)); fseek(fp, 0, SEEK_SET); fread(ref[f].data(), 1, ref[f].size(), fp); fclose(fp); }
    std::mt19937_64 rng(42);
    int fails = 0;
    for (int trial = 0; trial < 40; ++trial) {
        int n = 1 + rng() % 4000, thr = 1 + rng() % 8, qd = 1 + rng() % 64;
        std::vector<llama_scatter_piece> ps; std::vector<std::vector<uint8_t>> out(n);
        for (int i = 0; i < n; ++i) {
            uint32_t file = rng() % 2; uint64_t sz = ref[file].size();
            uint64_t sec = (rng() % (sz / 4096)) * 4096; uint32_t in = rng() % 4096; uint32_t len = 1 + rng() % (4096 - in);
            if (trial % 3 == 0) sec = (rng() % 8) * 4096; // many pieces share few sectors
            out[i].assign(len, 0xEE); ps.push_back({ sec + in, len, file, out[i].data() });
        }
        auto copy = ps;
        bool ok = llama_read_scattered(paths, ps, thr, qd);
        for (int i = 0; i < n; ++i) if (memcmp(copy[i].dst, ref[copy[i].file].data() + copy[i].off, copy[i].len)) { fails++; break; }
        if (!ok) fails++;
    }
    // throughput: 200k random 4K pieces, 8 threads x 32
    std::vector<llama_scatter_piece> ps; std::vector<uint8_t> sink(200000 * 64);
    for (int i = 0; i < 200000; ++i) { uint64_t sz = ref[0].size(); ps.push_back({ (rng() % (sz / 4096)) * 4096 + 7, 64, 0, sink.data() + i * 64 }); }
    auto t0 = std::chrono::steady_clock::now(); bool ok = llama_read_scattered(paths, ps, 8, 32);
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("correctness fails=%d  throughput ok=%d %.0f sectors/s\n", fails, ok, 200000 / s);
    return fails != 0;
}
