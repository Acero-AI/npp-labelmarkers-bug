// npp_batch: nppiLabelMarkersUFBatch_8u32u_C1R_Advanced_Ctx, called as documented (batch size 2), judged on the CPU,
// and nppiLabelMarkersUF_8u32u_C1R_Ctx on the same images for comparison.
//
// Each call labels two random 64 x 64 images of 1s and 2s (8 neighbours), 20 calls with different seeds.
// Every pair of touching, equal pixels must share a label; the program counts the pairs that do not
// ("broken links") and labels covering two true regions ("merged labels"). Exit status 2 when any image is wrong.
//
//     g++ -O2 npp_batch.cpp -I<npp include> -I<cudart include> -L<libs> -l:libnppif.so.N -l:libnppc.so.N -l:libcudart.so.N
#include <cstdio>
#include <cstdint>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <vector>
#include <cuda_runtime.h>
#include <npp.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s, line %d\n", cudaGetErrorString(e), __LINE__); return 1; } } while (0)

static int find(std::vector<int>& p, int i) { while (p[i] != i) i = p[i] = p[p[i]]; return i; }

// Counts broken links and merged labels of `lab` against the true 8-neighbour regions of `img`.
static void judge(const std::vector<uint8_t>& img, const std::vector<uint32_t>& lab, int W, int H, int* broken, int* merged) {
    std::vector<int> p(W * H);
    std::iota(p.begin(), p.end(), 0);
    const int dx[4] = {1, 0, 1, -1}, dy[4] = {0, 1, 1, 1};
    *broken = 0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            for (int d = 0; d < 4; d++) {
                int X = x + dx[d], Y = y + dy[d];
                if (X < 0 || X >= W || Y >= H) continue;
                int i = y * W + x, j = Y * W + X;
                if (img[i] != img[j]) continue;
                int a = find(p, i), b = find(p, j);
                if (a != b) p[std::max(a, b)] = std::min(a, b);
                if (lab[i] != lab[j]) ++*broken;
            }
    std::map<uint32_t, int> owner;
    *merged = 0;
    for (int i = 0; i < W * H; i++) {
        int r = find(p, i);
        auto it = owner.find(lab[i]);
        if (it == owner.end()) owner[lab[i]] = r;
        else if (it->second != r) { ++*merged; it->second = r; }
    }
}

int main() {
    const int W = 64, H = 64, N = W * H, B = 2, CALLS = 20;
    NppStreamContext ctx = {};
    ctx.hStream = 0;
    CK(cudaGetDevice(&ctx.nCudaDeviceId));
    CK(cudaDeviceGetAttribute(&ctx.nCudaDevAttrComputeCapabilityMajor, cudaDevAttrComputeCapabilityMajor, ctx.nCudaDeviceId));
    CK(cudaDeviceGetAttribute(&ctx.nCudaDevAttrComputeCapabilityMinor, cudaDevAttrComputeCapabilityMinor, ctx.nCudaDeviceId));
    CK(cudaStreamGetFlags(ctx.hStream, &ctx.nStreamFlags));
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, ctx.nCudaDeviceId));
    ctx.nMultiProcessorCount = p.multiProcessorCount;
    ctx.nMaxThreadsPerMultiProcessor = p.maxThreadsPerMultiProcessor;
    ctx.nMaxThreadsPerBlock = p.maxThreadsPerBlock;
    ctx.nSharedMemPerBlock = p.sharedMemPerBlock;
    const NppLibraryVersion* v = nppGetLibVersion();
    int rt = 0, drv = 0;
    CK(cudaRuntimeGetVersion(&rt));
    CK(cudaDriverGetVersion(&drv));
    printf("NPP %d.%d.%d, %s (sm_%d%d), CUDA runtime %d, driver supports %d\n", v->major, v->minor, v->build, p.name, p.major, p.minor, rt, drv);
    printf("batch API, %d images of %d x %d per call, 8 neighbours, %d calls\n\n", B, W, H, CALLS);

    NppiSize roi = {W, H};
    Npp8u* src[B];
    Npp32u* dst[B];
    for (int b = 0; b < B; b++) {
        CK(cudaMalloc((void**)&src[b], N));
        CK(cudaMalloc((void**)&dst[b], N * sizeof(Npp32u)));
    }
    NppiImageDescriptor hs[B], hd[B], *ds, *dd;
    for (int b = 0; b < B; b++) {
        hs[b] = {src[b], W, roi};
        hd[b] = {dst[b], W * (int)sizeof(Npp32u), roi};   // the batch call writes in place: pitch must be width * 4
    }
    CK(cudaMalloc((void**)&ds, sizeof hs));
    CK(cudaMalloc((void**)&dd, sizeof hd));
    CK(cudaMemcpy(ds, hs, sizeof hs, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dd, hd, sizeof hd, cudaMemcpyHostToDevice));

    {   // the judge itself, on a CPU labelling by flood fill: right labels must pass, a split or a merge must fail
        std::vector<uint8_t> img(N);
        std::mt19937 rng(7);
        for (auto& px : img) px = 1 + (rng() & 1);
        std::vector<uint32_t> lab(N, 0);
        uint32_t next = 0;
        for (int s = 0; s < N; s++) {
            if (lab[s]) continue;
            std::vector<int> todo = {s};
            lab[s] = ++next;
            while (!todo.empty()) {
                int i = todo.back(); todo.pop_back();
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++) {
                        int X = i % W + dx, Y = i / W + dy, j = Y * W + X;
                        if (X < 0 || X >= W || Y < 0 || Y >= H || lab[j] || img[j] != img[i]) continue;
                        lab[j] = next; todo.push_back(j);
                    }
            }
        }
        int b1, m1, b2, m2, b3, m3;
        judge(img, lab, W, H, &b1, &m1);
        int cut = 0;
        while (img[cut] != img[cut + 1]) cut++;                  // a pixel with an equal right neighbour
        std::vector<uint32_t> split = lab; split[cut] = next + 1; // cut off
        judge(img, split, W, H, &b2, &m2);
        std::vector<uint32_t> joined(N, 1);                       // everything one region
        judge(img, joined, W, H, &b3, &m3);
        printf("judge self-test: flood fill (%u regions) -> broken %d, merged %d; one pixel cut off -> broken %d; one label -> merged %d\n",
               next, b1, m1, b2, m3);
        if (b1 || m1 || !b2 || !m3) { printf("judge self-test FAILED\n"); return 1; }
    }

    Npp32u* one;   // the single-image call on the same images, for comparison
    Npp8u* buf;
    int bufsize = 0;
    CK(cudaMalloc((void**)&one, N * sizeof(Npp32u)));
    if (nppiLabelMarkersUFGetBufferSize_32u_C1R(roi, &bufsize) != NPP_SUCCESS) { printf("buffer size failed\n"); return 1; }
    CK(cudaMalloc((void**)&buf, bufsize));

    int wrong = 0, wrong_one = 0;
    for (int call = 0; call < CALLS; call++) {
        std::vector<std::vector<uint8_t>> img(B, std::vector<uint8_t>(N));
        for (int b = 0; b < B; b++) {
            std::mt19937 rng(1000 * call + b);
            for (auto& px : img[b]) px = 1 + (rng() & 1);
            CK(cudaMemcpy(src[b], img[b].data(), N, cudaMemcpyHostToDevice));
        }
        NppStatus st = nppiLabelMarkersUFBatch_8u32u_C1R_Advanced_Ctx(ds, dd, B, roi, nppiNormInf, ctx);
        CK(cudaDeviceSynchronize());
        if (st != NPP_SUCCESS) { printf("NPP status %d\n", st); return 1; }
        for (int b = 0; b < B; b++) {
            std::vector<uint32_t> lab(N);
            CK(cudaMemcpy(lab.data(), dst[b], N * sizeof(Npp32u), cudaMemcpyDeviceToHost));
            int broken, merged;
            judge(img[b], lab, W, H, &broken, &merged);
            wrong += broken || merged;
            printf("call %2d image %d: batch %zu labels, broken links %d, merged labels %d -> %s", call, b,
                   std::set<uint32_t>(lab.begin(), lab.end()).size(), broken, merged, broken || merged ? "WRONG" : "correct");
            NppStatus s1 = nppiLabelMarkersUF_8u32u_C1R_Ctx(src[b], W, one, W * (int)sizeof(Npp32u), roi, nppiNormInf, buf, ctx);
            CK(cudaDeviceSynchronize());
            if (s1 != NPP_SUCCESS) { printf("\nNPP status %d\n", s1); return 1; }
            CK(cudaMemcpy(lab.data(), one, N * sizeof(Npp32u), cudaMemcpyDeviceToHost));
            judge(img[b], lab, W, H, &broken, &merged);
            wrong_one += broken || merged;
            printf(" | single-image call: broken links %d, merged labels %d -> %s\n", broken, merged, broken || merged ? "WRONG" : "correct");
        }
    }
    printf("\nsingle-image call: %d of %d images wrong\n", wrong_one, B * CALLS);
    printf("%d of %d images wrong\n", wrong, B * CALLS);
    return wrong || wrong_one ? 2 : 0;
}
