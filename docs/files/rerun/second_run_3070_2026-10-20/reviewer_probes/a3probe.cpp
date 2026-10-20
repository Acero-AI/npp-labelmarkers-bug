// audit3: own tests. (1) 3x3 at every offset 0..S-3 of an SxS image of 9s, single call, judged; prints rule exceptions.
// (2) the same map through the batch call with TWO images per call (both images hold the pattern, at the same offset
//     in image 0 and at a fixed safe offset in image 1). (3) row-end join: W x 2 image, a[W-1]=a[W]=3, rest unique-ish.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <numeric>
#include <vector>
#include <cuda_runtime.h>
#include <npp.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

static int fnd(std::vector<int>& p, int i) { while (p[i] != i) i = p[i] = p[p[i]]; return i; }
static bool right(const std::vector<uint8_t>& img, const std::vector<uint32_t>& lab, int W, int H, int conn) {
    std::vector<int> p(W * H); std::iota(p.begin(), p.end(), 0);
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++)
        for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
            if (!dx && !dy) continue;
            if (conn == 4 && dx && dy) continue;
            int X = x + dx, Y = y + dy; if (X < 0 || Y < 0 || X >= W || Y >= H) continue;
            int i = y * W + x, j = Y * W + X; if (img[i] != img[j]) continue;
            int a = fnd(p, i), b = fnd(p, j); if (a != b) p[a] = b;
        }
    std::map<int, uint32_t> r2l; std::map<uint32_t, int> l2r;
    for (int i = 0; i < W * H; i++) {
        int r = fnd(p, i);
        auto a = r2l.find(r); if (a == r2l.end()) r2l[r] = lab[i]; else if (a->second != lab[i]) return false;
        auto b = l2r.find(lab[i]); if (b == l2r.end()) l2r[lab[i]] = r; else if (b->second != r) return false;
    }
    return true;
}
static NppStreamContext ctx;
static std::vector<uint32_t> single(const std::vector<uint8_t>& img, int W, int H, int conn) {
    NppiSize roi = {W, H}; int bytes = 0;
    nppiLabelMarkersUFGetBufferSize_32u_C1R(roi, &bytes);
    Npp8u *s, *buf; Npp32u* d;
    CK(cudaMalloc((void**)&s, W * H)); CK(cudaMalloc((void**)&d, 4 * W * H)); CK(cudaMalloc((void**)&buf, bytes));
    CK(cudaMemset(buf, 0, bytes));
    CK(cudaMemcpy(s, img.data(), W * H, cudaMemcpyHostToDevice));
    NppStatus st = nppiLabelMarkersUF_8u32u_C1R_Ctx(s, W, d, 4 * W, roi, conn == 8 ? nppiNormInf : nppiNormL1, buf, ctx);
    CK(cudaDeviceSynchronize());
    if (st) { printf("status %d\n", st); exit(1); }
    std::vector<uint32_t> lab(W * H); CK(cudaMemcpy(lab.data(), d, 4 * W * H, cudaMemcpyDeviceToHost));
    cudaFree(s); cudaFree(d); cudaFree(buf);
    return lab;
}
static void batch2(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, int W, int H, std::vector<uint32_t>& la, std::vector<uint32_t>& lb) {
    NppiSize roi = {W, H};
    Npp8u* s[2]; Npp32u* d[2]; NppiImageDescriptor hs[2], hd[2], *ds, *dd;
    const std::vector<uint8_t>* im[2] = {&a, &b};
    for (int k = 0; k < 2; k++) {
        CK(cudaMalloc((void**)&s[k], W * H)); CK(cudaMalloc((void**)&d[k], 4 * W * H));
        CK(cudaMemcpy(s[k], im[k]->data(), W * H, cudaMemcpyHostToDevice));
        hs[k] = {s[k], W, roi}; hd[k] = {d[k], 4 * W, roi};
    }
    CK(cudaMalloc((void**)&ds, sizeof hs)); CK(cudaMalloc((void**)&dd, sizeof hd));
    CK(cudaMemcpy(ds, hs, sizeof hs, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dd, hd, sizeof hd, cudaMemcpyHostToDevice));
    NppStatus st = nppiLabelMarkersUFBatch_8u32u_C1R_Advanced_Ctx(ds, dd, 2, roi, nppiNormInf, ctx);
    CK(cudaDeviceSynchronize());
    if (st) { printf("batch status %d\n", st); exit(1); }
    la.resize(W * H); lb.resize(W * H);
    CK(cudaMemcpy(la.data(), d[0], 4 * W * H, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(lb.data(), d[1], 4 * W * H, cudaMemcpyDeviceToHost));
    for (int k = 0; k < 2; k++) { cudaFree(s[k]); cudaFree(d[k]); } cudaFree(ds); cudaFree(dd);
}
static const uint8_t PAT[9] = {0, 2, 0, 0, 2, 0, 1, 0, 1};
static std::vector<uint8_t> placed(int S, int x0, int y0) {
    std::vector<uint8_t> im(S * S, 9);
    for (int k = 0; k < 9; k++) im[(y0 + k / 3) * S + x0 + k % 3] = PAT[k];
    return im;
}
int main(int argc, char** argv) {
    ctx.hStream = 0;
    CK(cudaGetDevice(&ctx.nCudaDeviceId));
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, ctx.nCudaDeviceId));
    ctx.nCudaDevAttrComputeCapabilityMajor = p.major; ctx.nCudaDevAttrComputeCapabilityMinor = p.minor;
    ctx.nMultiProcessorCount = p.multiProcessorCount; ctx.nMaxThreadsPerMultiProcessor = p.maxThreadsPerMultiProcessor;
    ctx.nMaxThreadsPerBlock = p.maxThreadsPerBlock; ctx.nSharedMemPerBlock = p.sharedMemPerBlock;
    CK(cudaStreamGetFlags(ctx.hStream, &ctx.nStreamFlags));
    const NppLibraryVersion* v = nppGetLibVersion();
    printf("NPP %d.%d.%d on %s\n", v->major, v->minor, v->build, p.name);
    for (int S : {70, 135}) {
        int n = 0, ok = 0, exc = 0, bn = 0, bok = 0, bexc = 0, bimg1bad = 0;
        for (int y0 = 0; y0 <= S - 3; y0++) for (int x0 = 0; x0 <= S - 3; x0++) {
            auto im = placed(S, x0, y0);
            bool r = right(im, single(im, S, S, 8), S, S, 8);
            bool rule = x0 % 16 == 15 || y0 % 16 == 14 || y0 % 16 == 15;
            n++; ok += r; exc += r != rule;
            if (S == 70) {
                auto other = placed(S, 2, 2);
                std::vector<uint32_t> la, lb; batch2(im, other, S, S, la, lb);
                bool br = right(im, la, S, S, 8);
                bn++; bok += br; bexc += br != rule; bimg1bad += !right(other, lb, S, S, 8);
            }
        }
        printf("single, %dx%d, %d offsets: correct %d, exceptions to the mod-16 rule %d\n", S, S, n, ok, exc);
        if (S == 70) printf("batch of 2, %dx%d, %d offsets: image0 correct %d, exceptions to the rule %d; image1 (pattern at 2,2) wrong %d\n", S, S, bn, bok, bexc, bimg1bad);
    }
    {   // row-end join: W x 3 images, every pixel its own value except pixel (W-1,0) and (0,1) both 200
        int joined = 0, tried = 0, wrongany = 0;
        for (int W = 3; W <= 130; W++) {
            int H = 3; std::vector<uint8_t> im(W * H);
            for (int i = 0; i < W * H; i++) im[i] = (i * 7 + (i / W) * 3) % 100;   // no two equal neighbours? ensured below
            for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) im[y * W + x] = ((x + 2 * y) % 4) * 10 + y;   // equal neighbours impossible: same row differs by 10, rows differ in low digit
            im[W - 1] = 200; im[W] = 200;
            auto lab = single(im, W, H, 8);
            tried++; joined += lab[W - 1] == lab[W]; wrongany += !right(im, lab, W, H, 8);
        }
        printf("row-end join, widths 3..130 (8 neighbours): row 0's last and row 1's first pixel share a label in %d of %d; labelling wrong in %d\n", joined, tried, wrongany);
    }
    return 0;
}
