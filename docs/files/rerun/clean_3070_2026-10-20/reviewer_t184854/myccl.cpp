// Independent audit test: own shapes, own BFS judge, own call sequence.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>
#include <deque>
#include <cuda_runtime.h>
#include <npp.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

struct Img { std::string name; int W, H; std::vector<uint32_t> v; };

static std::vector<int> bfs_truth(const Img& im, int conn, int* n) {
    std::vector<int> t(im.W * im.H, -1); *n = 0;
    for (int s = 0; s < im.W * im.H; s++) {
        if (t[s] >= 0) continue;
        std::deque<int> q{s}; t[s] = *n;
        while (!q.empty()) {
            int i = q.front(); q.pop_front(); int x = i % im.W, y = i / im.W;
            for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
                if (!dx && !dy) continue;
                if (conn == 4 && dx && dy) continue;
                int X = x + dx, Y = y + dy; if (X < 0 || Y < 0 || X >= im.W || Y >= im.H) continue;
                int j = Y * im.W + X; if (t[j] >= 0 || im.v[j] != im.v[i]) continue;
                t[j] = *n; q.push_back(j);
            }
        }
        ++*n;
    }
    return t;
}

// returns (#truth regions given >1 label, #labels covering >1 truth region)
static void judge(const Img& im, int conn, const std::vector<uint32_t>& lab, int* splits, int* merges, int* ntrue, int* nlab) {
    std::vector<int> t = bfs_truth(im, conn, ntrue);
    std::map<int, std::map<uint32_t,int>> t2l; std::map<uint32_t, std::map<int,int>> l2t;
    for (size_t i = 0; i < lab.size(); i++) { t2l[t[i]][lab[i]]++; l2t[lab[i]][t[i]]++; }
    *splits = 0; *merges = 0; *nlab = (int)l2t.size();
    for (auto& kv : t2l) if (kv.second.size() > 1) ++*splits;
    for (auto& kv : l2t) if (kv.second.size() > 1) ++*merges;
}

static NppStreamContext make_ctx() {
    NppStreamContext c; memset(&c, 0, sizeof c);
    CK(cudaGetDevice(&c.nCudaDeviceId));
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, c.nCudaDeviceId));
    c.nMultiProcessorCount = p.multiProcessorCount; c.nMaxThreadsPerMultiProcessor = p.maxThreadsPerMultiProcessor;
    c.nMaxThreadsPerBlock = p.maxThreadsPerBlock; c.nSharedMemPerBlock = p.sharedMemPerBlock;
    c.nCudaDevAttrComputeCapabilityMajor = p.major; c.nCudaDevAttrComputeCapabilityMinor = p.minor;
    c.hStream = 0; c.nStreamFlags = 0;
    return c;
}

template <class T> static std::vector<uint32_t> run(const Img& im, int conn, int depth_tag, bool compress, NppStreamContext ctx) {
    NppiSize roi = {im.W, im.H}; size_t N = (size_t)im.W * im.H;
    std::vector<T> h(N); for (size_t i = 0; i < N; i++) h[i] = (T)im.v[i];
    T* src; Npp32u* dst; Npp8u* buf; int bs = 0;
    CK(cudaMalloc((void**)&src, N * sizeof(T))); CK(cudaMalloc((void**)&dst, N * 4));
    CK(cudaMemcpy(src, h.data(), N * sizeof(T), cudaMemcpyHostToDevice));
    CK(cudaMemset(dst, 0, N * 4));
    if (nppiLabelMarkersUFGetBufferSize_32u_C1R(roi, &bs)) { printf("bs fail\n"); exit(1); }
    CK(cudaMalloc((void**)&buf, bs)); CK(cudaMemset(buf, 0, bs));
    NppiNorm nm = conn == 8 ? nppiNormInf : nppiNormL1; NppStatus st;
    if (depth_tag == 8) st = nppiLabelMarkersUF_8u32u_C1R_Ctx((Npp8u*)src, im.W * (int)sizeof(T), dst, im.W * 4, roi, nm, buf, ctx);
    else if (depth_tag == 16) st = nppiLabelMarkersUF_16u32u_C1R_Ctx((Npp16u*)src, im.W * (int)sizeof(T), dst, im.W * 4, roi, nm, buf, ctx);
    else st = nppiLabelMarkersUF_32u_C1R_Ctx((Npp32u*)src, im.W * (int)sizeof(T), dst, im.W * 4, roi, nm, buf, ctx);
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    if (st) { printf("status %d\n", st); exit(1); }
    if (compress) {
        int cbs = 0, nn = 0; Npp8u* cb;
        if (nppiCompressMarkerLabelsGetBufferSize_32u_C1R((int)N, &cbs)) { printf("cbs fail\n"); exit(1); }
        CK(cudaMalloc((void**)&cb, cbs));
        st = nppiCompressMarkerLabelsUF_32u_C1IR_Ctx(dst, im.W * 4, roi, (int)N, &nn, cb, ctx);
        CK(cudaDeviceSynchronize()); if (st) { printf("cstatus %d\n", st); exit(1); }
        cudaFree(cb);
    }
    std::vector<uint32_t> out(N); CK(cudaMemcpy(out.data(), dst, N * 4, cudaMemcpyDeviceToHost));
    cudaFree(src); cudaFree(dst); cudaFree(buf);
    return out;
}

static Img from(const char* name, int W, int H, std::vector<uint32_t> v) { return {name, W, H, v}; }

int main() {
    const NppLibraryVersion* lv = nppGetLibVersion(); int rt; cudaRuntimeGetVersion(&rt);
    printf("NPP %d.%d.%d runtime %d\n", lv->major, lv->minor, lv->build, rt);
    std::vector<Img> ims;
    ims.push_back(from("V3x3_7_42_99", 3, 3, {7, 42, 7, 7, 42, 7, 99, 7, 99}));
    ims.push_back(from("V3x3_200_13_250", 3, 3, {200, 13, 200, 200, 13, 200, 250, 200, 250}));
    { // all-same image: 1 region
        ims.push_back(from("flat_37_40x40", 40, 40, std::vector<uint32_t>(1600, 37)));
    }
    { // checkerboard 8-conn: 2 regions; 4-conn: W*H regions
        Img c{"checker_11_77_48x48", 48, 48, std::vector<uint32_t>(48 * 48)};
        for (int y = 0; y < 48; y++) for (int x = 0; x < 48; x++) c.v[y * 48 + x] = ((x + y) & 1) ? 77 : 11;
        ims.push_back(c);
    }
    { // big U, value 123 on 9 background, arms 30 px tall, spans tiles, 4-conn friendly
        Img u{"U_123_on_9_40x40", 40, 40, std::vector<uint32_t>(1600, 9)};
        for (int y = 3; y < 35; y++) { u.v[y * 40 + 5] = 123; u.v[y * 40 + 33] = 123; }
        for (int x = 5; x <= 33; x++) u.v[34 * 40 + x] = 123;
        ims.push_back(u);
    }
    { // serpentine, value 66 in 180 background
        int W = 37, H = 29; Img s{"snake_66_180_37x29", W, H, std::vector<uint32_t>(W * H, 180)};
        for (int y = 1; y < H - 1; y += 2) for (int x = 1; x < W - 1; x++) s.v[y * W + x] = 66;
        for (int y = 1; y < H - 2; y += 2) { int x = ((y / 2) & 1) ? 1 : W - 2; s.v[(y + 1) * W + x] = 66; }
        ims.push_back(s);
    }
    { // diagonal staircase connected only through corners (8-conn one region), value 5 in 250
        int W = 50, H = 50; Img d{"diag_5_250_50x50", W, H, std::vector<uint32_t>(W * H, 250)};
        for (int i = 0; i < 50; i++) d.v[i * W + i] = 5;
        for (int i = 0; i < 49; i++) d.v[i * W + 49 - i] = 5;
        ims.push_back(d);
    }
    for (int k = 0; k < 6; k++) { // random 3-value images
        int W = k < 3 ? 64 : 200, H = k < 3 ? 64 : 150; std::mt19937 r(91 + k);
        Img im{"rand3_17_130_240_" + std::to_string(W) + "x" + std::to_string(H) + "_s" + std::to_string(k), W, H, std::vector<uint32_t>(W * H)};
        const uint32_t vals[3] = {17, 130, 240};
        for (auto& p : im.v) p = vals[r() % 3];
        ims.push_back(im);
    }
    NppStreamContext ctx = make_ctx();
    int bad = 0, total = 0;
    for (auto& im : ims) for (int conn : {8, 4}) for (int depth : {8, 16, 32}) for (int comp : {0, 1}) {
        std::vector<uint32_t> lab = depth == 8 ? run<uint8_t>(im, conn, 8, comp, ctx) : depth == 16 ? run<uint16_t>(im, conn, 16, comp, ctx) : run<uint32_t>(im, conn, 32, comp, ctx);
        int sp, mg, nt, nl; judge(im, conn, lab, &sp, &mg, &nt, &nl);
        total++; bad += (sp || mg);
        printf("%-34s conn%d %2du %s truth %5d labels %5d split-regions %4d merged-labels %3d %s\n", im.name.c_str(), conn, depth, comp ? "compressed" : "raw       ", nt, nl, sp, mg, (sp || mg) ? "WRONG" : "ok");
        if (im.W == 3 && depth == 8 && !comp) { for (int y = 0; y < 3; y++) printf("      %u %u %u\n", lab[3*y], lab[3*y+1], lab[3*y+2]); }
    }
    printf("TOTAL %d wrong of %d\n", bad, total);
    return 0;
}
