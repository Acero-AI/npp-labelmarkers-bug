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
    const NppLibraryVersion* lv = nppGetLibVersion(); printf("NPP %d.%d.%d\n", lv->major, lv->minor, lv->build);
    Img im{"t184854", 8, 6, {0,0,0,0,0,0,0,0, 0,0,0,0,0,1,0,0, 0,1,1,1,1,1,0,0, 0,1,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0}};
    NppStreamContext ctx = make_ctx();
    for (int conn : {4, 8}) for (int c : {0, 1}) { auto lab = run<uint16_t>(im, conn, 16, c, ctx); int sp, mg, nt, nl; judge(im, conn, lab, &sp, &mg, &nt, &nl);
      printf("conn%d %s truth %d labels %d splits %d merges %d %s\n", conn, c ? "compressed" : "raw", nt, nl, sp, mg, sp||mg ? "WRONG" : "ok"); }
}
