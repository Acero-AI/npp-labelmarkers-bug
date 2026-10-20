// npp_min: the smallest input on which nppiLabelMarkersUF gives a wrong answer (NPP 12.4.1, 13.x).
//
//     nvcc -O2 npp_min.cu -lnppif -lnppc -o npp_min && ./npp_min
// It has no GPU code of its own, so it also builds with g++ against any NPP and CUDA runtime (see run_clean.sh):
//     g++ -O2 -x c++ npp_min.cu -I<npp include> -I<cudart include> -L<libs> -l:libnppif.so.N -l:libnppc.so.N -l:libcudart.so.N
//
// The image is 3 x 3, 8 neighbours:      0 2 0
//                                        0 2 0
//                                        1 0 1
// The five 0s touch (the bottom-middle 0 touches both top 0s through corners), so there are four
// regions: the 0s, the 2s, and each 1. NPP is called the way NVIDIA's sample
// batchedLabelMarkersAndLabelCompressionNPP calls it: 8-bit input, the NULL stream, and a stream
// context filled from the device properties. Unlike the sample, the scratch buffer is zeroed; that
// makes no difference.
#include <cstdio>
#include <set>
#include <cuda_runtime.h>
#include <npp.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s, line %d\n", cudaGetErrorString(e), __LINE__); return 1; } } while (0)

int main() {
    const int W = 3, H = 3;
    const Npp8u img[W * H] = {0, 2, 0,
                              0, 2, 0,
                              1, 0, 1};
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

    NppiSize roi = {W, H};
    Npp8u *src, *scratch;
    Npp32u* dst;
    int bytes = 0;
    if (nppiLabelMarkersUFGetBufferSize_32u_C1R(roi, &bytes) != NPP_SUCCESS) { printf("buffer size failed\n"); return 1; }
    CK(cudaMalloc((void**)&src, W * H));
    CK(cudaMalloc((void**)&dst, W * H * sizeof(Npp32u)));
    CK(cudaMalloc((void**)&scratch, bytes));
    CK(cudaMemset(scratch, 0, bytes));
    CK(cudaMemcpy(src, img, W * H, cudaMemcpyHostToDevice));
    NppStatus st = nppiLabelMarkersUF_8u32u_C1R_Ctx(src, W, dst, W * sizeof(Npp32u), roi, nppiNormInf, scratch, ctx);
    CK(cudaDeviceSynchronize());
    if (st != NPP_SUCCESS) { printf("NPP status %d\n", st); return 1; }
    Npp32u lab[W * H];
    CK(cudaMemcpy(lab, dst, sizeof lab, cudaMemcpyDeviceToHost));

    const NppLibraryVersion* v = nppGetLibVersion();
    int rt = 0, drv = 0;
    CK(cudaRuntimeGetVersion(&rt));
    CK(cudaDriverGetVersion(&drv));
    printf("NPP %d.%d.%d, %s (sm_%d%d), CUDA runtime %d, driver supports %d\n\n", v->major, v->minor, v->build, p.name, p.major, p.minor, rt, drv);
    for (int y = 0; y < H; y++) printf("   %d %d %d    |   %u %u %u\n", img[3 * y], img[3 * y + 1], img[3 * y + 2], lab[3 * y], lab[3 * y + 1], lab[3 * y + 2]);
    std::set<Npp32u> distinct(lab, lab + W * H);
    // Pixels 0, 3, 7, 2 and 5 are the five 0s and must share one label; pixels 1 and 4 are the 2s and must share
    // another; pixels 6 and 8 are the 1s, each a region of its own. Four regions, four different labels.
    const bool zeros = lab[0] == lab[3] && lab[3] == lab[7] && lab[7] == lab[2] && lab[2] == lab[5];
    const bool twos = lab[1] == lab[4];
    const std::set<Npp32u> four = {lab[0], lab[1], lab[6], lab[8]};
    const bool ok = zeros && twos && four.size() == 4 && distinct.size() == 4;
    printf("\nexpected 4 regions, NPP gave %zu: %s\n", distinct.size(), ok ? "correct" : "WRONG");
    return ok ? 0 : 2;
}
