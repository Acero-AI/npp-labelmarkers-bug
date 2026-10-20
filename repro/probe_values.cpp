// probe_values: label one small 8-bit image given on the command line and print NPP's raw labels.
//
//     probe_values W H CONN v0 v1 ...      (CONN 8 = nppiNormInf, 4 = nppiNormL1)
//
// run_clean.sh uses it for: the 3 x 3 with the values 5/9/7 instead of 0/1/2; 0s labelled like any other value;
// and a 4 x 2 image whose row 0 ends with the value row 1 starts with, and nothing else touches (NPP 11.8.0 joins
// them across the row ends).
//
//     g++ -O2 probe_values.cpp -I<npp include> -I<cudart include> -L<libs> -l:libnppif.so.N -l:libnppc.so.N -l:libcudart.so.N
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <npp.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s, line %d\n", cudaGetErrorString(e), __LINE__); return 1; } } while (0)

int main(int argc, char** argv) {
    if (argc < 4) { printf("usage: probe_values W H CONN v0 v1 ...\n"); return 1; }
    const int W = atoi(argv[1]), H = atoi(argv[2]), conn = atoi(argv[3]);
    if (argc != 4 + W * H) { printf("expected %d pixel values\n", W * H); return 1; }
    std::vector<Npp8u> img(W * H);
    for (int i = 0; i < W * H; i++) img[i] = atoi(argv[4 + i]);
    NppStreamContext ctx = {};
    ctx.hStream = 0;
    CK(cudaGetDevice(&ctx.nCudaDeviceId));
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, ctx.nCudaDeviceId));
    ctx.nCudaDevAttrComputeCapabilityMajor = p.major;
    ctx.nCudaDevAttrComputeCapabilityMinor = p.minor;
    ctx.nMultiProcessorCount = p.multiProcessorCount;
    ctx.nMaxThreadsPerMultiProcessor = p.maxThreadsPerMultiProcessor;
    ctx.nMaxThreadsPerBlock = p.maxThreadsPerBlock;
    ctx.nSharedMemPerBlock = p.sharedMemPerBlock;
    CK(cudaStreamGetFlags(ctx.hStream, &ctx.nStreamFlags));
    NppiSize roi = {W, H};
    int bytes = 0;
    if (nppiLabelMarkersUFGetBufferSize_32u_C1R(roi, &bytes) != NPP_SUCCESS) { printf("buffer size failed\n"); return 1; }
    Npp8u *src, *buf;
    Npp32u* dst;
    CK(cudaMalloc((void**)&src, W * H));
    CK(cudaMalloc((void**)&dst, 4 * W * H));
    CK(cudaMalloc((void**)&buf, bytes));
    CK(cudaMemset(buf, 0, bytes));
    CK(cudaMemcpy(src, img.data(), W * H, cudaMemcpyHostToDevice));
    NppStatus st = nppiLabelMarkersUF_8u32u_C1R_Ctx(src, W, dst, 4 * W, roi, conn == 8 ? nppiNormInf : nppiNormL1, buf, ctx);
    CK(cudaDeviceSynchronize());
    std::vector<Npp32u> lab(W * H);
    CK(cudaMemcpy(lab.data(), dst, 4 * W * H, cudaMemcpyDeviceToHost));
    const NppLibraryVersion* v = nppGetLibVersion();
    printf("NPP %d.%d.%d, %d neighbours, status %d\n", v->major, v->minor, v->build, conn, st);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) printf("%4d", img[y * W + x]);
        printf("    |");
        for (int x = 0; x < W; x++) printf("%4u", lab[y * W + x]);
        printf("\n");
    }
    return st == NPP_SUCCESS ? 0 : 1;
}
