// probe: when does nppiLabelMarkersUF fail?  Reuses npp_debug.cu's loader, run() and check().
//     nvcc -std=c++17 -O2 -I<npp include> probe.cu -ldl -o probe && ./probe LIB_DIR
// T1  same input, 50 runs: distinct answers vs image size (random 2-value images; and one 3x3 pattern per 16x16 tile)
// T2  the 3 x 3 alone, 200 runs: how many distinct answers
// T3  the 3 x 3 with other values in place of 0/1/2 (8u and 32u input)
// T4  the 3 x 3 at (2,2) on a background of 9, image width 3..80 and height 3..40
// T5  single-image vs batch API on T1's images
#define main npp_debug_main
#include "npp_debug.cu"
#undef main
#include <unordered_set>

static size_t hsh(const std::vector<uint32_t>& v) { size_t h = 1469598103934665603ull; for (uint32_t x : v) h = (h ^ x) * 1099511628211ull; return h; }
static const uint32_t PAT[9] = {0, 2, 0, 0, 2, 0, 1, 0, 1};
static Img tiled(int S) {  // the 3 x 3 pattern at (2,2) of every 16 x 16 tile, background 9
    Img im{S, S, std::vector<uint32_t>((size_t)S * S, 9)};
    for (int ty = 0; ty + 16 <= S; ty += 16) for (int tx = 0; tx + 16 <= S; tx += 16)
        for (int k = 0; k < 9; k++) im.v[(ty + 2 + k / 3) * S + tx + 2 + k % 3] = PAT[k];
    return im;
}
static void repeat(const char* what, const Img& im, How h, int runs) {
    std::unordered_set<size_t> seen; int wrong = 0, lo = 1 << 30, hi = 0, nref = 0;
    for (int r = 0; r < runs; r++) { auto lab = run(im, h); seen.insert(hsh(lab)); Verdict v = check(im, h.conn, lab); nref = v.n_ref;
        wrong += v.broken || v.merges; lo = std::min(lo, v.n_npp); hi = std::max(hi, v.n_npp); }
    printf("  %-34s %4dx%-4d: wrong %3d/%d, distinct answers %3zu, labels %d-%d (truth %d)\n", what, im.W, im.H, wrong, runs, seen.size(), lo, hi, nref);
}

int main(int argc, char** argv) {
    load(argv[1], "libnppc");
    void* f = load(argv[1], "libnppif");
    getBuf = sym<GetBuf>(f, "nppiLabelMarkersUFGetBufferSize_32u_C1R");
    uf8 = sym<UF8>(f, "nppiLabelMarkersUF_8u32u_C1R_Ctx");
    uf32 = sym<UF32>(f, "nppiLabelMarkersUF_32u_C1R_Ctx");
    batch8 = sym<Batch>(f, "nppiLabelMarkersUFBatch_8u32u_C1R_Advanced_Ctx");
    for (const char* v : {"13", "12", "11"}) {
        void* c = dlopen((std::string(argv[1]) + "/libnppc.so." + v).c_str(), RTLD_NOW);
        if (c) { if (auto gv = (const NppLibraryVersion* (*)())dlsym(c, "nppGetLibVersion")) { auto lv = gv(); printf("NPP %d.%d.%d\n", lv->major, lv->minor, lv->build); } break; }
    }
    ctx.hStream = 0;  // as NVIDIA's sample fills it
    CK(cudaGetDevice(&ctx.nCudaDeviceId));
    CK(cudaDeviceGetAttribute(&ctx.nCudaDevAttrComputeCapabilityMajor, cudaDevAttrComputeCapabilityMajor, ctx.nCudaDeviceId));
    CK(cudaDeviceGetAttribute(&ctx.nCudaDevAttrComputeCapabilityMinor, cudaDevAttrComputeCapabilityMinor, ctx.nCudaDeviceId));
    CK(cudaStreamGetFlags(ctx.hStream, &ctx.nStreamFlags));
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, ctx.nCudaDeviceId));
    ctx.nMultiProcessorCount = p.multiProcessorCount; ctx.nMaxThreadsPerMultiProcessor = p.maxThreadsPerMultiProcessor;
    ctx.nMaxThreadsPerBlock = p.maxThreadsPerBlock; ctx.nSharedMemPerBlock = p.sharedMemPerBlock;
    printf("GPU %s (sm_%d%d)\n", p.name, p.major, p.minor);
    How h8; h8.depth = 8; How hb; hb.depth = 0;

    if (argc > 2 && std::string(argv[2]) == "--batch-map") {  // T6
        for (int api = 0; api < 2; api++) {
            printf("\nT6. %s: the 3 x 3 at every offset (x0, y0) of a 70 x 70 image of 9s, '#' = wrong; then a V with arms 1..6 tall\n", api ? "batch 8u32u_Advanced" : "single 8u32u");
            int ok = 0;
            for (int y0 = 0; y0 <= 67; y0++) { printf("  %2d ", y0);
                for (int x0 = 0; x0 <= 67; x0++) { Img im{70, 70, std::vector<uint32_t>(4900, 9)};
                    for (int k = 0; k < 9; k++) im.v[(y0 + k / 3) * 70 + x0 + k % 3] = PAT[k];
                    Verdict v = check(im, 8, run(im, api ? hb : h8)); bool w = v.broken || v.merges; ok += !w; printf("%c", w ? '#' : '.'); }
                printf("\n"); }
            printf("  correct offsets: %d of %d\n", ok, 68 * 68);
            // V of 0s on 9s: left arm height a, right arm height b, joined at the bottom middle, at (2,2) of a 64 x 64 image
            for (int a = 1; a <= 6; a++) { printf("  left arm %d:", a);
                for (int b = 1; b <= 6; b++) { Img im{64, 64, std::vector<uint32_t>(4096, 9)}; int base = 2 + 6;
                    for (int k = 0; k < a; k++) im.v[(base - 1 - k) * 64 + 2] = 0;
                    for (int k = 0; k < b; k++) im.v[(base - 1 - k) * 64 + 4] = 0;
                    im.v[base * 64 + 3] = 0;
                    Verdict v = check(im, 8, run(im, api ? hb : h8)); printf(" %c", v.broken || v.merges ? 'X' : '.'); }
                printf("   (right arm 1..6)\n"); }
        }
        return 0;
    }
    printf("\nT1. same input, 50 runs each (8u, 8 neighbours)\n");
    std::mt19937 g(7);
    for (int S : {8, 12, 16, 17, 20, 24, 32, 48, 64, 96, 128, 256, 512, 1024}) repeat("random, 2 values", random_img(S, S, 2, g), h8, 50);
    for (int S : {16, 32, 48, 64, 128, 256, 512, 1024}) repeat("3x3 pattern in every 16x16 tile", tiled(S), h8, 50);

    printf("\nT2. the 3 x 3 image alone, 200 runs\n");
    Img tiny{3, 3, std::vector<uint32_t>(PAT, PAT + 9)};
    repeat("3x3, 8u", tiny, h8, 200);

    printf("\nT3. the 3 x 3 with other values: (region of five, column, corners)\n");
    std::vector<std::vector<uint32_t>> sets = {{0, 2, 1}, {0, 1, 2}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}, {5, 9, 7}, {255, 0, 128}, {200, 100, 50}, {7, 7, 3}, {7, 3, 3}, {0, 1, 1}};
    for (auto s : sets) {
        Img im{3, 3, std::vector<uint32_t>(9)};
        for (int k = 0; k < 9; k++) im.v[k] = PAT[k] == 0 ? s[0] : PAT[k] == 2 ? s[1] : s[2];
        char b[64]; snprintf(b, 64, "five=%u column=%u corners=%u", s[0], s[1], s[2]); repeat(b, im, h8, 20);
    }
    for (auto s : std::vector<std::vector<uint32_t>>{{0, 2, 1}, {70000, 70002, 70001}, {4000000000u, 1, 2}}) {
        Img im{3, 3, std::vector<uint32_t>(9)};
        for (int k = 0; k < 9; k++) im.v[k] = PAT[k] == 0 ? s[0] : PAT[k] == 2 ? s[1] : s[2];
        How h32; h32.depth = 32; char b[64]; snprintf(b, 64, "32u five=%u column=%u corners=%u", s[0], s[1], s[2]); repeat(b, im, h32, 20);
    }

    printf("\nT4. the 3 x 3 at (2,2) on background 9, by image size: '#' = wrong (1 run each)\n     width 3..80 ->\n");
    for (int H : {3, 4, 5, 6, 8, 16, 17, 32, 40}) {
        printf("  H=%2d ", H);
        for (int W = 3; W <= 80; W++) {
            if (W < 5 || H < 5) { printf(" "); continue; }
            Img im{H, W, std::vector<uint32_t>((size_t)H * W, 9)};
            for (int k = 0; k < 9; k++) im.v[(2 + k / 3) * W + 2 + k % 3] = PAT[k];
            Verdict v = check(im, 8, run(im, h8)); printf("%c", v.broken || v.merges ? '#' : '.');
        }
        printf("\n");
    }

    printf("\nT5. batch API (8u32u_Advanced, one image), same images as T1, 20 runs\n");
    std::mt19937 g2(7);
    for (int S : {8, 16, 32, 64, 128, 256, 512}) repeat("batch, random 2 values", random_img(S, S, 2, g2), hb, 20);
    for (int S : {16, 64, 256}) repeat("batch, 3x3 in every tile", tiled(S), hb, 20);
    repeat("batch, 3x3 alone", tiny, hb, 20);
    return 0;
}
