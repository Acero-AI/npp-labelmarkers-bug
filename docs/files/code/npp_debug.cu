// npp_debug: is nppiLabelMarkersUF wrong, or are we calling it wrong?
//
// Plain CUDA, no PyTorch. NPP is loaded at run time from a given folder, so the
// same binary tests NPP from CUDA 11, 12 and 13:
//
//     ./npp_debug /path/to/folder/with/libnppif   [quick]
//
// Every call is checked against a CPU union-find (ground truth). A wrong answer
// is reported as "broken" links (two neighbouring equal pixels with different
// labels) and "false merges" (one NPP label covering two true components).
//
// Sections:
//   A  the 3 x 3 image, under every way of calling NPP we could vary
//   B  random images: sizes x number of values x connectivity, called exactly as
//      NVIDIA's sample batchedLabelMarkersAndLabelCompressionNPP calls it
//   C  where the broken links are: direction and position modulo tile sizes
//   D  the 3 x 3 pattern moved to every offset in a 70 x 70 image
//   E  the same image 50 times: how many different answers
//   F  entry points: 8u32u / 16u32u / 32u / batch 8u32u Advanced
//
//     ./npp_debug LIB_DIR --raw FILE W H     NVIDIA's own sample images, 30 runs each
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <nppdefs.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

typedef NppStatus (*GetBuf)(NppiSize, int*);
typedef NppStatus (*UF8)(Npp8u*, int, Npp32u*, int, NppiSize, NppiNorm, Npp8u*, NppStreamContext);
typedef NppStatus (*UF16)(Npp16u*, int, Npp32u*, int, NppiSize, NppiNorm, Npp8u*, NppStreamContext);
typedef NppStatus (*UF32)(Npp32u*, int, Npp32u*, int, NppiSize, NppiNorm, Npp8u*, NppStreamContext);
typedef NppStatus (*Batch)(const NppiImageDescriptor*, NppiImageDescriptor*, int, NppiSize, NppiNorm, NppStreamContext);
typedef NppStatus (*GetVer)();
typedef NppStatus (*CmpBuf)(int, int*);
typedef NppStatus (*Cmp)(Npp32u*, int, NppiSize, int, int*, Npp8u*, NppStreamContext);
typedef NppStatus (*GetCtx)(NppStreamContext*);

static GetBuf getBuf;
static UF8 uf8;
static UF16 uf16;
static UF32 uf32;
static Batch batch8;
static CmpBuf cmpBuf;
static Cmp cmp;
static GetCtx getCtx;
static NppStreamContext ctx;

static void* load(const std::string& dir, const char* base) {
    for (const char* v : {"13", "12", "11", ""}) {
        std::string p = dir + "/" + base + (*v ? std::string(".so.") + v : std::string(".so"));
        if (void* h = dlopen(p.c_str(), RTLD_NOW | RTLD_GLOBAL)) { printf("loaded %s\n", p.c_str()); return h; }
    }
    printf("cannot load %s from %s: %s\n", base, dir.c_str(), dlerror());
    exit(1);
}

template <class F> static F sym(void* h, const char* name) {
    F f = (F)dlsym(h, name);
    if (!f) printf("missing %s\n", name);
    return f;
}

// ---------- ground truth ----------
struct Img { int H, W; std::vector<uint32_t> v; };

static int find(std::vector<int>& p, int i) { while (p[i] != i) i = p[i] = p[p[i]]; return i; }

// Reference labels 0..n-1 in raster order of first pixel.
static std::vector<int> reference(const Img& im, int conn, int* n) {
    int N = im.H * im.W;
    std::vector<int> p(N);
    std::iota(p.begin(), p.end(), 0);
    auto join = [&](int a, int b) { a = find(p, a); b = find(p, b); if (a != b) p[std::max(a, b)] = std::min(a, b); };
    for (int y = 0; y < im.H; y++)
        for (int x = 0; x < im.W; x++) {
            int i = y * im.W + x;
            if (x + 1 < im.W && im.v[i] == im.v[i + 1]) join(i, i + 1);
            if (y + 1 < im.H && im.v[i] == im.v[i + im.W]) join(i, i + im.W);
            if (conn == 8 && y + 1 < im.H) {
                if (x + 1 < im.W && im.v[i] == im.v[i + im.W + 1]) join(i, i + im.W + 1);
                if (x > 0 && im.v[i] == im.v[i + im.W - 1]) join(i, i + im.W - 1);
            }
        }
    std::vector<int> lab(N);
    std::map<int, int> id;
    for (int i = 0; i < N; i++) { int r = find(p, i); auto it = id.find(r); if (it == id.end()) it = id.emplace(r, (int)id.size()).first; lab[i] = it->second; }
    *n = (int)id.size();
    return lab;
}

struct Broken { int y, x, dir; };  // dir: 0 right, 1 down, 2 down-right, 3 down-left
struct Verdict { int n_ref, n_npp, broken, merges; std::vector<Broken> where; };

static Verdict check(const Img& im, int conn, const std::vector<uint32_t>& got) {
    Verdict r{};
    std::vector<int> ref = reference(im, conn, &r.n_ref);
    std::set<uint32_t> distinct(got.begin(), got.end());
    r.n_npp = (int)distinct.size();
    const int dx[4] = {1, 0, 1, -1}, dy[4] = {0, 1, 1, 1};
    for (int y = 0; y < im.H; y++)
        for (int x = 0; x < im.W; x++)
            for (int d = 0; d < (conn == 8 ? 4 : 2); d++) {
                int X = x + dx[d], Y = y + dy[d];
                if (X < 0 || X >= im.W || Y >= im.H) continue;
                int i = y * im.W + x, j = Y * im.W + X;
                if (im.v[i] == im.v[j] && got[i] != got[j]) { r.broken++; r.where.push_back({y, x, d}); }
            }
    std::map<uint32_t, int> owner;  // one NPP label must not cover two true components
    for (size_t i = 0; i < got.size(); i++) {
        auto it = owner.find(got[i]);
        if (it == owner.end()) owner[got[i]] = ref[i];
        else if (it->second != ref[i]) { r.merges++; it->second = ref[i]; }
    }
    return r;
}

// ---------- calling NPP ----------
struct How {
    int depth = 32;         // 8, 16, 32 (input type), or 0 for the batch API (8-bit input)
    int scratch_fill = 0;   // -1 leave as allocated, else memset value
    int dst_fill = -1;      // -1 leave, else memset value
    int conn = 8;
    bool own_stream = false;
};

static std::vector<uint32_t> run(const Img& im, const How& h) {
    NppiSize roi{im.W, im.H};
    size_t N = (size_t)im.H * im.W;
    int bpp = h.depth == 0 ? 1 : h.depth / 8;
    void* src; Npp32u* dst; Npp8u* scratch = nullptr;
    CK(cudaMalloc(&src, N * bpp));
    CK(cudaMalloc((void**)&dst, N * 4));
    std::vector<uint8_t> host(N * bpp);
    for (size_t i = 0; i < N; i++) {
        uint32_t v = im.v[i];
        if (bpp == 1) host[i] = (uint8_t)v; else if (bpp == 2) ((uint16_t*)host.data())[i] = (uint16_t)v; else ((uint32_t*)host.data())[i] = v;
    }
    CK(cudaMemcpy(src, host.data(), N * bpp, cudaMemcpyHostToDevice));
    if (h.dst_fill >= 0) CK(cudaMemset(dst, h.dst_fill, N * 4));
    NppStreamContext c = ctx;
    cudaStream_t s = 0;
    if (h.own_stream) { CK(cudaStreamCreate(&s)); c.hStream = s; CK(cudaStreamGetFlags(s, &c.nStreamFlags)); }
    NppStatus st;
    NppiNorm norm = h.conn == 8 ? nppiNormInf : nppiNormL1;
    if (h.depth == 0) {
        NppiImageDescriptor hs{src, im.W, roi}, hd{dst, im.W * 4, roi}, *ds, *dd;
        CK(cudaMalloc((void**)&ds, sizeof hs)); CK(cudaMalloc((void**)&dd, sizeof hd));
        CK(cudaMemcpy(ds, &hs, sizeof hs, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dd, &hd, sizeof hd, cudaMemcpyHostToDevice));
        st = batch8(ds, dd, 1, roi, norm, c);
        CK(cudaFree(ds)); CK(cudaFree(dd));
    } else {
        int bytes = 0;
        getBuf(roi, &bytes);
        CK(cudaMalloc((void**)&scratch, bytes));
        if (h.scratch_fill >= 0) CK(cudaMemset(scratch, h.scratch_fill, bytes));
        if (h.depth == 8) st = uf8((Npp8u*)src, im.W, dst, im.W * 4, roi, norm, scratch, c);
        else if (h.depth == 16) st = uf16((Npp16u*)src, im.W * 2, dst, im.W * 4, roi, norm, scratch, c);
        else st = uf32((Npp32u*)src, im.W * 4, dst, im.W * 4, roi, norm, scratch, c);
    }
    CK(cudaDeviceSynchronize());
    if (st != NPP_SUCCESS) { printf("NPP status %d\n", st); exit(1); }
    std::vector<uint32_t> out(N);
    CK(cudaMemcpy(out.data(), dst, N * 4, cudaMemcpyDeviceToHost));
    CK(cudaFree(src)); CK(cudaFree(dst));
    if (scratch) CK(cudaFree(scratch));
    if (h.own_stream) CK(cudaStreamDestroy(s));
    return out;
}

static Img random_img(int H, int W, int values, std::mt19937& g) {
    Img im{H, W, std::vector<uint32_t>((size_t)H * W)};
    for (auto& v : im.v) v = g() % values;
    return im;
}

static void print_grid(const Img& im, const std::vector<uint32_t>& lab) {
    for (int y = 0; y < im.H; y++) {
        printf("   ");
        for (int x = 0; x < im.W; x++) printf("%u ", im.v[y * im.W + x]);
        printf("  |  ");
        for (int x = 0; x < im.W; x++) printf("%3u ", lab[y * im.W + x]);
        printf("\n");
    }
}

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: %s NPP_LIB_DIR [quick]\n", argv[0]); return 1; }
    bool quick = argc > 2;
    load(argv[1], "libnppc");
    void* f = load(argv[1], "libnppif");
    getBuf = sym<GetBuf>(f, "nppiLabelMarkersUFGetBufferSize_32u_C1R");
    uf8 = sym<UF8>(f, "nppiLabelMarkersUF_8u32u_C1R_Ctx");
    uf16 = sym<UF16>(f, "nppiLabelMarkersUF_16u32u_C1R_Ctx");
    uf32 = sym<UF32>(f, "nppiLabelMarkersUF_32u_C1R_Ctx");
    batch8 = sym<Batch>(f, "nppiLabelMarkersUFBatch_8u32u_C1R_Advanced_Ctx");
    cmpBuf = (CmpBuf)dlsym(f, "nppiCompressMarkerLabelsGetBufferSize_32u_C1R");
    cmp = (Cmp)dlsym(f, "nppiCompressMarkerLabelsUF_32u_C1IR_Ctx");
    void* c = dlopen((std::string(argv[1]) + "/libnppc.so.13").c_str(), RTLD_NOW);
    for (const char* v : {"13", "12", "11"}) if (!c) c = dlopen((std::string(argv[1]) + "/libnppc.so." + v).c_str(), RTLD_NOW);
    if (c) if (auto gv = (const NppLibraryVersion* (*)())dlsym(c, "nppGetLibVersion")) { auto lv = gv(); printf("NPP %d.%d.%d\n", lv->major, lv->minor, lv->build); }
    if (c) getCtx = (GetCtx)dlsym(c, "nppGetStreamContext");

    // Stream context filled exactly as NVIDIA's sample does.
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
    int rt; cudaRuntimeGetVersion(&rt); int drv; cudaDriverGetVersion(&drv);
    printf("GPU %s (sm_%d%d), runtime %d, driver %d\n\n", p.name, p.major, p.minor, rt, drv);

    if (argc > 6 && std::string(argv[2]) == "--check") {
        // ./npp_debug LIB --check IN_8u.raw W H LABELS_32u.raw CONN : judge a label file written by anything
        int W = atoi(argv[4]), H = atoi(argv[5]), conn = argc > 7 ? atoi(argv[7]) : 8;
        std::vector<uint8_t> b((size_t)W * H); std::vector<uint32_t> lab((size_t)W * H);
        FILE* fp = fopen(argv[3], "rb"); if (!fp || fread(b.data(), 1, b.size(), fp) != b.size()) { printf("cannot read %s\n", argv[3]); return 1; } fclose(fp);
        fp = fopen(argv[6], "rb"); if (!fp || fread(lab.data(), 4, lab.size(), fp) != lab.size()) { printf("cannot read %s\n", argv[6]); return 1; } fclose(fp);
        Img im{H, W, std::vector<uint32_t>(b.begin(), b.end())};
        auto v = check(im, conn, lab);
        printf("%s: truth %d, file has %d labels, broken links %d, false merges %d -> %s\n", argv[6], v.n_ref, v.n_npp, v.broken, v.merges, v.broken || v.merges ? "WRONG" : "correct");
        return 0;
    }
    if (argc > 2 && std::string(argv[2]) == "--hyp") {
        // Ways we might be calling NPP wrong, each tested on: the 3x3 image, 20 random 64x64 two-value
        // images, and NVIDIA's PCB sample if data/PCB_1280x720_8u.raw exists.
        std::vector<std::pair<std::string, Img>> imgs;
        imgs.push_back({"3x3", Img{3, 3, {0, 2, 0, 0, 2, 0, 1, 0, 1}}});
        std::mt19937 gh(5);
        for (int t = 0; t < 20; t++) imgs.push_back({"rand64", random_img(64, 64, 2, gh)});
        if (FILE* fp = fopen("data/PCB_1280x720_8u.raw", "rb")) {
            std::vector<uint8_t> b(1280 * 720); fread(b.data(), 1, b.size(), fp); fclose(fp);
            imgs.push_back({"pcb", Img{720, 1280, std::vector<uint32_t>(b.begin(), b.end())}});
        }
        enum { PLAIN, BUF4, BUF16, CTX_ZERO, CTX_GET, STEP_ELEMS, SRC_PITCHED, COMPRESS, VAL255 };
        const char* names[] = {"as before (baseline)", "scratch buffer x4", "scratch buffer x16", "stream context all zero but stream",
                               "stream context from nppGetStreamContext", "row steps in elements, not bytes", "source rows padded (cudaMallocPitch)",
                               "after nppiCompressMarkerLabelsUF", "values mapped to 0/255 (8u input)"};
        for (int hyp = 0; hyp <= VAL255; hyp++) {
            if (hyp == CTX_GET && !getCtx) { printf("  %-44s (nppGetStreamContext not in this NPP)\n", names[hyp]); continue; }
            if (hyp == COMPRESS && !cmp) { printf("  %-44s (compress not found)\n", names[hyp]); continue; }
            int wrong[3] = {0, 0, 0}, total[3] = {0, 0, 0}; std::string err;
            for (auto& [kind, im0] : imgs) {
                Img im = im0;
                int K = kind == "3x3" ? 0 : kind == "rand64" ? 1 : 2;
                if (hyp == VAL255) for (auto& v : im.v) v = v ? 255 : 0;
                NppiSize roi{im.W, im.H}; size_t N = (size_t)im.W * im.H;
                bool u8 = hyp == VAL255;
                void* src; size_t sstep = (size_t)im.W * (u8 ? 1 : 4);
                if (hyp == SRC_PITCHED) CK(cudaMallocPitch(&src, &sstep, im.W * 4 + 64, im.H)); else CK(cudaMalloc(&src, N * (u8 ? 1 : 4)));
                std::vector<uint8_t> h8(N); for (size_t i = 0; i < N; i++) h8[i] = (uint8_t)im.v[i];
                CK(cudaMemcpy2D(src, sstep, u8 ? (void*)h8.data() : (void*)im.v.data(), im.W * (u8 ? 1 : 4), im.W * (u8 ? 1 : 4), im.H, cudaMemcpyHostToDevice));
                Npp32u* dst; CK(cudaMalloc((void**)&dst, N * 4));
                int bytes = 0; getBuf(roi, &bytes);
                size_t bsz = (size_t)bytes * (hyp == BUF4 ? 4 : hyp == BUF16 ? 16 : 1);
                Npp8u* scratch; CK(cudaMalloc((void**)&scratch, bsz)); CK(cudaMemset(scratch, 0, bsz));
                NppStreamContext c = ctx;
                if (hyp == CTX_ZERO) { memset(&c, 0, sizeof c); c.hStream = 0; }
                if (hyp == CTX_GET) getCtx(&c);
                int ss = hyp == STEP_ELEMS ? im.W : (int)sstep, ds = hyp == STEP_ELEMS ? im.W : im.W * 4;
                NppStatus st = u8 ? uf8((Npp8u*)src, ss, dst, ds, roi, nppiNormInf, scratch, c)
                                  : uf32((Npp32u*)src, ss, dst, ds, roi, nppiNormInf, scratch, c);
                CK(cudaDeviceSynchronize());
                if (st == NPP_SUCCESS && hyp == COMPRESS) {
                    int cb = 0; cmpBuf((int)N, &cb); Npp8u* cbuf; CK(cudaMalloc((void**)&cbuf, cb)); int n = 0;
                    st = cmp(dst, im.W * 4, roi, (int)N, &n, cbuf, c); CK(cudaDeviceSynchronize()); CK(cudaFree(cbuf));
                }
                if (st != NPP_SUCCESS) { err = "NPP status " + std::to_string(st); }
                else {
                    std::vector<uint32_t> lab(N); CK(cudaMemcpy(lab.data(), dst, N * 4, cudaMemcpyDeviceToHost));
                    auto v = check(im, 8, lab); wrong[K] += v.broken || v.merges; total[K]++;
                }
                CK(cudaFree(src)); CK(cudaFree(dst)); CK(cudaFree(scratch));
            }
            printf("  %-44s 3x3: %s   random 64x64: %d/%d wrong   NVIDIA PCB: %s %s\n", names[hyp],
                   total[0] ? (wrong[0] ? "WRONG" : "ok") : "-", wrong[1], total[1], total[2] ? (wrong[2] ? "WRONG" : "ok") : "-", err.c_str());
        }
        // Are 4-way and 8-way swapped? Compare NPP's 8-way answer with 4-way truth and vice versa.
        Img v3{3, 3, {0, 2, 0, 0, 2, 0, 1, 0, 1}};
        for (int conn : {8, 4}) {
            auto lab = run(v3, {32, 0, -1, conn, false});
            auto as8 = check(v3, 8, lab), as4 = check(v3, 4, lab);
            printf("  NPP %d-way on 3x3 gives %d labels; 8-way truth %d (%s), 4-way truth %d (%s)\n", conn, as8.n_npp, as8.n_ref,
                   as8.broken || as8.merges ? "mismatch" : "match", as4.n_ref, as4.broken || as4.merges ? "mismatch" : "match");
        }
        return 0;
    }
    if (argc > 7 && std::string(argv[2]) == "--dump") {
        // ./npp_debug LIB --dump IN.u32 W H CONN OUT.u32 : NPP's raw labels for a uint32 image
        int W = atoi(argv[4]), H = atoi(argv[5]), conn = atoi(argv[6]);
        Img im{H, W, std::vector<uint32_t>((size_t)W * H)};
        FILE* fp = fopen(argv[3], "rb");
        if (!fp || fread(im.v.data(), 4, im.v.size(), fp) != im.v.size()) { printf("cannot read %s\n", argv[3]); return 1; }
        fclose(fp);
        auto lab = run(im, {32, 0, -1, conn, false});
        auto v = check(im, conn, lab);
        printf("%s: truth %d, NPP %d, broken links %d, false merges %d\n", argv[3], v.n_ref, v.n_npp, v.broken, v.merges);
        fp = fopen(argv[7], "wb"); fwrite(lab.data(), 4, lab.size(), fp); fclose(fp);
        return 0;
    }
    if (argc > 3 && std::string(argv[2]) == "--shapes") {
        // Every shape of one value (0) that fits in 4 x 4, on a filler of value 2, inside a 32 x 32
        // image at (x0, y0) = (2, 2) and (13, 2). Keeps the failing shapes that contain no smaller
        // failing shape (minimal), and prints them.
        int conn = atoi(argv[3]);
        for (int x0 : {2, 13}) {
            std::vector<char> bad(1 << 16, 0);
            for (int bits = 1; bits < (1 << 16); bits++) {
                Img im{32, 32, std::vector<uint32_t>(1024, 2)};
                for (int k = 0; k < 16; k++) if (bits >> k & 1) im.v[(2 + k / 4) * 32 + x0 + k % 4] = 0;
                auto v = check(im, conn, run(im, {32, 0, -1, conn, false}));
                bad[bits] = v.broken || v.merges;
            }
            int nbad = 0; std::vector<int> minimal;
            for (int bits = 1; bits < (1 << 16); bits++) {
                if (!bad[bits]) continue;
                nbad++;
                bool has_smaller = false;  // any failing proper subset (one pixel fewer is enough to test transitively)
                for (int k = 0; k < 16 && !has_smaller; k++) if (bits >> k & 1) has_smaller = bad[bits & ~(1 << k)];
                if (!has_smaller) minimal.push_back(bits);
            }
            // fold shapes that are the same up to translation inside the 4 x 4 window
            std::set<std::string> seen;
            printf("%d neighbours, window at x0=%d: %d of 65535 shapes fail; minimal ones (up to translation):\n", conn, x0, nbad);
            for (int b : minimal) {
                int r0 = 4, r1 = -1, c0 = 4, c1 = -1;
                for (int k = 0; k < 16; k++) if (b >> k & 1) { r0 = std::min(r0, k / 4); r1 = std::max(r1, k / 4); c0 = std::min(c0, k % 4); c1 = std::max(c1, k % 4); }
                std::string key;
                for (int r = r0; r <= r1; r++) { for (int c = c0; c <= c1; c++) key += (b >> (r * 4 + c) & 1) ? '0' : '.'; key += '/'; }
                if (!seen.insert(key).second) continue;
                printf("   ");
                for (char ch : key) printf(ch == '/' ? "   " : "%c", ch);
                printf("\n");
            }
        }
        return 0;
    }
    if (argc > 2 && std::string(argv[2]) == "--rates") {
        // The minimal patterns found by --shrink, each placed in an image of filler value 2:
        // how often NPP gets it wrong in 200 runs, by position and image width.
        struct Pat { const char* name; int conn, h, w; const char* rows; };
        Pat pats[] = {
            {"V  (8 neighbours)", 8, 3, 3, "0.0" "0.0" ".0."},
            {"U  (4 neighbours)", 4, 3, 4, "0.00" "0..0" "0000"},
        };
        for (auto& pt : pats)
            for (int W : {32, 64, 100})
                for (int x0 : {1, 5, 13, 16, 29}) {
                    if (x0 + pt.w > W) continue;
                    Img im{24, W, std::vector<uint32_t>((size_t)24 * W, 2)};
                    for (int r = 0; r < pt.h; r++) for (int c = 0; c < pt.w; c++) if (pt.rows[r * pt.w + c] != '.') im.v[(5 + r) * W + x0 + c] = pt.rows[r * pt.w + c] - '0';
                    int bad = 0;
                    for (int t = 0; t < 200; t++) { auto v = check(im, pt.conn, run(im, {32, 0, -1, pt.conn, false})); bad += v.broken || v.merges; }
                    printf("%s width %3d, at column %2d: wrong %3d/200\n", pt.name, W, x0, bad);
                }
        // The wrap-around pair: last pixel of a row and first pixel of the next, same value, not neighbours.
        for (int W : {8, 16, 24, 31, 32, 33, 64, 100, 128})
            for (int conn : {8, 4}) {
                Img im{16, W, std::vector<uint32_t>((size_t)16 * W, 2)};
                im.v[5 * W + W - 1] = 0; im.v[6 * W] = 0;
                int bad = 0;
                for (int t = 0; t < 50; t++) { auto v = check(im, conn, run(im, {32, 0, -1, conn, false})); bad += v.broken || v.merges; }
                printf("wrap pair, width %3d, %d neighbours: wrong %2d/50\n", W, conn, bad);
            }
        return 0;
    }
    if (argc > 3 && std::string(argv[2]) == "--shrink") {
        // Find a random 2-value image NPP gets wrong, then turn pixels into a third value (2)
        // one at a time, keeping each change while NPP still fails in one of 20 runs.
        // Prints the smallest failing pattern found ('.' = 2, the filler value).
        int conn = atoi(argv[3]);
        std::mt19937 gs(conn * 7 + 3);
        auto fails = [&](const Img& im) { for (int t = 0; t < 20; t++) { auto v = check(im, conn, run(im, {32, 0, -1, conn, false})); if (v.broken || v.merges) return true; } return false; };
        for (int attempt = 0; attempt < 3; attempt++) {
            Img im; int tries = 0;
            do { im = random_img(24, 24, 2, gs); } while (!fails(im) && ++tries < 50);
            if (tries == 50) { printf("%d neighbours: no failing 24 x 24 image found\n", conn); return 0; }
            for (bool changed = true; changed;) {
                changed = false;
                for (int i = 0; i < 24 * 24; i++) {
                    if (im.v[i] == 2) continue;
                    uint32_t keep = im.v[i]; im.v[i] = 2;
                    if (fails(im)) changed = true; else im.v[i] = keep;
                }
            }
            int y0 = 24, y1 = -1, x0 = 24, x1 = -1, n = 0;
            for (int i = 0; i < 576; i++) if (im.v[i] != 2) { int y = i / 24, x = i % 24; y0 = std::min(y0, y); y1 = std::max(y1, y); x0 = std::min(x0, x); x1 = std::max(x1, x); n++; }
            printf("%d neighbours, attempt %d: %d pixels kept, at rows %d-%d, cols %d-%d:\n", conn, attempt, n, y0, y1, x0, x1);
            for (int y = y0; y <= y1; y++) { printf("      "); for (int x = x0; x <= x1; x++) { uint32_t v = im.v[y * 24 + x]; printf("%c ", v == 2 ? '.' : '0' + v); } printf("\n"); }
        }
        return 0;
    }
    if (argc > 2 && std::string(argv[2]) == "--enumerate") {
        // Every binary 3 x 3 pattern (values 0/1) inside a 32 x 32 image of value 2, at a few offsets
        // relative to NPP's apparent 16 x 16 tiles. A pattern fails if any run of 5 is wrong.
        // Prints the failing patterns with the fewest 1s first (the smallest ways to break NPP).
        int offs[][2] = {{2, 2}, {6, 5}, {13, 2}, {14, 2}, {15, 2}, {2, 13}, {2, 14}, {2, 15}};
        for (int conn : {8, 4}) for (auto& o : offs) {
            std::vector<std::pair<int, int>> fails;  // (ones, pattern bits)
            for (int bits = 0; bits < 512; bits++) {
                Img im{32, 32, std::vector<uint32_t>(1024, 2)};
                for (int k = 0; k < 9; k++) im.v[(o[1] + k / 3) * 32 + o[0] + k % 3] = (bits >> k) & 1;
                bool bad = false;
                for (int t = 0; t < 5 && !bad; t++) { auto v = check(im, conn, run(im, {32, 0, -1, conn, false})); bad = v.broken || v.merges; }
                if (bad) fails.push_back({__builtin_popcount(bits), bits});
            }
            std::sort(fails.begin(), fails.end());
            printf("%d neighbours, pattern at x0=%d y0=%d: %zu of 512 patterns fail", conn, o[0], o[1], fails.size());
            if (!fails.empty()) {
                printf("; smallest:\n");
                for (size_t i = 0; i < std::min<size_t>(fails.size(), 4); i++) {
                    int b = fails[i].second;
                    for (int r = 0; r < 3; r++) printf("      %d %d %d\n", b >> (3 * r) & 1, b >> (3 * r + 1) & 1, b >> (3 * r + 2) & 1);
                    printf("\n");
                }
            } else printf("\n");
        }
        return 0;
    }
    if (argc > 5 && std::string(argv[2]) == "--raw") {  // 8-bit image file, W x H, as the sample reads it
        int W = atoi(argv[4]), H = atoi(argv[5]);
        FILE* fp = fopen(argv[3], "rb");
        if (!fp) { printf("cannot open %s\n", argv[3]); return 1; }
        std::vector<uint8_t> buf((size_t)W * H);
        if (fread(buf.data(), 1, buf.size(), fp) != buf.size()) { printf("short read\n"); return 1; }
        fclose(fp);
        Img im{H, W, std::vector<uint32_t>(buf.begin(), buf.end())};
        std::set<uint32_t> vals(im.v.begin(), im.v.end());
        for (int conn : {8, 4}) {
            int nref; reference(im, conn, &nref);
            int wrong = 0; long broken = 0; std::set<std::vector<uint32_t>> answers; std::set<int> counts;
            for (int t = 0; t < 30; t++) {
                auto lab = run(im, {8, 0, -1, conn, false});
                auto v = check(im, conn, lab);
                wrong += v.broken || v.merges; broken += v.broken; answers.insert(lab); counts.insert(v.n_npp);
            }
            printf("%s %dx%d, %zu values, %d neighbours: truth %d; NPP wrong in %d/30 runs, %ld broken links in total, %zu distinct answers, counts:",
                   argv[3], W, H, vals.size(), conn, nref, wrong, broken, answers.size());
            for (int k : counts) printf(" %d", k);
            printf("\n");
        }
        return 0;
    }

    // ---------- A ----------
    Img a{3, 3, {0, 2, 0, 0, 2, 0, 1, 0, 1}};
    printf("A. the 3 x 3 image (left), labels (right); truth: 4 components with 8 neighbours\n");
    struct { const char* name; How h; } ways[] = {
        {"as NVIDIA's sample: 8u input, scratch zeroed", {8, 0, -1, 8, false}},
        {"32u input, scratch zeroed", {32, 0, -1, 8, false}},
        {"32u, scratch left as allocated", {32, -1, -1, 8, false}},
        {"32u, scratch filled 0xFF", {32, 0xFF, -1, 8, false}},
        {"32u, output pre-zeroed", {32, 0, 0, 8, false}},
        {"32u, output pre-filled 0xFF", {32, 0, 0xFF, 8, false}},
        {"32u, own stream", {32, 0, -1, 8, true}},
        {"16u input", {16, 0, -1, 8, false}},
        {"batch API (8u, Advanced)", {0, 0, -1, 8, false}},
        {"32u, 4 neighbours (truth: 5)", {32, 0, -1, 4, false}},
    };
    for (auto& w : ways) {
        auto lab = run(a, w.h);
        auto v = check(a, w.h.conn, lab);
        printf(" %-46s truth %d, NPP %d, broken links %d, false merges %d %s\n", w.name, v.n_ref, v.n_npp, v.broken, v.merges, v.broken || v.merges ? "WRONG" : "ok");
        if (&w == &ways[1]) print_grid(a, lab);
    }

    // ---------- B, C ----------
    printf("\nB. random images, called as the sample does (32u, zeroed scratch, NULL stream)\n");
    std::mt19937 g(1);
    int sizes[][2] = {{1, 64}, {64, 1}, {2, 2}, {3, 3}, {4, 4}, {8, 8}, {16, 16}, {31, 31}, {32, 32}, {33, 33}, {64, 64}, {128, 128}, {512, 512}};
    std::map<std::string, int> hist;  // C: where broken links are
    long total_broken = 0;
    for (int conn : {8, 4})
        for (int values : {2, 3, 256}) {
            printf(" %d neighbours, %3d values:", conn, values);
            for (auto& s : sizes) {
                int bad = 0, trials = quick ? 5 : 20;
                for (int t = 0; t < trials; t++) {
                    Img im = random_img(s[0], s[1], values, g);
                    auto v = check(im, conn, run(im, {32, 0, -1, conn, false}));
                    bad += v.broken || v.merges;
                    total_broken += v.broken;
                    for (auto& b : v.where) {
                        hist[std::string("dir ") + "RDXL"[b.dir]]++;
                        for (int m : {2, 4, 8, 16, 32}) {
                            hist["x%" + std::to_string(m) + "=" + std::to_string(b.x % m)]++;
                            hist["y%" + std::to_string(m) + "=" + std::to_string(b.y % m)]++;
                        }
                        hist[std::string("at ") + (b.x == 0 || b.x == s[1] - 1 ? "left/right edge" : b.y == 0 || b.y == s[0] - 1 ? "top/bottom edge" : "interior")]++;
                    }
                }
                printf(" %dx%d:%d/%d", s[0], s[1], bad, trials);
            }
            printf("\n");
        }
    printf("\nC. %ld broken links; counts by direction (R right, D down, X down-right, L down-left), position, and coordinate modulo tile sizes:\n", total_broken);
    for (auto& [k, v] : hist) if (k.rfind("dir", 0) == 0 || k.rfind("at", 0) == 0) printf("   %-24s %ld (%.1f%%)\n", k.c_str(), (long)v, 100.0 * v / std::max(1L, total_broken));
    for (int m : {2, 4, 8, 16, 32}) for (const char* ax : {"x", "y"}) {
        printf("   %s mod %2d:", ax, m);
        for (int r = 0; r < m; r++) printf(" %d", hist[std::string(ax) + "%" + std::to_string(m) + "=" + std::to_string(r)]);
        printf("\n");
    }

    // ---------- D ----------
    printf("\nD. the 3 x 3 pattern placed at offset (x0, y0) in a 70 x 70 image of value 9: '#' = NPP wrong\n    x0 0..63 ->\n");
    for (int y0 = 0; y0 < 66; y0 += quick ? 6 : 1) {
        printf(" %2d ", y0);
        for (int x0 = 0; x0 < 66; x0++) {
            Img im{70, 70, std::vector<uint32_t>(4900, 9)};
            for (int k = 0; k < 9; k++) im.v[(y0 + k / 3) * 70 + x0 + k % 3] = a.v[k];
            auto v = check(im, 8, run(im, {32, 0, -1, 8, false}));
            printf("%c", v.broken || v.merges ? '#' : '.');
        }
        printf("\n");
    }

    // ---------- E ----------
    printf("\nE. one 512 x 512 image (3 values), 50 runs: distinct answers\n");
    Img e = random_img(512, 512, 3, g);
    std::set<std::vector<uint32_t>> answers;
    std::set<int> counts;
    for (int t = 0; t < 50; t++) {
        auto lab = run(e, {32, 0, -1, 8, false});
        answers.insert(lab);
        counts.insert(check(e, 8, lab).n_npp);
    }
    int nref; reference(e, 8, &nref);
    printf("   truth %d components; NPP gave %zu different label images, component counts:", nref, answers.size());
    for (int k : counts) printf(" %d", k);
    printf("\n");

    // ---------- F ----------
    printf("\nF. entry points on 20 random 64 x 64 images with 3 values, 8 neighbours: wrong / 20\n");
    for (auto [name, depth] : std::vector<std::pair<const char*, int>>{{"8u32u", 8}, {"16u32u", 16}, {"32u", 32}, {"batch 8u32u Advanced", 0}}) {
        std::mt19937 gg(7);
        int bad = 0;
        for (int t = 0; t < 20; t++) { Img im = random_img(64, 64, 3, gg); auto v = check(im, 8, run(im, {depth, 0, -1, 8, false})); bad += v.broken || v.merges; }
        printf("   %-22s %d\n", name, bad);
    }
    return 0;
}
