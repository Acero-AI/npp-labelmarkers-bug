// Exact-colour connected components on the GPU (CUDA).
//
// Two pixels belong to the same component when they have exactly the same
// RGB value and touch (4- or 8-neighbourhood). One pass over the image
// labels every colour at once; there is no loop over colours.
//
// The kernels:
//   init     each pixel points at a parent: the first pixel of its
//            same-colour run inside its warp (RUNS), or itself
//   merge    each pixel joins the trees of its same-colour neighbours to
//            the left and in the row above (union-find, atomicMin)
//   flatten  each pixel points straight at its root
//   rank_roots, number
//            each root gets its rank inside a 256-pixel chunk; after a
//            prefix sum over the chunk counts, each pixel takes its root's
//            number 1..n (ignored pixels get 0)
// Until number runs, a label is a pixel index (its parent, then its root);
// ignored pixels hold -1.
//
// The Metal twin (../metal/kernels.metal) is the same code line for line.
#pragma once
#include <cstdint>

namespace colorcc {

// Optimisation switches, so each one can be measured on its own.
enum Flags : int {
    RUNS = 1,    // init: a warp ballot finds horizontal runs, so most left-joins vanish
    SKIP = 2,    // merge: skip joins another thread already makes
    PACKED = 4,  // init stores the colour as one int; merge reads 1 word, not 3 bytes
    HALVE = 8,   // merge: path halving keeps the trees shallow
};

struct Params {
    int H, W;
    int conn;    // 4 or 8
    int flags;
    int ignore;  // packed colour that gets no label (e.g. white paper), or -1
};

__device__ __forceinline__ int rgb(const uint8_t* __restrict__ img, int i) {
    const uint8_t* p = img + 3 * (size_t)i;
    return p[0] | (p[1] << 8) | (p[2] << 16);
}

__device__ __forceinline__ int colour(const uint8_t* __restrict__ img, const int* __restrict__ key,
                                      int i, int flags) {
    return (flags & PACKED) ? key[i] : rgb(img, i);
}

// Follow parents to the root. Labels only ever shrink, so a stale read is
// an older ancestor, still on the path to the root.
__device__ __forceinline__ int find_root(const int* label, int i) {
    int p = label[i];
    while (p != i) {
        i = p;
        p = label[i];
    }
    return i;
}

// Walk to the root, pointing each pixel on the way at its grandparent, so
// the trees stay shallow for every thread that walks them later. A parent
// always has a smaller index than its child and a root is the smallest in
// its tree, so atomicMin only ever moves a pointer to an ancestor, closer to
// the root: a late shortcut cannot undo a pointer that is already lower, and
// a join that loses a race to it retries from the new parent.
__device__ __forceinline__ int find_root_halving(int* label, int i) {
    while (true) {
        const int p = label[i];
        if (p == i) return i;
        const int gp = label[p];
        if (gp == p) return p;
        atomicMin(label + i, gp);
        i = gp;
    }
}

// Join the trees of a and b: the larger root is pointed at the smaller one.
// If another thread re-pointed that root first, atomicMin returns its new
// parent and we try again from there.
__device__ void unite(int* label, int a, int b, bool halve) {
    while (true) {
        a = halve ? find_root_halving(label, a) : find_root(label, a);
        b = halve ? find_root_halving(label, b) : find_root(label, b);
        if (a == b) return;
        if (a > b) { int t = a; a = b; b = t; }
        int old = atomicMin(label + b, a);
        if (old == b) return;
        b = old;
    }
}

// blockDim.x must be 32 so each warp is 32 consecutive pixels of one row.
__global__ void init(const uint8_t* __restrict__ img, int* __restrict__ key,
                     int* __restrict__ label, Params P) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const bool in = x < P.W && y < P.H;
    const int i = y * P.W + x;
    const int k = in ? rgb(img, i) : -2;
    int parent = i;
    if (P.flags & RUNS) {
        // A run starts where the colour changes. Every lane votes; the
        // nearest start at or left of this lane is the run's first pixel.
        const int lane = threadIdx.x & 31;
        const bool start = !in || lane == 0 || rgb(img, i - 1) != k;
        const unsigned starts = __ballot_sync(0xffffffffu, start);
        const int first = 31 - __clz(starts & (0xffffffffu >> (31 - lane)));
        parent = i - (lane - first);
    }
    if (!in) return;
    if (P.flags & PACKED) key[i] = k;
    label[i] = (k == P.ignore) ? -1 : parent;
}

// Joins to the row above. If the left neighbour has the same colour (left),
// it already reaches up-left and up, so with SKIP one join per run above is enough.
__device__ void unite_above(const uint8_t* __restrict__ img, const int* __restrict__ key,
                            int* label, Params P, int i, int x, int k, bool left) {
    const int f = P.flags;
    const int up = i - P.W;
    const bool u = colour(img, key, up, f) == k;
    if (P.conn == 4) {
        // If left, up-left and up all match, left already joined up-left,
        // which touches up.
        const bool covered = (f & SKIP) && left && colour(img, key, up - 1, f) == k;
        if (u && !covered) unite(label, i, up, f & HALVE);
        return;
    }
    const bool ul = x > 0 && colour(img, key, up - 1, f) == k;
    const bool ur = x + 1 < P.W && colour(img, key, up + 1, f) == k;
    if (!(f & SKIP)) {
        if (ul) unite(label, i, up - 1, f & HALVE);
        if (u) unite(label, i, up, f & HALVE);
        if (ur) unite(label, i, up + 1, f & HALVE);
    } else if (left) {
        if (ur && !u) unite(label, i, up + 1, f & HALVE);
    } else if (u) {
        unite(label, i, up, f & HALVE);
    } else {
        if (ul) unite(label, i, up - 1, f & HALVE);
        if (ur) unite(label, i, up + 1, f & HALVE);
    }
}

// The row above first, then the left neighbour. The row above is mostly joined
// already, so its paths are short. Joining left first, a whole row joins at the
// same moment and builds long chains: 11x slower without the other switches.
__global__ void merge(const uint8_t* __restrict__ img, const int* __restrict__ key,
                      int* label, Params P) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= P.W || y >= P.H) return;
    const int i = y * P.W + x;
    const int f = P.flags;
    const int k = colour(img, key, i, f);
    if (k == P.ignore) return;

    const bool left = x > 0 && colour(img, key, i - 1, f) == k;
    if (y > 0) unite_above(img, key, label, P, i, x, k, left);
    // With RUNS, pixels inside one warp already share a parent, so only lane 0
    // has to reach across to the previous warp.
    if (left && !((f & RUNS) && (threadIdx.x & 31) != 0)) unite(label, i, i - 1, f & HALVE);
}

// Point every pixel at its root. Merging is over, so the trees are fixed:
// other threads only read, and a reader sees either a pixel's old parent or
// its root, both on the way to the root.
__global__ void flatten(int* label, Params P) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= P.W || y >= P.H) return;
    const int i = y * P.W + x;
    const int l = label[i];
    if (l >= 0 && l != i) {
        const int r = find_root(label, l);
        if (r != l) label[i] = r;
    }
}

// Numbering, without a second image-sized buffer. The pixels are cut into
// chunks of CHUNK consecutive indices (one 1-D block each). rank_roots gives
// every root its rank among the roots of its chunk, stored in its own slot
// as -(rank + 2) so that -1 still means "ignored", and writes each chunk's
// root count. A prefix sum over the counts (n / CHUNK numbers, not n) then
// gives each chunk its first label, and number hands out the labels.
constexpr int CHUNK = 256;

__global__ void rank_roots(int* label, int* __restrict__ count, int n) {
    __shared__ int warp_roots[CHUNK / 32];
    const int i = blockIdx.x * CHUNK + threadIdx.x;
    const bool root = i < n && label[i] == i;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const unsigned votes = __ballot_sync(0xffffffffu, root);
    if (lane == 0) warp_roots[warp] = __popc(votes);
    __syncthreads();
    int rank = __popc(votes & ((1u << lane) - 1));  // roots before me in my warp
    for (int w = 0; w < warp; w++) rank += warp_roots[w];  // and in earlier warps
    if (root) label[i] = -(rank + 2);
    if (threadIdx.x == 0) {
        int total = 0;
        for (int w = 0; w < CHUNK / 32; w++) total += warp_roots[w];
        count[blockIdx.x] = total;
    }
}

// last[c] is the number of roots in chunks 0..c (an inclusive prefix sum).
// A root's slot holds its encoded rank until its own thread replaces it with
// the final label; a reader may see either, and both give the same answer.
__global__ void number(int* label, const int* __restrict__ last, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int l = label[i];
    if (l == -1) { label[i] = 0; return; }
    const int r = l < 0 ? i : l;      // a root holds its rank, every other pixel its root
    const int v = l < 0 ? l : label[r];
    const int c = r / CHUNK;
    label[i] = v > 0 ? v : (c ? last[c - 1] : 0) + (-v - 2) + 1;
}

}  // namespace colorcc
