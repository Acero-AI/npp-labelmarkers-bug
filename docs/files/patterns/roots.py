"""NPP's raw output is each pixel's root index. For the cut-off pieces in the harness dumps (raw, not compressed),
where is the piece's root: inside the piece? at which position modulo 16?    python roots.py"""
import numpy as np, collections, pathlib
from scipy import ndimage
B = pathlib.Path.home() / "ccl-blog" / "npp-vis"
for n, (w, h) in {"pcb_nvidia_sample": (1280, 720), "ct_skull_nvidia_sample": (512, 512)}.items():
    img = np.fromfile(B / "in" / f"{n}.u32", np.uint32)[: w * h].reshape(h, w)
    truth = np.zeros((h, w), np.int64); off = 0
    for v in np.unique(img):
        t, k = ndimage.label(img == v, structure=np.ones((3, 3), int)); truth[t > 0] = t[t > 0] + off; off += k
    for ver in ("13", "11"):
        lab = np.fromfile(B / "out" / f"{n}_npp{ver}.u32", np.uint32)[: w * h].reshape(h, w).astype(np.int64)
        pairs = collections.Counter(zip(truth.ravel().tolist(), lab.ravel().tolist()))
        best = {}
        for (t, l), c in pairs.items():
            if c > best.get(t, (0, -1))[1]: best[t] = (l, c)
        cut = [(t, l) for (t, l) in pairs if best[t][0] != l]
        inside = sum(lab.ravel()[l] == l and truth.ravel()[l] == t for t, l in cut) if cut else 0
        ry = collections.Counter(int((l // w) % 16) for t, l in cut); rx = collections.Counter(int((l % w) % 16) for t, l in cut)
        # the main piece's root vs the cut piece's root: which comes first in raster order?
        later = sum(l > best[t][0] for t, l in cut)
        # does the cut piece's root sit at the top-left pixel of the piece (min raster index)?
        flat_t, flat_l = truth.ravel(), lab.ravel()
        firsts = 0
        for t, l in cut:
            idx = np.flatnonzero((flat_t == t) & (flat_l == l)); firsts += idx.min() == l
        print(f"{n} NPP {ver}: {len(cut)} cut pieces; root is a pixel of the piece and its own root: {inside}; root is the piece's first pixel in raster order: {firsts}; "
              f"root index larger than the main piece's root: {later}\n   root y mod 16: {[ry[i] for i in range(16)]}\n   root x mod 16: {[rx[i] for i in range(16)]}")
