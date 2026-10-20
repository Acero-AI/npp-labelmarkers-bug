import sys, glob, os, re
import numpy as np
from scipy import ndimage

def truth(img, conn):
    st = np.ones((3, 3), int) if conn == 8 else ndimage.generate_binary_structure(2, 1)
    out = np.zeros(img.shape, np.int64); n = 0
    for v in np.unique(img):
        t, k = ndimage.label(img == v, structure=st)
        out[t > 0] = t[t > 0] + n; n += k
    return out, n

def judge(img, lab, conn=8):
    t, n = truth(img, conn)
    a = t.ravel(); b = lab.ravel().astype(np.int64)
    # bijection test
    tl = {}; lt = {}
    pairs = np.unique(np.stack([a, b]), axis=1)
    splits = pairs.shape[1] - len(np.unique(pairs[0]))
    merges = pairs.shape[1] - len(np.unique(pairs[1]))
    return n, len(np.unique(b)), splits, merges

if __name__ == "__main__":
    imgdir, labdir = sys.argv[1], sys.argv[2]
    for f in sorted(glob.glob(os.path.join(labdir, "*_8Way_*_32u.raw"))):
        base = os.path.basename(f)
        if "Segment" in base: continue
        name = base.split("_LabelMarkers")[0].split("_CompressedMarker")[0]
        W, H = map(int, re.search(r"_(\d+)x(\d+)_32u", base).groups())
        cands = [p for p in glob.glob(os.path.join(imgdir, "*_8u.raw")) if os.path.basename(p).lower().startswith(name.lower() + "_%dx%d" % (W, H))]
        img = np.fromfile(cands[0], np.uint8, W * H).reshape(H, W)
        lab = np.fromfile(f, np.uint32).reshape(H, W)
        n, nl, s, m = judge(img, lab)
        print(f"{base}: truth {n} labels {nl} splits {s} merges {m} {'WRONG' if s or m else 'ok'}")
