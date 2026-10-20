"""Judge a label file against its image, on the CPU, with no GPU and no cc3d.

    python check_labels.py IMAGE W H CONN LABELS [LABELS ...]

IMAGE is either <name>.u32 (one uint32 value per pixel) or an 8-bit .raw from cuda-samples
(only the first W*H bytes are read, as NVIDIA's sample program does). LABELS are uint32 label
files of W*H values, such as NPP's output. CONN is 4 or 8.

Ground truth here is SciPy's connected_components on the graph whose edges join neighbouring
pixels of equal value. For each label file it prints:
  broken links   pairs of touching, equal pixels that got different labels (a region was cut)
  merged labels  labels that cover pixels of more than one true region (regions were joined)
A correct labelling has both at zero; the label values themselves may be anything.
"""
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def edges(img, conn):
    H, W = img.shape
    idx = np.arange(H * W).reshape(H, W)
    for dy, dx in [(0, 1), (1, 0)] + ([(1, 1), (1, -1)] if conn == 8 else []):
        xs, xe = max(0, -dx), W - max(0, dx)
        p, q = idx[:H - dy, xs:xe], idx[dy:, xs + dx:xe + dx]
        same = img[:H - dy, xs:xe] == img[dy:, xs + dx:xe + dx]
        yield p[same], q[same]


def truth(img, conn):
    a, b = map(np.concatenate, zip(*edges(img, conn)))
    n = img.size
    return connected_components(coo_matrix((np.ones(len(a), bool), (a, b)), shape=(n, n)), directed=False)


def judge(img, conn, lab):
    n_true, t = truth(img, conn)
    lab = lab.ravel()
    broken = sum(int((lab[p] != lab[q]).sum()) for p, q in edges(img, conn))
    pairs = np.unique(np.stack([lab.astype(np.int64), t]), axis=1)
    labels, per_label = np.unique(pairs[0], return_counts=True)
    return {"truth": n_true, "labels": len(labels), "broken links": broken, "merged labels": int((per_label > 1).sum())}


if __name__ == "__main__":
    path, W, H, conn = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    img = (np.fromfile(path, np.uint8, W * H) if path.endswith(".raw") else np.fromfile(path, np.uint32)).reshape(H, W)
    for f in sys.argv[5:]:
        r = judge(img, conn, np.fromfile(f, np.uint32).reshape(H, W))
        print(f"{f}: truth {r['truth']}, file has {r['labels']} labels, broken links {r['broken links']}, "
              f"merged labels {r['merged labels']} -> {'WRONG' if r['broken links'] or r['merged labels'] else 'correct'}")
