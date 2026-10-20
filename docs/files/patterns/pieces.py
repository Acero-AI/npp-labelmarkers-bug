"""The pieces NPP cut off: for each true region that NPP split, every label other than the one covering most of it.
Per file: number of pieces, their sizes, and whether each piece lies inside one T x T tile (T = 16, 32, 64) or one
T-wide column band / T-tall row band. Truth from scipy (cuts.py's inputs).    python pieces.py"""
import numpy as np, collections, sys
sys.path.insert(0, ".")
from scipy import ndimage
import cuts  # reuses the job list (prints cuts.py output too; ignored)
st8 = np.ones((3, 3), int)
for name, img, lab in cuts.jobs:
    # true regions: label each value separately (8 neighbours, as every job here is 8-way)
    truth = np.zeros(img.shape, np.int64); off = 0
    for v in np.unique(img):
        t, n = ndimage.label(img == v, structure=st8); truth[t > 0] = t[t > 0] + off; off += n
    pairs = collections.Counter(zip(truth.ravel().tolist(), lab.ravel().tolist()))
    main = {}
    for (t, l), c in pairs.items():
        if c > main.get(t, (None, -1))[1]: main[t] = (l, c)
    pieces = [(t, l, c) for (t, l), c in pairs.items() if main[t][0] != l]
    if not pieces: print(name, ": no pieces"); continue
    flat_t, flat_l = truth.ravel(), lab.ravel()
    ys, xs = np.divmod(np.arange(flat_t.size), img.shape[1])
    key = flat_t * (flat_l.max() + 1) + flat_l
    want = {t * (flat_l.max() + 1) + l for t, l, _ in pieces}
    sel = np.isin(key, list(want))
    k, y, x = key[sel], ys[sel], xs[sel]
    order = np.argsort(k, kind="stable"); k, y, x = k[order], y[order], x[order]
    starts = np.r_[0, np.nonzero(np.diff(k))[0] + 1, len(k)]
    res = collections.Counter(); sizes = []; tops = collections.Counter(); lefts = collections.Counter()
    for a, b in zip(starts[:-1], starts[1:]):
        yy, xx = y[a:b], x[a:b]; sizes.append(b - a)
        for T in (16, 32, 64):
            res[f"tile{T}"] += (yy.min() // T == yy.max() // T) and (xx.min() // T == xx.max() // T)
            res[f"col{T}"] += xx.min() // T == xx.max() // T
            res[f"row{T}"] += yy.min() // T == yy.max() // T
        tops[int(yy.min() % 16)] += 1; lefts[int(xx.min() % 16)] += 1
    n = len(sizes)
    print(f"{name}: {n} pieces cut off {len({p[0] for p in pieces})} regions; size median {int(np.median(sizes))}, max {max(sizes)}; "
          + ", ".join(f"inside one {k}: {v}/{n}" for k, v in res.items()))
    print(f"   top row of piece mod 16: {[tops[i] for i in range(16)]}; left column mod 16: {[lefts[i] for i in range(16)]}")
