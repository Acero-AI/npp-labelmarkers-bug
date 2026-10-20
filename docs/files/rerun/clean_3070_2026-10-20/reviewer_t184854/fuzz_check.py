import sys, numpy as np
sys.path.insert(0, sys.argv[1])
import check as K
from scipy import ndimage
rng = np.random.default_rng(2); bad = 0
for trial in range(300):
    H, W = rng.integers(1, 12, 2)
    img = rng.integers(0, 3, (H, W)).astype(np.uint8) * 50 + 3
    t = np.zeros((H, W), np.int64); n = 0
    for v in np.unique(img):
        l, k = ndimage.label(img == v, structure=np.ones((3, 3))); t[l > 0] = l[l > 0] + n; n += k
    good = (t * 7919 + 13).astype(np.uint32)
    if K.check(img, good) != (0, 0): bad += 1
    if n > 1:
        lab = good.copy(); lab[t == 2] = good[t == 1][0]
        if K.check(img, lab)[1] == 0: bad += 1
print("check.py errors", bad)
