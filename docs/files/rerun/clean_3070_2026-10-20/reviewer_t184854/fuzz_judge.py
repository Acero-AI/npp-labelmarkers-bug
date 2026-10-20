import sys, numpy as np
sys.path.insert(0, sys.argv[1])
import check_labels as C
from collections import deque
def bfs(img, conn):
    H, W = img.shape; t = -np.ones((H, W), np.int64); n = 0
    nb = [(-1,0),(1,0),(0,-1),(0,1)] + ([(-1,-1),(-1,1),(1,-1),(1,1)] if conn == 8 else [])
    for y in range(H):
        for x in range(W):
            if t[y, x] >= 0: continue
            q = deque([(y, x)]); t[y, x] = n
            while q:
                a, b = q.popleft()
                for dy, dx in nb:
                    Y, X = a+dy, b+dx
                    if 0 <= Y < H and 0 <= X < W and t[Y, X] < 0 and img[Y, X] == img[a, b]:
                        t[Y, X] = n; q.append((Y, X))
            n += 1
    return t, n
rng = np.random.default_rng(1); bad = 0; N = 0
for trial in range(400):
    H, W = rng.integers(1, 12, 2); conn = [4, 8][trial % 2]
    img = rng.integers(0, rng.integers(1, 4) + 1, (H, W)).astype(np.uint32) * 37 + 5
    t, n = bfs(img, conn)
    perm = rng.permutation(4_000_000_000 // 1000)[:n].astype(np.uint32) * 1000 + 7
    good = perm[t]
    r = C.judge(img, conn, good); N += 1
    if r["broken links"] or r["merged labels"] or r["truth"] != n: bad += 1; print("FALSE WRONG", img, conn, r)
    # mutate: split one pixel off a multi-pixel region, must be WRONG
    sizes = np.bincount(t.ravel())
    big = np.where(sizes > 1)[0]
    if len(big):
        k = big[0]; ys, xs = np.where(t == k); lab = good.copy(); lab[ys[0], xs[0]] = 1
        if not (lab == 1).sum() == 1: continue
        r2 = C.judge(img, conn, lab)
        if not (r2["broken links"] or r2["merged labels"]): bad += 1; print("MISSED SPLIT", img, conn)
    if n > 1:
        lab = good.copy(); lab[t == 1] = good[t == 0][0]
        r3 = C.judge(img, conn, lab)
        if not r3["merged labels"]: bad += 1; print("MISSED MERGE", img, conn)
print("trials", N, "judge errors", bad)
