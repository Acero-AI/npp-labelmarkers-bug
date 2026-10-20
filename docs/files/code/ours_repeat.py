"""Label NVIDIA's sample images with our labeller N times on this GPU; report how many distinct label images came out.
    python ours_repeat.py N OUTDIR data/NAME:W:H ...      (8-bit .raw as cuda-samples ships them)"""
import hashlib, sys
import numpy as np, torch, colorcc

n_runs, out = int(sys.argv[1]), sys.argv[2]
for spec in sys.argv[3:]:
    path, w, h = spec.split(":"); w, h = int(w), int(h)
    g = np.fromfile(path, np.uint8)[: w * h].reshape(h, w)
    img = torch.from_numpy(np.repeat(g[..., None], 3, 2)).cuda()
    seen = set()
    for _ in range(n_runs):
        lab, n = colorcc.label(img, conn=8)
        torch.cuda.synchronize()
        a = lab.cpu().numpy().astype(np.uint32)
        seen.add(hashlib.sha256(a.tobytes()).hexdigest())
    name = path.rsplit("/", 1)[-1].rsplit("_", 2)[0]
    a.tofile(f"{out}/{name}_ours.u32")
    print(f"{name}: {n} regions, {n_runs} runs, {len(seen)} distinct label images")
