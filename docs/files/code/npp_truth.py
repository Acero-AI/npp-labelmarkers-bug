"""Ground truth for every image NPP was tested on, from two independent labellers.

    python bench/npp_truth.py OUT_DIR DIR_OR_RAW...

Each DIR holds <name>.u32 and manifest.json ({name: [W, H, conn]}); a RAW is
<name>_<W>x<H>_8u.raw from cuda-samples (8 neighbours, like the sample program).
For each image writes OUT_DIR/<name>_cc3d.u32 and <name>_ours.u32 and adds a row
to OUT_DIR/truth.json. Both number components in raster order of their first
pixel, so they must agree value for value; the script stops if they do not.
"""
import json
import os
import re
import sys

import cc3d
import numpy as np
import torch

import colorcc

out = sys.argv[1]
os.makedirs(out, exist_ok=True)


def images(arg):
    if arg.endswith(".raw"):
        name, W, H = re.match(r"(.+)_(\d+)x(\d+)_8u\.raw", os.path.basename(arg)).groups()
        W, H = int(W), int(H)  # the sample reads the first W*H bytes (teapot's file is longer)
        yield name, np.fromfile(arg, np.uint8, W * H).reshape(H, W).astype(np.uint32), 8
        return
    for name, (W, H, conn) in json.load(open(os.path.join(arg, "manifest.json"))).items():
        yield name, np.fromfile(os.path.join(arg, name + ".u32"), np.uint32).reshape(H, W), conn


rows = {}
for arg in sys.argv[2:]:
    for name, img, conn in images(arg):
        a, n_a = cc3d.connected_components(img.astype(np.uint64) + 1, connectivity=conn, return_N=True)
        rgb = np.stack([img & 255, img >> 8 & 255, img >> 16 & 255], -1).astype(np.uint8)
        b, n_b = colorcc.label(torch.from_numpy(rgb).to("mps"), conn=conn)
        b = b.cpu().numpy()
        same = n_a == n_b and np.array_equal(a.astype(np.int64), b.astype(np.int64))
        assert same, f"{name}: cc3d {n_a}, ours {n_b}"
        a.astype(np.uint32).tofile(os.path.join(out, f"{name}_cc3d.u32"))
        b.astype(np.uint32).tofile(os.path.join(out, f"{name}_ours.u32"))
        rows[name] = {"W": img.shape[1], "H": img.shape[0], "conn": conn, "values": int(len(np.unique(img))),
                      "cc3d": int(n_a), "ours": int(n_b), "identical": bool(same)}
        print(name, rows[name])

json.dump({"cc3d": "connected-components-3d 4.1.0", "torch": torch.__version__,
           "device": "Apple M5 Pro (Metal)", "images": rows}, open(os.path.join(out, "truth.json"), "w"), indent=1)
