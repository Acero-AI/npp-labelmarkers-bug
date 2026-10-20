"""Where do NPP's broken links sit? For each label file: every pair of touching equal pixels with different labels,
by direction, and the position of the link modulo 8/16/32/64. A link crosses a tile line of size T when the two pixels
fall in different T-tiles along that axis.
    python cuts.py   (paths are fixed below; CPU only)"""
import pathlib, numpy as np, collections
B = pathlib.Path.home() / "ccl-blog"
D8 = B / "npp-bundle2" / "data"
RR = B / "npp-report" / "rerun" / "4060ti_2026-10-19"
def raw8(name, w, h): return np.fromfile(D8 / name, np.uint8)[: w * h].reshape(h, w).astype(np.int64)
def u32(p, w, h): return np.fromfile(p, np.uint32)[: w * h].reshape(h, w).astype(np.int64)
IM = {"teapot": ("teapot_512x512_8u.raw", 512, 512), "CT_skull": ("CT_skull_512x512_8u.raw", 512, 512), "PCB_METAL": ("PCB_METAL_509x335_8u.raw", 509, 335),
      "PCB2": ("PCB2_1024x683_8u.raw", 1024, 683), "PCB": ("PCB_1280x720_8u.raw", 1280, 720)}
jobs = []
for im, (f, w, h) in IM.items():
    for kind in ("LabelMarkersUF", "LabelMarkersUFBatch"):
        for src, d in (("nvidia-committed-12.0.0", B / "npp-report" / "nvidia_committed"), ("sample-12.4.1-4060", RR / "sample12" / "run1"), ("nppplus-0.10-4060", RR / "nppplus")):
            p = d / f"{im}_{kind}_8Way_{w}x{h}_32u.raw"
            if not p.exists() and im == "CT_skull": p = d / f"CT_Skull_{kind}_8Way_{w}x{h}_32u.raw"
            if p.exists(): jobs.append((f"{src}/{im}/{kind}", raw8(f, w, h), u32(p, w, h)))
man = {"pcb_nvidia_sample": (1280, 720), "ct_skull_nvidia_sample": (512, 512)}
for n, (w, h) in man.items():
    img = u32(B / "npp-vis" / "in" / f"{n}.u32", w, h)
    for v in ("13", "11"): jobs.append((f"harness-npp{v}-3070/{n}", img, u32(B / "npp-vis" / "out" / f"{n}_npp{v}.u32", w, h)))
DIRS = {"right": (0, 1), "down": (1, 0), "down-right": (1, 1), "down-left": (1, -1)}
tot = collections.Counter()
for name, img, lab in jobs:
    h, w = img.shape
    rows = []
    for d, (dy, dx) in DIRS.items():
        y0, y1 = 0, h - dy
        x0, x1 = max(0, -dx), w - max(0, dx)
        a, b = img[y0:y1, x0:x1], img[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        la, lb = lab[y0:y1, x0:x1], lab[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        same = a == b
        br = same & (la != lb)
        ys, xs = np.nonzero(br); ys += y0; xs += x0
        if not len(ys): continue
        line = [f"{d}: {len(ys)} broken"]
        for T in (8, 16, 32, 64):
            cx = (xs // T != (xs + dx) // T) if dx else np.zeros(len(xs), bool)
            cy = (ys // T != (ys + dy) // T) if dy else np.zeros(len(ys), bool)
            cross = cx | cy
            # fraction of ALL equal-value links of this direction that cross a T line, for comparison
            Ys, Xs = np.nonzero(same); Ys += y0; Xs += x0
            base = (((Xs // T) != (Xs + dx) // T) if dx else np.zeros(len(Xs), bool)) | (((Ys // T) != (Ys + dy) // T) if dy else np.zeros(len(Ys), bool))
            line.append(f"cross {T}-line {cross.mean():.2f} (all links {base.mean():.2f})")
            if d in ("right", "down") and T == 16: tot[(d, "cross16")] += cross.sum(); tot[(d, "n")] += len(ys)
        rows.append("   " + "; ".join(line))
        if d == "down" and len(ys) > 20: rows.append(f"      y mod 16 of upper pixel: {np.bincount(ys % 16, minlength=16).tolist()}; x mod 16: {np.bincount(xs % 16, minlength=16).tolist()}")
        if d == "right" and len(ys) > 20: rows.append(f"      x mod 16 of left pixel: {np.bincount(xs % 16, minlength=16).tolist()}; y mod 16: {np.bincount(ys % 16, minlength=16).tolist()}")
    print(name); print("\n".join(rows) if rows else "   no broken links")
print("\nTOTAL", {f"{k[0]} {k[1]}": int(v) for k, v in tot.items()})
