# When nppiLabelMarkersUF fails: patterns (2026-10-19)

GPU runs: rented RTX 4060 Ti (sm_89, driver 580.173.02), pinned wheels NPP 13.2.0.58, 12.4.1.87, 11.8.0.86
(see ../rerun/4060ti_2026-10-19/versions.txt), NPP+ 0.10.0 cuda12 archive. CPU analyses: this folder's .py files.

| # | pattern | proof |
|---|---|---|
| 1 | Inside one 16×16 tile the failure is deterministic: one 3×3 pattern per tile gives the same wrong answer in 50/50 runs from 16² to 1024² (exactly one extra label per tile, 20,481 for 16,385 at 1024²); the 3×3 alone gives one identical wrong answer in 200/200 runs. | probe_npp13.txt T1, T2 |
| 2 | Run-to-run variation appears only when regions span many tiles: random 2-value images give 1 distinct answer in 50 runs up to 64² (13.2.0), 4 at 96² and 128², and 50 different answers in 50 runs from 256² up. | probe_npp13.txt, probe_npp12.txt T1 |
| 3 | Pixel values do not matter: every permutation of 0/1/2, 5/9/7, 255/0/128, 200/100/50 as 8u, and 70000.. and 4,000,000,000 as 32u, fails 20/20. Image width (5..80) and height (5..40) do not matter for a pattern inside the first tile. | probe_npp13.txt T3, T4 |
| 4 | Position rule for 13.x is exact: the 3×3 is labelled correctly iff x0 mod 16 = 15 or y0 mod 16 ∈ {14, 15}: 0 exceptions in 4,356 offsets for 13.2.0 and 13.0.1. Read as geometry: it fails unless a tile line separates the left arm from the joining pixel, or runs between any of its rows; a tile line between the joining pixel and the right arm (x0 mod 16 = 14) does not help. | dmap_compare.txt |
| 5 | 13.x maps are identical on RTX 3070 and 4060 Ti; 12.4.1 differs in 4 cells (y0 = 16, 32; first tile column), and reproduces cell for cell on the 4060 Ti across two days: GPU-dependent but not random. | dmap_compare.txt |
| 6 | The batch API (8u32u_Advanced) fails the same way with one pixel more reach: a V splits when 2 ≤ right arm ≤ left arm in the single-image call, 3 ≤ right arm ≤ left arm in the batch call (13.2.0 and 12.4.1). So batch labels the 3×3 correctly at all 4,624 offsets but fails random images from 32² up. | batchmap_npp13.txt, batchmap_npp12.txt, probe_npp13.txt T5 |
| 7 | Every cut-off piece is a well-formed tree whose root is its own first pixel in raster order (43/43 PCB, 30/30 CT_skull, NPP 13.2.0); its root index is larger than the main piece's root in 67 of 73 (the "second tree" of the rule). | roots.txt |
| 8 | Cut pieces are small and local: median 4–17 px in the sample outputs, and over half lie inside one 16×16 tile (e.g. 44 of 71 in NPP 12.4.1's PCB2). Broken links are only mildly enriched on tile lines (right links 16% cross a 16-line vs 6% of all links). | pieces.txt, cuts.txt |
| 9 | NPP+ 0.10.0 is NPP 12.4: its nppGetLibVersion returns 12.4.0, its 70×70 map equals NPP 12.4.1's cell for cell (807 correct, 0 differ, GPU-specific cells included), all 50 two-arm shapes give the same broken-link counts, section A and the 200-run rates are identical; it differs only in run-to-run noise. | harness_nppplus.txt, rates_nppplus.txt, grid_nppplus.txt, nppplus_compare.txt, probe_nppplus.txt |

No pattern found: input depth (8u, 32u), pixel values, image width/height for a pattern within a tile.
Not tested: 16u in the probe (covered in the report's section A), pitch beyond cudaMallocPitch (report section 5), other GPUs.

Files: probe.cu (includes ../files code/npp_debug.cu; build: `nvcc -std=c++17 -O2 -I<npp include> -I<dir of npp_debug.cu> probe.cu -ldl`),
cuts.py, pieces.py, roots.py (CPU, inputs from ~/ccl-blog/npp-vis, npp-bundle2/data and ../rerun).
NPP+ was loaded by symlinking libnpp_plus_c.so / libnpp_plus_if.so as libnppc.so.12 / libnppif.so.12 (same C symbol names).
