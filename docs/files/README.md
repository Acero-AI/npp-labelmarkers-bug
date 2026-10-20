# nppiLabelMarkersUF splits connected regions: evidence

The full write-up is the report page this archive came with. Check the files first:

    sha256sum -c SHA256SUMS

## What each folder holds, and where it came from

| folder | contents | produced by |
|---|---|---|
| `code/` | `npp_debug.cu` (the harness), `npp_min.cu` (one-call repro, the same file as `repro/npp_min.cu`), `check_labels.py` (GPU-free judge), `npp_truth.py` (cc3d + our labeller), `npp_vm.sh`, `run_all.sh`, `hyp.sh`, `kernels.cuh` (our labeller, for the fix) | us |
| `repro/` (in the archive; `repro/` at the top of the repository) | `run_clean.sh`, the one-command rerun, and the programs it builds: `npp_min.cu`, `npp_batch.cpp`, `npp_bufsize.cpp`, `probe_values.cpp`, plus `check_labels.py` and `check.py` | us |
| `inputs/` | the small test images and NVIDIA's PCB and CT_skull images as one uint32 per pixel; `manifest.json` gives `[W, H, connectivity]` | us |
| `inputs/nvidia_samples/` | the five 8-bit images shipped with cuda-samples, unchanged (md5-identical to tag v12.5) | NVIDIA |
| `inputs/grid/` | the 50 two-arm shapes (V: 8 neighbours, U: 4 neighbours, arm heights 1 to 5) | us |
| `npp_outputs/` | NPP's raw labels for `inputs/*.u32`: `_npp11` = NPP 11.8.0, `_npp13` = NPP 13.2.0, both on the RTX 3070, written by `npp_debug LIB --dump` | NPP |
| `nvidia_committed/` | label files NVIDIA committed to cuda-samples at tag v12.8 (`bin/x86_64/linux/release/`), with NVIDIA's own run log, which reports NPP 12.0.0 | NVIDIA |
| `truth/` | labels from cc3d 4.1.0 and from our labeller (Metal, Apple M5 Pro) for every input, byte-identical pairs; `truth.json` lists the counts | `npp_truth.py` |
| `logs/3070/`, `logs/4060/` | one harness log per NPP version (`11.txt 12.txt 13.txt sys.txt`), `machine.txt` (`nvidia-smi`, `nvcc --version`), the 150-shape grid (both GPUs); rates and shape catalogues (3070 only) | `npp_vm.sh`, `run_all.sh` |
| `rerun/4060ti_2026-10-19/` | everything from `code/gpu_run.sh` on a freshly rented RTX 4060 Ti with pinned wheels (`versions.txt` has their sha256): `min/` npp_min per version, `npp_logs/` the harness, `sample12/` NVIDIA's sample (10 runs, run 1's files kept), `nppplus/` the NPP+ 0.10.0 sample (5 runs, run 1's files kept), `ours/` our labeller (50 runs per image) | `gpu_run.sh` |
| `rerun/a100_2026-10-19/` | the same script on an A100-SXM4-40GB (driver 570.133.20; NPP 13 does not run on that driver); its NPP+ sample printed "failed" in all five runs | `gpu_run.sh` |
| `rerun/clean_3070_2026-10-20/` | the one-command rerun on an RTX 3070 (driver 580.159.03): each NPP version with its own release's CUDA runtime and headers, built with g++; `summary.txt`, `npp_min_*`, `npp_batch_*`, `npp_bufsize_*`, `ldd_*` (what each binary loaded), `wheels.sha256`, NVIDIA's committed files judged at tags v12.5 and v12.8, `sample12/` NVIDIA's sample (10 runs, run 1's files kept, every run judged), `probe/` (`probe_values_*.txt`: other pixel values, 0 as an ordinary value, the row-end join; `npp_min_*.txt` from the four-region `npp_min.cu`; `ldd_*`), `reviewer_t184854/` (forum thread 184854's own input) | `../../repro/run_clean.sh`; `probe/` by hand, see below |
| `rerun/second_run_3070_2026-10-20/` | `run_clean_output/`: a reviewer's fresh run of this archive's `run_clean.sh`, unpacked anew, on the same rented RTX 3070, 2026-10-20 11:21 UTC (label files of the sample runs left out); `reviewer_probes/`: that reviewer's own program (`a3probe.cpp`, built by `mine.sh` against each version) with its outputs: the 3 × 3 at every offset of 70 × 70 and 135 × 135, the batch call with two images per call, and the row-end join at widths 3 to 130 | `run_clean.sh`; `a3probe.cpp` |
| `patterns/` | the second pass: what the failure depends on (README.md lists each pattern and its log), `probe.cu`, the CPU analyses, and `rule.py`, which checks the section 9 rule against all 150 measured shapes | `probe.cu`, `*.py` |
| `nvidia_committed/nppplus/` | NPP+ sample inputs and the label files NVIDIA committed to CUDALibrarySamples | NVIDIA |
| `logs/4060/hypotheses/` | the misuse hypotheses, and NVIDIA's sample program: `sample_build.txt` is a first build against CUDA 13 that fails to compile; `sample12_build_and_run.txt` is the build against NPP 12.4.1 that ran; `nvidia_sample_output_checked.txt` judges the 20 files it wrote | `hyp.sh` and the command in `sample12_build_and_run.txt` |

## Known blemishes, left as they are

- In the first runs (`logs/`) the NPP wheels were installed without pinned versions and the exact builds were not recorded;
  each log's first lines give the version NPP reports (`nppGetLibVersion`). The rerun in `rerun/` pins them
  (`nvidia-npp-cu11==11.8.0.86`, `nvidia-npp-cu12==12.4.1.87`, `nvidia-npp==13.2.0.58`) and reproduces every result.
- `logs/3070/grid13.txt` ends with an ssh disconnect message and repeats some 12.x lines that `grid_rest.txt` also holds.
- In the harness logs, section A's last row is labelled "(truth: 5)"; the computed truth on the same line (6) is right.
- `hyp.sh`, `run_all.sh` and `gpu_run.sh` use the paths of the machines they ran on (`/root/...`); `repro/run_clean.sh` is the one meant to be rerun.
- The 2026-10-19 runs (`rerun/4060ti_…`, `rerun/a100_…`) and `logs/` were built with each machine's CUDA toolkit, so NPP 11.8 and 12.4.1 ran beside a newer CUDA runtime than their own. `rerun/clean_3070_2026-10-20/` pairs every version with its own runtime and reproduces the results.
- The batch calls in `npp_debug.cu` and `probe.cu` pass one image, which the documentation forbids (`nBatchSize` must be > 1); `logs/*` section A's batch row and `patterns/batchmap_*.txt` come from that call. `repro/npp_batch.cpp` makes the documented call.
- `rerun/clean_3070_2026-10-20/` is one run of an earlier `run_clean.sh`, touched by hand afterwards: `npp_batch_*.txt` were rerun the same day with the final `npp_batch.cpp`; the CompressedSegmentLabels lines were removed from `nvidia_committed_v12.8.txt` and the summary line recounted; the note at the end of `summary.txt` was added. `probe/` was added the same day on the same machine: `repro/probe_values.cpp` and the four-region `repro/npp_min.cu`, each built with g++ against every NPP version and its own CUDA runtime exactly as `run_clean.sh` builds them (the `ldd_*` files show what each loaded), with the cases `run_clean.sh` runs. `reviewer_t184854/` is a reviewer's program for forum thread 184854's own input (`t184854.cpp`), its output on 11.8.0, 12.4.1 and 13.2.0 (`t184854_output.txt`), and the reviewer's own labeller and judges. A fresh run of the current archive's `run_clean.sh` on the same rented RTX 3070, later the same day, reproduced every verdict: `rerun/second_run_3070_2026-10-20/`.
- Our logs say "false merges" for the number of times an NPP label is seen covering a second true region;
  `check_labels.py` says "merged labels" for the number of NPP labels that cover more than one region.
