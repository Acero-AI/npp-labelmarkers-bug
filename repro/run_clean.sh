#!/bin/bash
# run_clean.sh: every GPU experiment in this report, rebuilt from nothing on a Linux machine with an NVIDIA GPU.
#
#     bash repro/run_clean.sh            (results in repro/out-<date>/, summary in summary.txt)
#
# Needs python3 with pip, g++, git and an NVIDIA driver. It does not use a CUDA toolkit or nvcc: each NPP version
# comes from NVIDIA's pip wheel together with the CUDA runtime wheel of the same CUDA release, and every program is
# built with g++ against exactly those headers and libraries. `ldd` of every binary is saved, so the log shows which
# NPP and which CUDA runtime each run really loaded.
set -u
cd "$(dirname "$0")"; R=$PWD; O=$R/out-$(date -u +%Y%m%d-%H%M%S); mkdir -p "$O/bin"
PY=${PY:-python3}
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$O/progress.txt"; }

# NPP and the CUDA runtime from the same CUDA release: 11.8, 12.9 and 13.4.
declare -A NPP=([11]="nvidia-npp-cu11==11.8.0.86" [12]="nvidia-npp-cu12==12.4.1.87" [13]="nvidia-npp==13.2.0.58")
declare -A CRT=([11]="nvidia-cuda-runtime-cu11==11.8.89" [12]="nvidia-cuda-runtime-cu12==12.9.79" [13]="nvidia-cuda-runtime==13.4.92")
# cuda_runtime.h includes crt/host_config.h, which ships in these wheels of the same releases (only headers are used)
declare -A HDR=([11]="nvidia-cuda-nvcc-cu11==11.8.89" [12]="nvidia-cuda-nvcc-cu12==12.9.86" [13]="nvidia-cuda-crt==13.4.92")

log "machine"
{ nvidia-smi; uname -a; g++ --version | head -1; $PY --version; } > "$O/machine.txt" 2>&1

log "checker: numpy and scipy in their own folder"
$PY -m pip install -q --target "$R/pydeps" numpy scipy > "$O/pip_checker.txt" 2>&1
check() { PYTHONPATH="$R/pydeps" $PY "$R/check_labels.py" "$@"; }

for v in 11 12 13; do
  E=$R/env$v
  log "NPP from CUDA $v: ${NPP[$v]} with ${CRT[$v]} (headers ${HDR[$v]})"
  if [ ! -d "$E/nvidia" ]; then
    $PY -m pip download -q --no-deps -d "$E/wheels" "${NPP[$v]}" "${CRT[$v]}" "${HDR[$v]}" > "$O/pip_$v.txt" 2>&1
    $PY -m pip install -q --no-deps --target "$E" "$E"/wheels/*.whl >> "$O/pip_$v.txt" 2>&1
  fi
  sha256sum "$E"/wheels/*.whl >> "$O/wheels.sha256"
  INC=$(dirname "$(find "$E" -name npp.h | head -1)"); CINC=$(dirname "$(find "$E" -name cuda_runtime.h | head -1)")
  HINC=$(dirname "$(dirname "$(find "$E" -path '*crt/host_config.h' | head -1)")")
  LIB=$(dirname "$(find "$E" -name 'libnppif.so.*' | head -1)"); CLIB=$(dirname "$(find "$E" -name 'libcudart.so.*' | head -1)")
  NPPIF=$(basename "$(ls "$LIB"/libnppif.so.* | head -1)"); NPPC=$(basename "$(ls "$LIB"/libnppc.so.* | head -1)")
  CUDART=$(basename "$(ls "$CLIB"/libcudart.so.* | head -1)")
  for prog in npp_min.cu npp_batch.cpp npp_bufsize.cpp probe_values.cpp; do
    b=$O/bin/${prog%.*}_$v
    g++ -O2 -x c++ "$R/$prog" -I"$INC" -I"$CINC" -I"$HINC" -L"$LIB" -L"$CLIB" -l:"$NPPIF" -l:"$NPPC" -l:"$CUDART" \
        -Wl,-rpath,"$LIB" -Wl,-rpath,"$CLIB" -o "$b" > "$O/build_${prog%.*}_$v.txt" 2>&1 || log "build failed: $prog for $v"
    ldd "$b" > "$O/ldd_${prog%.*}_$v.txt" 2>&1
  done
  log "  npp_min, 20 runs"
  for i in $(seq 20); do "$O/bin/npp_min_$v"; echo "exit $?"; done > "$O/npp_min_$v.txt" 2>&1
  log "  npp_batch (batch size 2, 20 calls)"
  "$O/bin/npp_batch_$v" > "$O/npp_batch_$v.txt" 2>&1; echo "exit $?" >> "$O/npp_batch_$v.txt"
  log "  npp_bufsize"
  "$O/bin/npp_bufsize_$v" > "$O/npp_bufsize_$v.txt" 2>&1
  log "  probe_values"
  { echo "== the 3 x 3 with the values 5/9/7 instead of 0/1/2, 8 neighbours (4 regions)"; "$O/bin/probe_values_$v" 3 3 8 5 9 5 5 9 5 7 5 7
    echo "== the same, 4 neighbours (6 regions)"; "$O/bin/probe_values_$v" 3 3 4 5 9 5 5 9 5 7 5 7
    echo "== 0s are labelled like any value: 0 1 1 1 0 is three regions"; "$O/bin/probe_values_$v" 5 1 8 0 1 1 1 0
    echo "== 4 x 2: row 0 ends with 3 and row 1 starts with 3; they do not touch (6 regions)"; "$O/bin/probe_values_$v" 4 2 8 1 2 2 3 3 4 4 1
  } > "$O/probe_values_$v.txt" 2>&1
done

log "NVIDIA's committed output files (cuda-samples tags v12.5 and v12.8), judged without a GPU"
CS=$R/cuda-samples
[ -d "$CS" ] || git clone -q --filter=blob:none --no-checkout https://github.com/NVIDIA/cuda-samples.git "$CS"
SAMPLE=Samples/4_CUDA_Libraries/batchedLabelMarkersAndLabelCompressionNPP
IMAGES="teapot:512:512 PCB_METAL:509:335 CT_skull:512:512 PCB2:1024:683 PCB:1280:720"
judge_dir() {   # judge_dir DIR_WITH_LABEL_FILES IMAGE_DIR OUT
  for im in $IMAGES; do IFS=: read -r n w h <<< "$im"
    for f in "$1"/"${n}"_*_8Way_"${w}x${h}"_32u.raw; do   # CompressedSegmentLabels files label a different input
      case $f in *CompressedSegmentLabels*) continue;; esac
      [ -f "$f" ] && check "$2/${n}_${w}x${h}_8u.raw" "$w" "$h" 8 "$f"; done
  done | sed "s#$1/##" > "$3"
}
for tag in v12.5 v12.8; do
  (cd "$CS" && git -c advice.detachedHead=false checkout -q "$tag" && git sparse-checkout init --no-cone 2>/dev/null \
     && git sparse-checkout set "/$SAMPLE/" "/Common/" "/bin/x86_64/linux/release/") >> "$O/git.txt" 2>&1
  echo "$tag $(git -C "$CS" rev-parse HEAD)" >> "$O/cuda-samples-commits.txt"
  committed=$CS/$SAMPLE; [ "$tag" = v12.8 ] && committed=$CS/bin/x86_64/linux/release
  judge_dir "$committed" "$CS/Common/data" "$O/nvidia_committed_$tag.txt"
done

log "NVIDIA's sample program, unmodified (tag v12.5), built with g++ against NPP 12.4.1 and CUDA runtime 12.9, 10 runs"
(cd "$CS" && git checkout -q v12.5) >> "$O/git.txt" 2>&1
E=$R/env12
INC=$(dirname "$(find "$E" -name npp.h | head -1)"); CINC=$(dirname "$(find "$E" -name cuda_runtime.h | head -1)")
HINC=$(dirname "$(dirname "$(find "$E" -path '*crt/host_config.h' | head -1)")")
LIB=$(dirname "$(find "$E" -name 'libnppif.so.*' | head -1)"); CLIB=$(dirname "$(find "$E" -name 'libcudart.so.*' | head -1)")
g++ -O2 -x c++ "$CS/$SAMPLE/batchedLabelMarkersAndLabelCompressionNPP.cpp" -I"$INC" -I"$CINC" -I"$HINC" -I"$CS/Common" \
    -L"$LIB" -L"$CLIB" -l:libnppif.so.12 -l:libnppc.so.12 -l:libnppisu.so.12 -l:libcudart.so.12 \
    -Wl,-rpath,"$LIB" -Wl,-rpath,"$CLIB" -o "$O/bin/sample12" > "$O/build_sample12.txt" 2>&1 || log "sample build failed"
ldd "$O/bin/sample12" > "$O/ldd_sample12.txt" 2>&1
for i in $(seq 10); do
  mkdir -p "$O/sample12/run$i"; cp "$CS"/Common/data/*_8u.raw "$O/sample12/run$i/"
  (cd "$O/sample12/run$i" && "$O/bin/sample12" > stdout.txt 2>&1)
  judge_dir "$O/sample12/run$i" "$CS/Common/data" "$O/sample12/run$i/checked.txt"
  [ "$i" -gt 1 ] && rm -f "$O/sample12/run$i"/*_32u.raw   # keep run 1's label files, the judgements of all runs
  rm -f "$O/sample12/run$i"/*_8u.raw
done

log "summary"
{
  echo "machine: $(grep -m1 -o 'NVIDIA [A-Z][^|]*[0-9]' "$O/machine.txt" | head -1); driver $(grep -m1 -o 'Driver Version: [0-9.]*' "$O/machine.txt")"
  for v in 11 12 13; do
    echo; echo "== NPP from CUDA $v"; grep -m1 '^NPP' "$O/npp_min_$v.txt"
    echo "loaded: $(grep -oE 'lib(nppif|nppc|cudart)\.so[.0-9]* => [^ ]+' "$O/ldd_npp_min_$v.txt" | tr '\n' ' ')"
    echo "npp_min: $(grep -c 'exit 2' "$O/npp_min_$v.txt") of 20 runs wrong, $(grep -c 'exit 0' "$O/npp_min_$v.txt") right"
    grep 'images wrong' "$O/npp_batch_$v.txt" | sed 's/^/npp_batch: /' 
    grep 'megapixels' "$O/npp_bufsize_$v.txt" | tail -2
  done
  for tag in v12.5 v12.8; do echo; echo "== NVIDIA's committed files at $tag: $(grep -c WRONG "$O/nvidia_committed_$tag.txt") of $(wc -l < "$O/nvidia_committed_$tag.txt") wrong"; done
  echo; echo "== NVIDIA's sample, NPP 12.4.1, 10 runs: files wrong per run"
  for i in $(seq 10); do echo "run $i: $(grep -c WRONG "$O/sample12/run$i/checked.txt") of $(wc -l < "$O/sample12/run$i/checked.txt")"; done
} > "$O/summary.txt"
cat "$O/summary.txt"
log "done: $O"
