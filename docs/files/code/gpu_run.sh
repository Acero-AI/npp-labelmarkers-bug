#!/bin/bash
# Reruns every GPU claim of the NPP report on one machine and keeps every output file.
#     bash gpu_run.sh            (inside the unpacked bundle; NVIDIA driver >= 580, nvcc, python3, git)
# Results: out/  (progress.txt says which steps ran; each step writes its own log)
set -u
cd "$(dirname "$0")"; B=$PWD; O=$B/out; mkdir -p $O/sample12 $O/nppplus $O/ours $O/min
NVCC=$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)
PY=${PY:-python3}
step() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $O/progress.txt; }
IMGS="teapot_512x512_8u.raw:512:512 CT_skull_512x512_8u.raw:512:512 PCB_METAL_509x335_8u.raw:509:335 PCB2_1024x683_8u.raw:1024:683 PCB_1280x720_8u.raw:1280:720"

step "0 machine"
{ date -u; nvidia-smi; nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv; $NVCC --version; uname -a; $PY --version; } > $O/machine.txt 2>&1

step "1 NPP, pinned"
declare -A PK=([11]="nvidia-npp-cu11==11.8.0.86 nvidia-cuda-runtime-cu11==11.8.89"
               [12]="nvidia-npp-cu12==12.4.1.87 nvidia-cuda-runtime-cu12==12.9.79"
               [13]="nvidia-npp==13.2.0.58 nvidia-cuda-runtime==13.4.92")
for v in 11 12 13; do [ -d npp$v ] || $PY -m pip install -q --no-deps --target npp$v ${PK[$v]} || step "pip failed for $v"; done
$PY -m pip install -q --target pylib numpy scipy
{ for v in 11 12 13; do echo "== npp$v: ${PK[$v]}"; ls -d npp$v/*.dist-info; find npp$v -name 'libnpp[ci]f*.so*' -o -name 'libnppc.so*' -o -name 'libcudart.so*' | xargs sha256sum; done
  echo "== system toolkit"; ls -l /usr/local/cuda/lib64/libnppif.so*; sha256sum /usr/local/cuda/lib64/libnppif.so.* /usr/local/cuda/lib64/libnppc.so.*
  echo "== checker"; PYTHONPATH=pylib $PY -c "import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)"; } > $O/versions.txt 2>&1
libdir() { [ $1 = sys ] && echo /usr/local/cuda/lib64 || dirname "$(find npp$1 -name 'libnppif.so*' | head -1)"; }
rtdir()  { [ $1 = sys ] && echo /usr/local/cuda/lib64 || dirname "$(find npp$1 -name 'libcudart.so*' | head -1)"; }
INC=$(dirname "$(find npp13 -name nppdefs.h | head -1)")

step "2 npp_min, the 3 x 3 repro, 20 runs per version"
for v in sys 13 12 11; do
  L=$(libdir $v); so=$(basename "$(ls $L/libnppif.so.* | head -1)"); sc=$(basename "$(ls $L/libnppc.so.* | head -1)")
  $NVCC -O2 -arch=native -I"$INC" code/npp_min.cu -L$L -l:$so -l:$sc -Xlinker -rpath=$L -o npp_min_$v > $O/min/build_$v.txt 2>&1 || { step "npp_min build failed for $v"; continue; }
  for i in $(seq 20); do LD_LIBRARY_PATH=$(rtdir $v):$L ./npp_min_$v; echo "exit $?"; done > $O/min/run_$v.txt 2>&1
  step "   $v: $(grep -c WRONG $O/min/run_$v.txt) wrong, $(grep -c ': correct' $O/min/run_$v.txt) correct, $(grep -c -i 'error\|status' $O/min/run_$v.txt) failed to run, of 20; $(head -1 $O/min/run_$v.txt)"
done

step "3 harness, sections A-F and NVIDIA's images, every version"
cp code/npp_vm.sh code/npp_debug.cu .
bash npp_vm.sh > $O/npp_vm_run.txt 2>&1; cp -r npp_logs $O/
for v in 13 12 11; do
  L=$(libdir $v); R=$(rtdir $v)
  LD_LIBRARY_PATH=$R:$L ./npp_debug $L --rates > $O/rates$v.txt 2>&1
  LD_LIBRARY_PATH=$R:$L ./npp_debug $L --hyp > $O/hyp$v.txt 2>&1
  $PY -c "import json;[print(k,*v) for k,v in json.load(open('grid/manifest.json')).items()]" | while read name W H conn; do
    echo "$v $name $(LD_LIBRARY_PATH=$R:$L ./npp_debug $L --dump grid/$name.u32 $W $H $conn /dev/null | tail -1 | grep -o 'broken links [0-9]*, false merges [0-9]*')"
  done >> $O/grid.txt
done

step "4 NVIDIA's sample (cuda-samples v12.5, unmodified) against NPP 12.4.1; 10 runs, run 1's files kept"
if [ -d sample ]; then
N=$(dirname "$(dirname "$(find $B/npp12 -name 'libnppif.so*' | head -1)")"); R=$(rtdir 12)
( cd sample && $NVCC -O2 -arch=native -I$N/include -ICommon batchedLabelMarkersAndLabelCompressionNPP.cpp -L$N/lib \
    -l:libnppif.so.12 -l:libnppc.so.12 -l:libnppisu.so.12 -Xlinker -rpath=$N/lib -o $B/sample12 ) > $O/sample12/build.txt 2>&1 || step "sample build failed"
for i in $(seq 10); do
  d=$O/sample12/run$i; mkdir -p $d; cp data/*.raw $d/
  ( cd $d && LD_LIBRARY_PATH=$N/lib:$R $B/sample12 > stdout.txt 2>&1 )
  ( cd $d && sha256sum *_8Way_*.raw > SHA256SUMS && rm -f $(cd $B/data && ls *.raw) )
  [ $i -gt 1 ] && rm -f $d/*_8Way_*.raw
done
for g in $O/sample12/run1/*_8Way_*_32u.raw; do
  case $(basename $g) in
    teapot_*) s=teapot_512x512_8u.raw:512:512 ;;   CT_[sS]kull_*) s=CT_skull_512x512_8u.raw:512:512 ;;
    PCB_METAL_*) s=PCB_METAL_509x335_8u.raw:509:335 ;;   PCB2_*) s=PCB2_1024x683_8u.raw:1024:683 ;;   PCB_*) s=PCB_1280x720_8u.raw:1280:720 ;;
  esac; IFS=: read img w h <<< "$s"
  PYTHONPATH=pylib $PY code/check_labels.py data/$img $w $h 8 $g | tail -1
done > $O/sample12/checked.txt 2>&1
grep -h "count is" $O/sample12/run*/stdout.txt | sort | uniq -c > $O/sample12/counts_over_runs.txt
else step "   skipped: no sample/ folder (cuda-samples v12.5 source)"; fi

step "4b NVIDIA's NPP+ 0.10.0 sample (CUDALibrarySamples), unmodified; 5 runs, run 1's files kept"
NPV=libnpp_plus-linux-x86_64-0.10.0.0_cuda12-archive; NP=$B/$NPV; S=$B/cls/NPP+/batchedLabelMarkersAndCompression; R=$(rtdir 12); P=$O/nppplus
[ -d $NP ] || { wget -q https://developer.download.nvidia.com/compute/nppplus/redist/libnpp_plus/linux-x86_64/$NPV.tar.xz && tar xJf $NPV.tar.xz; }
sha256sum $NPV.tar.xz > $P/archive_sha256.txt
[ -d cls ] || { git clone -q --depth 1 --filter=blob:none --sparse https://github.com/NVIDIA/CUDALibrarySamples.git cls && git -C cls sparse-checkout set NPP+/batchedLabelMarkersAndCompression; }
git -C cls log -1 --format="CUDALibrarySamples %H %cI" > $P/sample_commit.txt
mkdir -p $P/w/images $P/w/build && cp $S/images/*_8u.raw $P/w/images/
( cd $P/w/build && $NVCC -std=c++17 -O2 -arch=native -DNPP_PLUS_ON -DNPP_PLUS_ENABLE -I$NP/include -x cu $S/batchedLabelMarkersAndCompressionNPPPlus.cpp \
    -L$NP/lib -lnpp_plus_if -lnpp_plus_c -lnpp_plus_isu -Xlinker -rpath=$NP/lib -o sampleplus ) > $P/build.txt 2>&1 || step "NPP+ sample build failed"
for i in 1 2 3 4 5; do
  ( cd $P/w/build && LD_LIBRARY_PATH=$NP/lib:$R ./sampleplus > $P/stdout$i.txt 2>&1 )
  ( cd $P/w/images && sha256sum *_8Way_*.raw > $P/sha_run$i.txt ); [ $i = 1 ] && cp $P/w/images/*_8Way_*.raw $P/
done
rm -rf $P/w

step "4c NPP+ through the harness (its libraries named as NPP 12: same C symbols)"
mkdir -p nppplus_as_npp && ln -sf $NP/lib/libnpp_plus_c.so nppplus_as_npp/libnppc.so.12 && ln -sf $NP/lib/libnpp_plus_if.so nppplus_as_npp/libnppif.so.12
LD_LIBRARY_PATH=$R:$NP/lib:$B/nppplus_as_npp ./npp_debug $B/nppplus_as_npp > $O/npp_logs/plus.txt 2>&1
LD_LIBRARY_PATH=$R:$NP/lib:$B/nppplus_as_npp ./npp_debug $B/nppplus_as_npp --rates > $O/rates_plus.txt 2>&1
head -3 $O/npp_logs/plus.txt | tee -a $O/progress.txt

step "5 our labeller, same GPU, 50 runs per image"
[ -d colorcc ] || step "   skipped: no colorcc/ folder"
[ -d colorcc ] && ( $PY -m pip install -q --no-build-isolation ./colorcc && cd $O/ours && $PY $B/rent/ours_repeat.py 50 . $(for f in $IMGS; do echo $B/data/$f; done) ) > $O/ours/log.txt 2>&1
[ -d colorcc ] && for f in $IMGS; do IFS=: read img w h <<< "$f"; name=${img%_*_8u.raw}; name=${name%_${w}x${h}}
  PYTHONPATH=pylib $PY code/check_labels.py data/$img $w $h 8 $O/ours/${name}_ours.u32 | tail -1; done >> $O/ours/log.txt 2>&1

step "6 pattern probe (patterns/probe.cu) per version, T1-T5 and the batch map"
mkdir -p $O/patterns
if $NVCC -std=c++17 -O2 -arch=native -I"$INC" -Icode patterns/probe.cu -ldl -o probe > $O/patterns/build.txt 2>&1; then
  for v in sys 13 12 11 plus; do
    if [ $v = plus ]; then L=$B/nppplus_as_npp; RT=$(rtdir 12):$NP/lib; else L=$(libdir $v); RT=$(rtdir $v); fi
    LD_LIBRARY_PATH=$RT:$L timeout 900 ./probe $L > $O/patterns/probe_$v.txt 2>&1
    LD_LIBRARY_PATH=$RT:$L timeout 600 ./probe $L --batch-map > $O/patterns/batchmap_$v.txt 2>&1
    step "   probe $v: $(head -3 $O/patterns/probe_$v.txt | grep -o 'NPP [0-9.]*')"
  done
else step "   probe build failed"; fi

step "7 done"; tar czf $B/results.tgz -C $B out; echo ALLDONE > $O/all.done
