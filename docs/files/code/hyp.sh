#!/bin/bash
cd /root/npp
export PATH=/usr/local/cuda/bin:$PATH
INC=$(dirname $(find npp12 -name nppdefs.h | head -1))
nvcc -std=c++17 -O2 -I$INC /root/hyp/npp_debug.cu -ldl -o npp_debug2 2>&1 | grep error
# 1. NVIDIA's own sample, unmodified, built against this machine's CUDA and NPP
mkdir -p /root/hyp/run && cd /root/hyp/run && cp /root/npp/data/*.raw .
nvcc -O2 -I../sample/Common ../sample/batchedLabelMarkersAndLabelCompressionNPP.cpp -lnppif -lnppc -lnppisu -o sample > ../sample_build.txt 2>&1 || echo "sample build failed" >> ../sample_build.txt
./sample > ../sample_run.txt 2>&1
cd /root/npp
L=$(dirname $(find npp13 -name "libnppif.so*" | head -1))
{
for f in teapot:512:512 CT_skull:512:512 PCB_METAL:509:335 PCB2:1024:683 PCB:1280:720; do
  IFS=: read n w h <<< "$f"
  for kind in LabelMarkersUF CompressedMarkerLabelsUF LabelMarkersUFBatch CompressedMarkerLabelsUFBatch; do
    out=/root/hyp/run/${n}_${kind}_8Way_${w}x${h}_32u.raw
    [ -f $out ] && LD_LIBRARY_PATH=$L ./npp_debug2 $L --check data/${n}_${w}x${h}_8u.raw $w $h $out 8 | tail -1
  done
done
} > /root/hyp/sample_check.txt 2>&1
# 2-7. hypotheses, per NPP version
for v in sys 13 12 11; do
  if [ $v = sys ]; then L=/usr/local/cuda/lib64; R=$L; else L=$(dirname $(find npp$v -name "libnppif.so*" | head -1)); R=$(dirname $(find npp$v -name "libcudart.so*" | head -1)); fi
  echo "== NPP $v"; LD_LIBRARY_PATH=$R:$L ./npp_debug2 $L --hyp 2>&1 | grep -v "^loaded\|^GPU\|^$"
done > /root/hyp/hyp.txt 2>&1
echo D > /root/hyp/done
