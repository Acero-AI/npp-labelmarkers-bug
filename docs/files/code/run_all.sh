#!/bin/bash
# Everything for a second GPU: bash run_all.sh   -> results in npp_logs/, rates*.txt, grid.txt, probe_out/
cd "$(dirname "$0")"
export PATH=/usr/local/cuda/bin:$PATH
bash npp_vm.sh > run.log 2>&1          # sections A-F + NVIDIA sample images, all versions
for v in 13 12 11; do
  L=$(dirname $(find npp$v -name "libnppif.so*" | head -1)); R=$(dirname $(find npp$v -name "libcudart.so*" | head -1))
  LD_LIBRARY_PATH=$R:$L ./npp_debug $L --rates > rates$v.txt 2>&1
  python3 -c "import json;[print(k,*v) for k,v in json.load(open('grid/manifest.json')).items()]" | while read name W H conn; do
    echo "$v $name $(LD_LIBRARY_PATH=$R:$L ./npp_debug $L --dump grid/$name.u32 $W $H $conn /dev/null | tail -1 | grep -o 'broken links [0-9]*, false merges [0-9]*')"
  done >> grid.txt
done
mkdir -p probe_out
L=$(dirname $(find npp13 -name "libnppif.so*" | head -1))
python3 -c "import json;[print(k,*v) for k,v in json.load(open('probe/manifest.json')).items()]" | while read name W H conn; do
  LD_LIBRARY_PATH=$L ./npp_debug $L --dump probe/$name.u32 $W $H $conn probe_out/$name.u32 | tail -1
done > probe.txt
tar czf results.tgz npp_logs rates*.txt grid.txt probe.txt probe_out run.log
echo ALLDONE > all.done
