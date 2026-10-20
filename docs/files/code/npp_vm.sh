#!/bin/bash
# Run the NPP investigation on any machine with an NVIDIA GPU and a CUDA toolkit:
#     bash npp_vm.sh            (from a folder holding npp_debug.cu and data/*.raw)
# Writes npp_logs/<version>.txt for NPP from CUDA 11, 12, 13 and the system toolkit.
cd "$(dirname "$0")"
mkdir -p npp_logs
{ nvidia-smi; nvcc --version 2>/dev/null || /usr/local/cuda/bin/nvcc --version; } > npp_logs/machine.txt 2>&1
NVCC=$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)
PY=$(command -v python3)

# One folder per NPP version: they install to the same paths, so they cannot share one environment.
for v in 11 12 13; do
  [ -d npp$v ] && continue
  case $v in
    11) pkgs="nvidia-npp-cu11 nvidia-cuda-runtime-cu11" ;;
    12) pkgs="nvidia-npp-cu12 nvidia-cuda-runtime-cu12" ;;
    13) pkgs="nvidia-npp nvidia-cuda-runtime" ;;
  esac
  $PY -m pip install -q --target npp$v $pkgs || echo "pip failed for CUDA $v"
done

INC=$(dirname "$(find npp12 npp13 /usr/local/cuda/include -name nppdefs.h 2>/dev/null | head -1)")
$NVCC -std=c++17 -O2 -I"$INC" npp_debug.cu -ldl -o npp_debug || { echo "build failed"; exit 1; }

dirs_for() {  # NPP lib folder and the CUDA runtime folder it needs
  case $1 in
    sys) echo "/usr/local/cuda/lib64 /usr/local/cuda/lib64" ;;
    *)   echo "$(dirname "$(find npp$1 -name 'libnppif.so*' | head -1)") $(dirname "$(find npp$1 -name 'libcudart.so*' | head -1)")" ;;
  esac
}
for v in sys 13 12 11; do
  read lib rt <<< "$(dirs_for $v)"
  [ -n "$lib" ] && [ -d "$lib" ] || { echo "no NPP for $v" > npp_logs/$v.txt; continue; }
  echo "== NPP $v from $lib"
  {
    LD_LIBRARY_PATH="$rt:$lib" ./npp_debug "$lib"
    echo; echo "== NVIDIA sample images"
    for f in teapot_512x512_8u.raw:512:512 CT_skull_512x512_8u.raw:512:512 PCB_METAL_509x335_8u.raw:509:335 PCB2_1024x683_8u.raw:1024:683 PCB_1280x720_8u.raw:1280:720; do
      IFS=: read name w h <<< "$f"
      LD_LIBRARY_PATH="$rt:$lib" ./npp_debug "$lib" --raw data/$name $w $h | tail -2
    done
  } > npp_logs/$v.txt 2>&1
  tail -3 npp_logs/$v.txt
done
echo done
