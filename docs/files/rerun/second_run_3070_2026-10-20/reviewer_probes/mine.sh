cd /root/audit3-npp; mkdir -p mine; R=/root/repro
for v in 11 12 13; do E=$R/env$v
 INC=$(dirname $(find $E -name npp.h|head -1)); CINC=$(dirname $(find $E -name cuda_runtime.h|head -1)); HINC=$(dirname $(dirname $(find $E -path "*crt/host_config.h"|head -1))); LIB=$(dirname $(find $E -name "libnppif.so.*"|head -1)); CLIB=$(dirname $(find $E -name "libcudart.so.*"|head -1))
 g++ -O2 -x c++ a3probe.cpp -I$INC -I$CINC -I$HINC -L$LIB -L$CLIB -l:$(basename $(ls $LIB/libnppif.so.*|head -1)) -l:$(basename $(ls $LIB/libnppc.so.*|head -1)) -l:$(basename $(ls $CLIB/libcudart.so.*|head -1)) -Wl,-rpath,$LIB -Wl,-rpath,$CLIB -o mine/a3_$v > mine/build_$v.txt 2>&1
 ldd mine/a3_$v | grep -E "npp|cudart" > mine/ldd_$v.txt
 mine/a3_$v > mine/out_$v.txt 2>&1
done
echo done > mine/done
