// npp_bufsize: the scratch-buffer sizes NPP reports for large images. The documentation says the LabelMarkersUF
// functions "currently only support image ROI sizes up to 4 gigapixels"; these sizes are far below that.
//
//     g++ -O2 npp_bufsize.cpp -I<npp include> -I<cudart include> -L<libs> -l:libnppif.so.N -l:libnppc.so.N -l:libcudart.so.N
#include <cstdio>
#include <cuda_runtime.h>
#include <npp.h>

int main() {
    const NppLibraryVersion* v = nppGetLibVersion();
    printf("NPP %d.%d.%d\n", v->major, v->minor, v->build);
    const int sizes[][2] = {{8400, 6000}, {16800, 12000}, {25200, 18000}, {33600, 24000}};
    for (auto& s : sizes) {
        NppiSize roi = {s[0], s[1]};
        int label = 0, compress = 0;
        NppStatus a = nppiLabelMarkersUFGetBufferSize_32u_C1R(roi, &label);
        NppStatus b = nppiCompressMarkerLabelsGetBufferSize_32u_C1R(s[0] * s[1], &compress);
        printf("%6d x %6d (%4.0f megapixels): LabelMarkersUF buffer %13d bytes (status %d), CompressMarkerLabels buffer %13d bytes (status %d)\n",
               s[0], s[1], s[0] * (double)s[1] / 1e6, label, a, compress, b);
    }
    return 0;
}
