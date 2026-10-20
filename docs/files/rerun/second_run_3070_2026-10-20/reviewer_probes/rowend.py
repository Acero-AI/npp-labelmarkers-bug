import sys, numpy as np
from scipy import ndimage
T=sys.argv[1]; W=H=512
img=np.fromfile(f"{T}/inputs/nvidia_samples/teapot_512x512_8u.raw",np.uint8)[:W*H].reshape(H,W)
for f in ["teapot_LabelMarkersUF_8Way_512x512_32u.raw"]:
    lab=np.fromfile(f"{T}/nvidia_committed/{f}",np.uint32).reshape(H,W)
    truth=np.zeros((H,W),np.int64); n=0
    for v in np.unique(img):
        t,k=ndimage.label(img==v,structure=np.ones((3,3))); truth[t>0]=t[t>0]+n; n+=k
    merged=0; rowend=0
    for l in np.unique(lab):
        m=lab==l; ts=np.unique(truth[m])
        if len(ts)>1:
            merged+=1
            ys,xs=np.nonzero(m)
            # is there a pair (W-1,y),(0,y+1) in this label belonging to different truths?
            pts=set(zip(ys.tolist(),xs.tolist()))
            rowend+=any((y+1,0) in pts for (y,x) in pts if x==W-1)
    print(f, "merged labels", merged, "of which contain a row-end pair", rowend)
