"""The label check shown in the NPP post: run it to test the snippet (python check.py)."""
import numpy as np
from scipy import ndimage


def check(img, labels):
    """Count splits (a region given an extra label) and merges (a label given an extra region), 8 neighbours. Both are 0 when right."""
    truth, n = np.zeros(img.shape, np.int64), 0
    for v in np.unique(img):
        t, k = ndimage.label(img == v, structure=np.ones((3, 3)))
        truth[t > 0], n = t[t > 0] + n, n + k
    pairs = np.unique(np.stack([truth.ravel(), labels.ravel()]), axis=1)
    return len(pairs[0]) - len(np.unique(pairs[0])), len(pairs[1]) - len(np.unique(pairs[1]))


if __name__ == "__main__":
    img = np.array([[0, 2, 0], [0, 2, 0], [1, 0, 1]])
    assert check(img, np.array([[0, 1, 2], [0, 1, 0], [6, 0, 8]])) == (1, 0)   # NPP 13.2's answer from the report
    assert check(img, np.array([[0, 1, 0], [0, 1, 0], [6, 0, 8]])) == (0, 0)   # a right answer, any numbering
    assert check(img, np.array([[0, 1, 0], [0, 1, 0], [0, 0, 8]])) == (0, 1)   # a 1 merged into the 0s
    print("ok")
