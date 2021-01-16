import numpy as np
import nibabel as nib
from pathlib import Path
from skimage.transform import resize

def fill_dataset(gts, slices, d):
    # Ground Truth
    wmh = d / "wmh.nii.gz"

    # Data
    flair = d / "orig" / "FLAIR.nii.gz"
    t1 = d / "orig" / "T1.nii.gz"

    gts.append(str(wmh))
    slices.append((str(flair), str(t1))) # Appending a tuple (flair_path, t1_path)

def construct_dataset(path):
    root = Path(path)
    gts, slices = [], []

    singapore = root / "Singapore"
    ultrecht = root / "Utrecht"
    ge3t = root / "GE3T"

    for d in singapore.iterdir():
        fill_dataset(gts, slices, d)
    for d in ultrecht.iterdir():
        fill_dataset(gts, slices, d)
    for d in ge3t.iterdir():
        fill_dataset(gts, slices, d)
    return np.array(gts), np.array(slices)