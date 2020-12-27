import numpy as np
import nibabel as nib
from pathlib import Path

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

    Y = [nib.load(gt).get_fdata() for gt in  gts]
    X = [(nib.load(slice[0]).get_fdata(), nib.load(slice[1]).get_fdata()) for slice in slices]
    return np.array(Y), np.array(X)

# Let's use the same function for 2d and 3d construction of slices
# The argument to specify is dimension, by default 2d slice
def construct_slices(path, dimension=2):
    gts, slices = construct_dataset(path)
    # Bound checking
    assert(len(gts) == len(slices))
    sz = len(gts)
    X = []
    Y = []
    # Typically this is like [FLAIR, T1, preprocess?]
    # With shapes like       [(h, w, c), ....]
    nb_channels = len(slices[0]) # Get the number of components
    for i in range(sz):
        h1, w1, c1 = gts[i].shape
        for channel in range(c1):
            # First slice take the first component
            assert(slices[i][0].shape == (h1, w1, c1))
            new_slice = np.array((slices[i][0])[:, :, channel]).reshape(h1, w1, 1)
            # This lead to new_slice to look like this as a volume
            """ flair, t1, preprocess ?
                flair, t1, preprocess ?
                flair, t1, preprocess ?
                flair, t1, preprocess ?
                flair, t1, preprocess ?
                flair, t1, preprocess ? """
            for component in range(1, nb_channels):
                curr_component = slices[i][component]
                # Mb remove in prod ?
                assert(curr_component.shape == (h1, w1, c1))
                curr_slice = ((slices[i][component])[:, :, channel]).reshape(h1, w1, 1)
                new_slice = np.concatenate((new_slice, curr_slice), axis=-1)
            # Check stacking
            assert(new_slice.shape == (h1, w1, nb_channels))
            X.append(new_slice)
            Y.append(gts[i][:, :, channel])
            
    return np.array(X), np.array(Y)
            
