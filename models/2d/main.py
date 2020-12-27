import sys
import os
from random import shuffle
import numpy as np
from pathlib import Path

from slices_generator import SlicesGenerator

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

def main(path):
    # Path leads to the where the data is Ultrecht, Singapore etc...

    # Get train generator
    gts, slices = construct_dataset(path)
    sz = len(gts)

    indices = np.arange(0, sz)
    shuffle(indices)
    # Randomize elements
    gts = gts[indices]
    slices = slices[indices]

    # Do a train test split => Layer ?

    # Create generator for train_data
    # We do not want to load all the data into memory so we will give paths instead
    slices_generator = SlicesGenerator(slices, gts, batch_size=1)


    # Let's make it
if __name__ == "__main__":
    main(sys.argv[1])