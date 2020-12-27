import sys
import os
from random import shuffle
import numpy as np
from pathlib import Path
import nibabel as nib
from slices_generator import SlicesGenerator
from construct_dataset import construct_slices


def main(path):
    # Path leads to the where the data is Ultrecht, Singapore etc...

    # Get train generator
    X, Y = construct_slices(path)
    # Do a train test split => Layer ?
    slices_generator = SlicesGenerator(X, Y, batch_size=1)
    # Do something
if __name__ == "__main__":
    main(sys.argv[1])