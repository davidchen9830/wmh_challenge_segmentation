import sys
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
from construct_dataset import construct_dataset

def main(path):
    gts, slices = construct_dataset(path)
    indices = np.arange(len(gts))
    np.random.shuffle(indices)

    gts = gts[indices]
    slices = slices[indices]

    # The train test split shuffle by itself
    X_train, X_test, Y_train, Y_test = train_test_split(slices, gts, test_size=0.33, random_state=42)

    np.save('X_train', X_train)
    np.save('X_test', X_test)
    np.save('Y_train', Y_train)
    np.save('Y_test', Y_test)

    # We also need to know the number of slices for each sample
    elts = [
        (X_train, 'X_train_slices'),
        (X_test, 'X_test_slices')
    ]
    for i in range(len(elts)):
        data, filename = elts[i]
        result = []
        for j in range(len(data)):
            img = data[j]
            # Big hack
            img1, _ = img
            header = (nib.load(img1)).header
            _, _, channels = header.get_data_shape()
            for c in range(channels):
                IMG_IDX = j
                IMG_SLICE = c
                result.append((IMG_IDX, IMG_SLICE))
         
        result = np.array(result)
        np.random.shuffle(result)
        np.save(filename, result)

if __name__ == "__main__":
    main(sys.argv[1])