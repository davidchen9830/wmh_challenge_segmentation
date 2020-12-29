import sys
import numpy as np
from sklearn.model_selection import train_test_split
from construct_dataset import construct_dataset

def main(path):
    gts, slices = construct_dataset(path)
    indices = np.arange(len(gts))
    np.random.shuffle(indices)

    gts = gts[indices]
    slices = slices[indices]

    X_train, X_test, Y_train, Y_test = train_test_split(slices, gts, test_size=0.33)

    np.save('X_train', X_train)
    np.save('X_test', X_test)
    np.save('Y_train', Y_train)
    np.save('Y_test', Y_test)

if __name__ == "__main__":
    main(sys.argv[1])