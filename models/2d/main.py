import sys
from slices_generator import SlicesGenerator
from construct_dataset import construct_slices
from unet import get_model

from keras.callbacks import ModelCheckpoint

def main(path):
    # Path leads to the where the data is Ultrecht, Singapore etc...

    # Get train generator
    X, Y = construct_slices(path)
    # Do a train test split => Layer ?
    gen = SlicesGenerator(X, Y, batch_size=32)
    model = get_model((572, 572), 3)
    checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True)

    model.fit(
        gen,
        verbose=1,
        callbacks=checkpoint)
if __name__ == "__main__":
    main(sys.argv[1])