import pickle
import sys
from pathlib import Path

import numpy as np
from generator import Generator2D, Generator3D, KFold
from unet import get_model
from tensorflow.keras.callbacks import ModelCheckpoint


def main(path, preprocess, dimensions):
    assert preprocess == 0 or preprocess == 1
    assert dimensions == 2 or dimensions == 3
    with path.open('rb') as file:
        dataset = pickle.load(file)

    generator = {2: Generator2D, 3: Generator3D}[dimensions](dataset['X'], dataset['y'], preprocess=preprocess == 1, batch_size=20)
    train_gen = KFold(generator, folds=5, validation=False)
    val_gen = KFold(generator, folds=5, validation=True)

    # Models
    model = get_model((200, 200), {0: 2, 1: 3}[preprocess] * {2: 1, 3: 3}[dimensions])
    # model.load_weights('best_model.h5')
    checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True)
    if preprocess:
        checkpoint = ModelCheckpoint("best_model_preprocess.h5", save_best_only=True)

    model.fit(
        x=train_gen,
        epochs=20,
        callbacks=[checkpoint],
        validation_data=val_gen
    )


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: main.py <path/to/dataset.pickle> <preprocess:0|1> <3d:2|3>")
        sys.exit(1)
    main(Path(sys.argv[1]), preprocess=int(sys.argv[2]), dimensions=int(sys.argv[3]))
