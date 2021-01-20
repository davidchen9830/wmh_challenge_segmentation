import pickle
import sys
from pathlib import Path

import tensorflow as tf
import numpy as np
import imgaug.augmenters as iaa

from generator import Generator2D, Generator3D, KFold
from unet import get_model
from tensorflow.keras.callbacks import ModelCheckpoint
import skimage.transform
import time

def desquare(image, size):
    w, h = size

    diff = abs(w - h)
    pad_before = diff // 2

    if w < h:
        return image[:, pad_before:w+pad_before]
    elif h < w:
        return image[:, :, pad_before:h+pad_before]
    else:
        return image

def main(path, preprocess, dimensions, weights=None, results=None):
    assert preprocess == 0 or preprocess == 1
    assert dimensions == 2 or dimensions == 3
    with path.open('rb') as file:
        dataset = pickle.load(file)

    model = get_model((208, 208), {0: 2, 1: 3}[preprocess] * {2: 1, 3: 3}[dimensions])
    model.summary()

    if weights:
        model.load_weights(weights)
        fun = {2: Generator2D, 3: Generator3D}[dimensions]
        predicted = []
        predicted_raw = []
        for patient, X, size in zip(dataset['patients'], dataset['X'], dataset['sizes']):
            print(f'Patient: {patient}')
            current_predicted = []
            for slice_index in range(len(X)):
                current_predicted.append(model.predict(np.expand_dims(fun.generate_data(X, slice_index, preprocess=preprocess == 1), 0)))

            current_predicted_raw = np.array(current_predicted).squeeze()
            current_predicted = skimage.transform.resize(current_predicted_raw, (current_predicted_raw.shape[0], max(size), max(size)))
            current_predicted = desquare(current_predicted, size)
            predicted.append(current_predicted)
            predicted_raw.append(current_predicted_raw)

        with open(results, 'wb') as file:
            pickle.dump({
                'raw': predicted_raw,
                'transformed': predicted,
            }, file)
    else:
        augmentor = iaa.Sequential([
            iaa.Affine(
                rotate=(-45, 45),
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            ),
        ])
        generator = {2: Generator2D, 3: Generator3D}[dimensions](dataset['X'], dataset['y'], preprocess=preprocess == 1,
                                                                 batch_size=10, augment=augmentor)
        train_gen = KFold(generator, folds=5, validation=False)
        val_gen = KFold(generator, folds=5, validation=True)

        name = "best_model"
        if preprocess == 0:
            name += "_raw"
        else:
            name += "_preprocessed"
        name += f"_{dimensions}d"
        name += f"_{time.time()}"
        name += ".h5"
        checkpoint = ModelCheckpoint(name, save_best_only=True, monitor='val_dice_coef', mode='max')

        model.fit(
            x=train_gen,
            epochs=100,
            callbacks=[checkpoint],
            validation_data=val_gen,
        )


if __name__ == "__main__":
    if len(sys.argv) != 4 and len(sys.argv) != 6:
        print("Usage: main.py <path/to/dataset.pickle> <preprocess:0|1> <3d:2|3> <weights> <results>")
        sys.exit(1)
    main(Path(sys.argv[1]), preprocess=int(sys.argv[2]), dimensions=int(sys.argv[3]),
         weights=sys.argv[4] if len(sys.argv) == 6 else None,
         results=sys.argv[5] if len(sys.argv) == 6 else None)
