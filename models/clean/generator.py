from imgaug import SegmentationMapsOnImage
from tensorflow.keras.utils import Sequence
import numpy as np
import math


class Generator(Sequence):
    def __init__(self, X, y, preprocess=False, batch_size=10, shuffle=True, augment=None):
        """
        Create a new generator.

        Parameters:
        X (list(np.array(n_slices, 208, 208, n_filters))): The individual's data
        y (list(np.array(n_slices, 208, 208,))): The individual's segmentations
        preprocess (bool): Whether to have the preprocess
        batch_size (int): The batch size
        shuffle (bool): Whether to shuffle the dataset
        augment (augmentor): The augmentor
        """
        self.X = X
        self.y = y
        self.preprocess = preprocess
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment

        self.slices_with_white = []
        self.slices_without_white = []

        for individual, (X_individual, y_individual) in enumerate(zip(X, y)):
            for slice_index, (_, y_slice) in enumerate(zip(X_individual, y_individual)):
                if np.any(y_slice != 0):
                    self.slices_with_white.append((individual, slice_index))
                else:
                    self.slices_without_white.append((individual, slice_index))

        self.slices_with_white = np.array(self.slices_with_white)
        self.slices_without_white = np.array(self.slices_without_white)
        # TODO CHECK LENGTHS

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.slices_with_white)
            np.random.shuffle(self.slices_without_white)

    def __len__(self):
        return len(self.slices_with_white) // self.batch_size

    def get_data(self, individual, slice_index):
        raise NotImplementedError()

    def get_batch(self, dataset, start, augment):
        indices = dataset[start:start + self.batch_size]
        X = []
        y = []
        for (individual, slice_index) in indices:
            X_slice, y_slice = self.get_data(individual, slice_index)
            if self.augment and augment:
                X_slice, y_slice = self.augment(image=X_slice, segmentation_maps=SegmentationMapsOnImage(y_slice, shape=X_slice.shape))
                y_slice = y_slice.get_arr()
            X.append(X_slice)
            y.append(y_slice)
        return np.array(X), np.array(y)

    def __getitem__(self, index):
        augment = True
        if isinstance(index, tuple):
            index, augment = index
        X_with, y_with = self.get_batch(self.slices_with_white, index * self.batch_size, augment)
        X_without, y_without = self.get_batch(self.slices_without_white, index * self.batch_size, augment)

        return np.concatenate((X_with, X_without)), np.concatenate((y_with, y_without))


class Generator2D(Generator):
    @staticmethod
    def generate_data(slices, slice_index, preprocess):
        return slices[slice_index, :, :, 0:({False: 2, True: 3}[preprocess])]

    def get_data(self, individual, slice_index):
        return self.generate_data(self.X[individual], slice_index, self.preprocess), self.y[individual][slice_index]


class Generator3D(Generator):
    @staticmethod
    def generate_data(slices, slice_index, preprocess):
        X = []
        n_channels = ({False: 2, True: 3}[preprocess])
        if slice_index == 0:
            X.append(np.zeros(slices[slice_index, :, :, 0:n_channels].shape))
        else:
            X.append(slices[slice_index - 1, :, :, 0:n_channels])
        X.append(slices[slice_index, :, :, 0:n_channels])
        if slice_index == len(slices) - 1:
            X.append(np.zeros(slices[slice_index, :, :, 0:n_channels].shape))
        else:
            X.append(slices[slice_index + 1, :, :, 0:n_channels])
        return np.concatenate(X, axis=-1)

    def get_data(self, individual, slice_index):
        return self.generate_data(self.X[individual], slice_index, self.preprocess), self.y[individual][slice_index]


class KFold(Sequence):
    """
    Creates a KFold generator

    Parameters:
    generator (Generator): The generator to KFold
    folds (int): The number of folds
    validation (bool): Whether the KFold should yield validation or training sample
    """
    def __init__(self, generator, folds, validation):
        self.generator = generator
        self.folds = folds
        self.validation = validation

        self.n_batches = len(self.generator)
        self.current_fold = self.folds - 1
        self.batches_by_fold = self.n_batches // self.folds
        self.on_epoch_end()

    def on_epoch_end(self):
        self.current_fold += 1
        if self.current_fold == self.folds:
            if not self.validation:
                self.generator.on_epoch_end()
            self.current_fold = 0

    def __len__(self):
        if self.validation:
            return self.batches_by_fold
        else:
            return self.n_batches - self.batches_by_fold

    def __getitem__(self, item):
        if self.validation:
            item += self.current_fold * self.batches_by_fold
            return self.generator[item, False]
        else:
            if item >= self.current_fold * self.batches_by_fold:
                item += self.batches_by_fold
            return self.generator[item]
