import numpy as np
from tensorflow.keras.utils import Sequence
import random

# This is a generator that load everything in RAM
class Generator(Sequence):
    def __init__(self, data_path, gts, imgs,
                batch_size=1,
                input_size=200, output_size=200, K_fold = 5, 
                validation = False, preprocess=False):

        # [Individu_0, individu_1....]
        # [n, 256, 256, 48, 2]
        # [N, 256, 256, 2]
        self.preprocess = preprocess
        self.data_path = data_path
        self.gts = gts
        self.imgs = imgs
        self.size = len(gts)

        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        self.K = K_fold

        self.validation = validation
        self.last_validation_idx = 0
        self.on_epoch_end()

    def on_epoch_end(self):
        validation_size = self.size // self.K
        training_size = self.size - validation_size

        validation_samples = []
        training_samples = []

        for i in range(self.last_validation_idx, self.last_validation_idx + validation_size):
            validation_samples.append(i % self.size)

        self.last_validation_idx = (self.last_validation_idx + validation_size) % self.size

        for i in range(self.last_validation_idx, self.last_validation_idx + training_size):
            training_samples.append(i % self.size)
        
        self.validation_samples = np.array(validation_samples)
        self.training_samples = np.array(training_samples)

        self.indices = np.arange(len(validation_samples)) if self.validation else np.arange(len(training_samples))

    def __len__(self):
        return int(np.ceil(len(self.indices) / float(self.batch_size)))

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        if self.validation:
            indices = self.validation_samples[indices]
        else:
            indices = self.training_samples[indices]
        X, Y = self.__getdata(indices)
        return X, Y

    def __getdata(self, indices):
        # We need X, Y
        # This function should return
        ids = self.imgs[indices]
        if not self.preprocess:
            imgs = np.array([
                np.load(self.data_path + '/data/' + str(idx) + '.npy') for idx in ids
            ])
        else:
            imgs = np.array([
                np.load(self.data_path + '/data/' + str(idx) + '_preprocessed' + '.npy') for idx in ids
            ])

        gts = np.array([
            np.load(self.data_path + '/labels/' + str(idx) + '.npy') for idx in ids
        ])
        return imgs, gts
