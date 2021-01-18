import numpy as np
from tensorflow.keras.utils import Sequence
import random

# This is a generator that load everything in RAM
class Generator(Sequence):
    def __init__(self, data_path, img_idx_slices,
                batch_size=16,
                input_size=200, output_size=200, K_fold = 5, 
                validation = False, preprocess=False):
        self.preprocess = preprocess
        self.data_path = data_path
        self.img_idx_slices = img_idx_slices
        self.size = len(img_idx_slices)

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
            validation_samples.append(self.img_idx_slices[i % self.size])

        self.last_validation_idx = (self.last_validation_idx + validation_size) % self.size

        for i in range(self.last_validation_idx, self.last_validation_idx + training_size):
            training_samples.append(self.img_idx_slices[i % self.size])
        
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
        # indices looks like [[img idx, img_sample]...]
        gts = []
        imgs = []
        for idx, img_sample in indices:
            gts.append(np.load(self.data_path + '/labels/' + str(idx) + '_' + str(img_sample) + '.npy'))
            if not self.preprocess:
                imgs.append(np.load(
                    self.data_path + '/data/' + str(idx) + '_' + str(img_sample) + '.npy'))
            else:
                imgs.append(np.load(
                    self.data_path + '/data/' + str(idx) + '_' + str(img_sample) + '_preprocessed.npy'))
        return np.array(imgs), np.array(gts)
