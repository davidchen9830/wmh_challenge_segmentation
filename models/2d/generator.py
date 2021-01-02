import numpy as np
import nibabel as nib
from utils import pad
from tensorflow.keras.utils import Sequence

from skimage.transform import resize

class Generator(Sequence):
    def __init__(self, gts_paths, img_paths, img_idx_slices, batch_size=1,
                input_size=572, output_size=388, K_fold = 5, validation=False):
        assert(len(gts_paths) == len(img_paths))
        self.size = len(img_idx_slices)
        self.img_idx_slices = img_idx_slices

        self.gts = gts_paths
        self.imgs = img_paths
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        self.K = K_fold
        self.validation = validation
        
        self.last_validation_idx = 0
        self.on_epoch_end()

    def on_epoch_end(self):
        # We want to build a K_fold cross validation generator
        # At each epoch, we will change training and validation data

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
        # The data are already shuffled, no use to shuffle it multiple times
    
    def __len__(self):
        return int(np.ceil(len(self.indices) / float(self.batch_size)))

    # Let's use the same function for 2d and 3d construction of slices
    # The argument to specify is dimension, by default 2d slice
    # Even though 
    def __construct_slices(self, gts, slices, dimension=2):
        X = []
        Y = []
        # Typically this is like [FLAIR, T1, preprocess?]
        # With shapes like       [(h, w, c), ....]
        nb_channels = len(slices[0]) # Get the number of components
        sz = len(gts)
        for i in range(sz):
            gt_h, gt_w, c1 = gts[i].shape
            for channel in range(c1):
                # First slice take the first component
                h1, w1, c1 = slices[i][0].shape
                new_slice = (np.array((slices[i][0])[:, :, channel])).reshape(h1, w1, 1)
                # This lead to new_slice to look like this as a volume
                """ flair, t1, preprocess ?
                    flair, t1, preprocess ?
                    flair, t1, preprocess ?
                """
                for component in range(1, nb_channels):
                    curr_slice = ((slices[i][component])[:, :, channel]).reshape(h1, w1, 1)
                    new_slice = np.concatenate((new_slice, curr_slice), axis=-1)
                # Check stacking
                X.append(new_slice)
                gt = ((gts[i])[:, :, channel]).reshape(gt_h, gt_w, 1)
                Y.append(gt)
        return np.array(X), np.array(Y)

    def __load_data(self, path, labels=True):
        # After data exploration, labels are on float32 and inputs on uint16, let's make it then
        img = (nib.load(path)).get_fdata(dtype=np.float32)
        _, _, channels = img.shape
        img = resize(img, (self.output_size, self.output_size, channels), preserve_range=True, order=1)
        # If it is ground truth, we need the data to be (388, 388, channels) nothing more
        if labels:
            return img
        # If it is input data
        img = np.expand_dims(img, axis=0) # This will lead to (1, 388, 388, c)
        img = np.swapaxes(img, 0, 3) # Lead to (c, 388, 388, 1)
        if (self.input_size != self.output_size):
            img = pad(img, self.input_size, self.output_size)
        return img

    def __get_data(self, indices):
        X_res = []
        Y_res = []
        dic = {}
        for img_idx, img_slice in indices:
            if img_idx in dic:
                dic[img_idx].append(img_slice)
            else:
                dic[img_idx] = [img_slice]
        # This will reduce the number of times a single image is loaded
        for img_idx, slices in dic.items():
            gt = self.gts[img_idx]
            img = self.imgs[img_idx]
            Y = np.array(self.__load_data(gt, labels=True))
            X = np.array([
                self.__load_data(img[0], labels=False),
                self.__load_data(img[1], labels=False)])
            # X is now 2, c, 572, 572, 1
            X = np.swapaxes(X, 1, 4)
            X = np.swapaxes(X, 0, 1)
            # Making it 1, 2, 572, 572, c for the function
            X_slices, Y_slices = self.__construct_slices([Y], X, dimension=2)
            for s in slices:
                X_slice = X_slices[s]
                Y_slice = Y_slices[s]
                X_res.append(X_slice)
                Y_res.append(Y_slice)
        return np.array(X_res), np.array(Y_res)

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        # Creates tuple of (image index, image slice) => 5th slice of 1st image for instance
        if self.validation:
            indices = self.validation_samples[indices]
        else:
            indices = self.training_samples[indices]

        X, Y = self.__get_data(indices)
        print(X.shape, Y.shape)
        return X, Y