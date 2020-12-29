import numpy as np
import nibabel as nib
from tensorflow.keras.utils import Sequence

from skimage.transform import resize

class Generator(Sequence):
    def __init__(self, gts_paths, img_paths, batch_size=1, shuffle=True,
                input_size=572, output_size=388, K_fold = 5, validation=False, slices_per_img=48):
        assert(len(gts_paths) == len(img_paths))
        self.size = len(gts_paths)
        
        self.gts = gts_paths
        self.imgs = img_paths
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        self.K = K_fold
        self.validation = validation
        self.slices_per_img = slices_per_img
        
        self.last_validation_idx = 0
        self.on_epoch_end()

    def on_epoch_end(self):
        # We want to build a K_fold cross validation generator
        # At each epoch, we will change training and validation data

        validation_size = self.size // self.K
        training_size = self.size - validation_size
        validation_gts = []
        validation_imgs = []

        training_gts = []
        training_imgs = []
        for i in range(self.last_validation_idx, self.last_validation_idx + validation_size):
            validation_gts.append(self.gts[i % self.size])
            validation_imgs.append(self.imgs[i % self.size])

        self.last_validation_idx = (self.last_validation_idx + validation_size) % self.size

        for i in range(self.last_validation_idx, self.last_validation_idx + training_size):
            training_gts.append(self.gts[i % self.size])
            training_imgs.append(self.imgs[i % self.size])

        self.training_gts = np.array(training_gts)
        self.training_imgs = np.array(training_imgs)

        self.validation_gts = np.array(validation_gts)
        self.validation_imgs = np.array(validation_imgs)

        if self.validation:
            self.indices = np.arange(len(self.validation_gts) * self.slices_per_img)
        else:
            self.indices = np.arange(len(self.training_gts) * self.slices_per_img)

        if self.shuffle == True:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return len(self.indices) // self.batch_size

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
                # Models asks for (w, h, 3)
                # So if we have a missing channel, add a dummy one
                if dimension == 2:
                    new_slice = np.concatenate((new_slice, np.zeros((h1, w1, 1))), axis=-1)
                X.append(new_slice)
                gt = ((gts[i])[:, :, channel]).reshape(gt_h, gt_w, 1)
                Y.append(gt)
        return np.array(X), np.array(Y)


    def __get_data(self, indices):
        X_res = []
        Y_res = []
        for idx in indices:
            img_idx, img_slice = idx
            if self.validation:
                gt = self.validation_gts[img_idx]
                slice = self.validation_imgs[img_idx]
            else:
                gt = self.training_gts[img_idx]
                slice = self.training_imgs[img_idx]
            Y = np.array(resize(nib.load(gt).get_fdata(), (388, 388, 48), preserve_range=True, order=1))
            X = np.array((
                np.pad(resize(nib.load(slice[0]).get_fdata(), (388, 388, 48), preserve_range=True, order=1), ((92,92), (92,92), (0,0))),
                np.pad(resize(nib.load(slice[1]).get_fdata(), (388, 388, 48), preserve_range=True, order=1), ((92,92), (92,92), (0,0)))))
            X, Y = self.__construct_slices([Y], [X], dimension=2)
            X = X[img_slice]
            Y = Y[img_slice]
            X_res.append(X)
            Y_res.append(Y)
        return np.array(X_res), np.array(Y_res)
            
        """    
        gts = self.gts[indices]
        imgs = self.imgs[indices]

        # The chain is nibabel image => Resize into 388, 388 (unet output) => Pad to 572, 572 (unet input)
        # We pad values with 0, but we could do something else aswell
        Y = np.array([resize(nib.load(gt).get_fdata(), (388, 388, 48), preserve_range=True, order=1) for gt in gts])
        X = np.array([(
            np.pad(resize(nib.load(slice[0]).get_fdata(), (388, 388, 48), preserve_range=True, order=1), ((92,92), (92,92), (0,0))),
            np.pad(resize(nib.load(slice[1]).get_fdata(), (388, 388, 48), preserve_range=True, order=1), ((92,92), (92,92), (0,0)))) for slice in imgs
            ])
        X_slices, Y_slices = self.__construct_slices(Y, X, dimension=2)
        # Should we shuffle ? Probably I guess
        indices = np.arange(len(X_slices))
        np.random.shuffle(indices)

        return X_slices[indices], Y_slices[indices]
        """

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        # Creates tuple of (image index, image slice) => 5th slice of 1st image for instance
        indices = [(indice // 48, indice % 48) for indice in indices]
        X, Y = self.__get_data(indices)
        return X, Y