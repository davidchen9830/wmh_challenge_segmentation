import numpy as np
import nibabel as nib
from tensorflow.keras.utils import Sequence

from skimage.transform import resize

class Generator(Sequence):
    def __init__(self, gts_paths, img_paths, batch_size=1, shuffle=True, input_size=572, output_size=388):
        assert(len(gts_paths) == len(img_paths))
        self.gts = gts_paths
        self.imgs = img_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.input_size = input_size
        self.output_size = output_size
        self.indices = np.arange(len(self.imgs))
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
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
            h1, w1, c1 = gts[i].shape
            for channel in range(c1):
                # First slice take the first component
                assert(slices[i][0].shape == (h1, w1, c1))
                new_slice = (np.array((slices[i][0])[:, :, channel])).reshape(h1, w1, 1)
                # This lead to new_slice to look like this as a volume
                """ flair, t1, preprocess ?
                    flair, t1, preprocess ?
                    flair, t1, preprocess ?
                """
                for component in range(1, nb_channels):
                    curr_component = slices[i][component]
                    # Mb remove in prod ?
                    assert(curr_component.shape == (h1, w1, c1))
                    curr_slice = ((slices[i][component])[:, :, channel]).reshape(h1, w1, 1)
                    new_slice = np.concatenate((new_slice, curr_slice), axis=-1)
                # Check stacking
                assert(new_slice.shape == (h1, w1, nb_channels))
                # Models asks for (w, h, 3)
                # So if we have a missing channel, add a dummy one
                if dimension == 2:
                    new_slice = np.concatenate((new_slice, np.zeros((h1, w1, 1))), axis=-1)
                X.append(new_slice)
                gt = ((gts[i])[:, :, channel]).reshape(h1, w1, 1)
                Y.append(gt)
        return np.array(X), np.array(Y)


    def __get_data(self, indices):
        gts = self.gts[indices]
        imgs = self.imgs[indices]

        # The chain is nibabel image => Resize into 388, 388 (unet output) => Pad to 572, 572 (unet input)
        # We pad values with 0, but we could do something else aswell
        Y = np.array([np.pad(resize(nib.load(gt).get_fdata(), (388, 388, 48), preserve_range=True, order=1), ((92, 92), (92, 92), (0, 0))) for gt in gts])
        X = np.array([(
            np.pad(resize(nib.load(slice[0]).get_fdata(), (388, 388, 48), preserve_range=True, order=1), ((92,92), (92,92), (0,0))),
            np.pad(resize(nib.load(slice[1]).get_fdata(), (388, 388, 48), preserve_range=True, order=1), ((92,92), (92,92), (0,0)))) for slice in imgs
            ])
        X_slices, Y_slices = self.__construct_slices(Y, X, dimension=2)
        # Should we shuffle ? Probably I guess
        indices = np.arange(len(X_slices))
        np.random.shuffle(indices)

        return X_slices[indices], Y_slices[indices]

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        X, Y = self.__get_data(indices)
        return X, Y