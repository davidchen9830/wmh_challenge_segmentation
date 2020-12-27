import numpy as np
from utils import get_slices, pad
from tensorflow.keras.utils import Sequence

class SlicesGenerator(Sequence):
    def __init__(self, images, segmentations, batch_size=2, shuffle=True, input_size=572, output_size=388):
        assert(len(images) == len(segmentations))
        self.images = images
        self.segmentations = segmentations
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.input_size = input_size
        self.output_size = output_size
        
        self.indices = np.arange(len(self.images))
        
        self.on_epoch_end()
    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indices)
            
    def __len__(self):
        return len(self.indices) // self.batch_size
    
    def __get_data(self, batch):
        X = get_slices(pad(self.images[batch], self.input_size, self.output_size), self.input_size, self.output_size)
        y = get_slices(pad(self.segmentations[batch], self.output_size, self.output_size), self.output_size, self.output_size)
        return X.reshape(-1, *X.shape[-3:]), y.reshape(-1, *y.shape[-3:])
    
    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:index * self.batch_size + self.batch_size]
        X, y = self.__get_data(indices)
        return X, y