import sys
import numpy as np
from slices_generator import SlicesGenerator
from generator import Generator
from construct_dataset import construct_dataset
from unet import get_model
from keras.callbacks import ModelCheckpoint

def main():
    # Make sure that you have X_train.npy, X_test.npy, Y_train.npy, Y_test.npy before calling the script
    gts = np.load('Y_train.npy')

    slices = np.load('X_train.npy')
    img_idx_slices = np.load('X_train_slices.npy')
    
    # Generators
    train_gen = Generator(gts, slices, img_idx_slices, batch_size=48, input_size=200, output_size=200)
    val_gen = Generator(gts, slices, img_idx_slices, batch_size=48, input_size=200, output_size=200, validation=True)
    # Models
    model = get_model((200, 200), 2)
    # model.load_weights('best_model.h5')
    checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True)
    model.fit(
        x = train_gen,
        batch_size = 1,
        epochs=100,
        callbacks=[checkpoint],
        validation_data= val_gen
    )
    
if __name__ == "__main__":
    main()