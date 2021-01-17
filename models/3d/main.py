import sys
import numpy as np
from generator import Generator
from construct_dataset import construct_dataset
from unet import get_model
from keras.callbacks import ModelCheckpoint

def main(data_path, preprocess=False):
    # Make sure that you have X_train.npy, X_test.npy, Y_train.npy, Y_test.npy before calling the script
    gts = np.load('Y_train.npy')

    slices = np.load('X_train.npy')
    
    # Generators
    train_gen = Generator(data_path, gts, slices, batch_size=1, input_size=200, output_size=200, preprocess=preprocess)
    val_gen = Generator(data_path, gts, slices, batch_size=1, input_size=200, output_size=200, validation=True, preprocess=preprocess)
    # Models
    model = get_model((200, 200, 48), 2 if not preprocess else 3)
    # model.load_weights('best_model.h5')
    checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True)
    if preprocess:
        checkpoint = ModelCheckpoint("best_model_preprocess.h5", save_best_only=True)

    model.fit(
        x = train_gen,
        batch_size = 1,
        epochs=20,
        callbacks=[checkpoint],
        validation_data= val_gen
    )
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please, provide a data path")
        exit(1)
    preprocess = False
    if len(sys.argv) > 2:
        preprocess = True
    main(sys.argv[1], preprocess=preprocess)