import sys
from generator import Generator
from construct_dataset import construct_dataset
from unet import get_model
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint

def main(path):
    # Path leads to the where the data is Ultrecht, Singapore etc...

    # Dataset
    gts, slices = construct_dataset(path)
    gts_train, gts_val, slices_train, slices_val = train_test_split(gts, slices, test_size=0.2, random_state=42)
    # Generators
    train_gen = Generator(gts_train, slices_train, batch_size=1, shuffle=True)
    val_gen = Generator(gts_val, slices_val, batch_size=1, shuffle=True)
    # Models
    model = get_model((572, 572), 3)
    checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True)
    model.fit(
        x = train_gen,
        batch_size = 1,
        epochs=100,
        callbacks=checkpoint,
        validation_data= val_gen
    )
    
if __name__ == "__main__":
    main(sys.argv[1])