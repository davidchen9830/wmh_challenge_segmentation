# wmh_segmentation_challenge

## Documentation

- https://wmh.isi.uu.nl/data/

## Implementation

U-NET model for white matter segmentation

## Results

- https://drive.google.com/file/d/11OSFLJ2qMyQj5LbVa2l9QQTvjYIcLfbn/view?usp=sharing

## Dataset

The dataset is the one used during the WMH Segmentation Challenge
In our repository, it should be placed under a datasets/ folder

## Usage

cd models/clean

### Training

python3 main.py <path/to/dataset.pickle> <preprocess:0|1> <3d:2|3>

Please look at the file main.py for more information regarding the parameters

Examples of training usage:

- python3 main.py data/train.pickle 0 2
- python3 main.py data/train.pickle 1 2
- python3 main.py data/train.pickle 0 3
- python3 main.py data/train.pickle 1 3

### Testing

python3 main.py <path/to/dataset.pickle> <preprocess:0|1> <3d:2|3> <weights> <results>

<path/to/dataset.pickle> should be something like */test.pickle

Please look at the file main.py for more information regarding the parameters
After generating the weights and the result file, you can test the model.

python3 test_model.py <path/to/dataset> <path/to/test_pickle> <result.pickle>

With:
- <path/to/dataset> The root folder of the dataset
- <path/to/test_pickle> The same argument given to <path/to/dataset.pickle>
- <result.pickle> The file generated by the command above with "python3 main.py ...."