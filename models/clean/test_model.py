import numpy as np
import sys
from pathlib import Path
import pickle
import tensorflow as tf
import tensorlayer as tl
import nibabel

from metrics import recall, precision, f1, dice_coef

met = {
    'recall': tf.keras.metrics.Recall(),
    'precision': tf.keras.metrics.Precision(),
    'bin_acc': tf.keras.metrics.BinaryAccuracy(),
    'dice_coef': dice_coef,
}

def calculate_metrics(Y_true, Y_pred):
    assert(len(Y_pred) == len(Y_true))
    dic = {}
    for k, v in met.items():
        batch_metric = []
        for i in range(len(Y_true)):
            y_true = np.array(Y_true[i],dtype=np.float32)
            shape = y_true.shape
            y_pred = np.array(Y_pred[i])
            y_pred = (y_true.reshape((*shape))).astype(np.float32)
            print(y_pred.max())
            batch_metric.append(v(y_true, y_pred))
        batch_metric = np.array(batch_metric)
        print(batch_metric)
        dic[k] = np.mean(batch_metric)
    return dic

def main(data_path, test_pickle_path, result_pickle_path):
    with open(test_pickle_path, 'rb') as data:
        test_data = pickle.load(data)
    with open(result_pickle_path, 'rb') as data:
        result_data = pickle.load(data)
    
    Y_true_raw = test_data['y']
    Y_pred_raw = result_data['raw']

    print('========= RAW =========')
    print(calculate_metrics(Y_true_raw, Y_pred_raw))

    # Get list of patients
    Y_true_orig = test_data['patients']

    Y_true_orig = [
        change_gt(np.expand_dims(nibabel.load(data_path / patient / 'wmh.nii.gz').get_fdata(dtype=np.float32), axis=-1)) for patient in Y_true_orig
    ]
    transformed = result_data['transformed']
    Y_pred_orig = [ data.transpose((1,2,0)).round() for data in transformed ]
    print('========= ORIG =========')
    print(calculate_metrics(Y_true_orig, Y_pred_orig))

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: test_model.py <path/to/dataset> <path/to/test_pickle> <result.pickle>")
        exit(1)
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))