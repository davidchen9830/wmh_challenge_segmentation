import sys
import nibabel
from pathlib import Path
import skimage.transform
import numpy as np
import skimage
import skimage.io
import skimage.morphology
from sklearn.model_selection import train_test_split
import pickle


def fill_dataset(gts, slices, d):
    """
    Fill the dataset with path to images

    Parameters:
    gts (list(str)): List containing all the gts path
    slices (list(str, str)): List containing all the slices path (FLAIR, T1)
    d (Path): Path of the dataset for the current patient
    """
    
    # Ground Truth
    wmh = d / "wmh.nii.gz"

    # Data
    flair = d / "pre" / "FLAIR.nii.gz"
    t1 = d / "pre" / "T1.nii.gz"

    gts.append(str(wmh))
    slices.append((str(flair), str(t1)))  # Appending a tuple (flair_path, t1_path)


def construct_dataset(path):
    """
    Construct the train/test split with 80/20 ratio
    with the given path where the dataset is located

    Parameters:
    path (Path): Path of the dataset root folder

    Returns:
    (X, Y) with X and Y being 3 elements (Patient slices, GT slices, Patient Name)
    """
    root = Path(path)
    train_slices, train_gts, train_patients = [], [], []
    test_slices, test_gts, test_patients = [], [], []


    for subset in ["Singapore", "Utrecht", "GE3T"]:
        dataset = root / subset
        slices, gts, patients = [], [], []
        for d in sorted(list(dataset.iterdir())):
            patients.append(f"{d.parent.name}/{d.name}")
            fill_dataset(gts, slices, d)
        split = train_test_split(slices, gts, patients, test_size=0.20, random_state=42)
        train_slices += split[0]
        test_slices += split[1]
        train_gts += split[2]
        test_gts += split[3]
        train_patients += split[4]
        test_patients += split[5]

    return (train_slices, train_gts, train_patients), (test_slices, test_gts, test_patients)


def square(image):
    """
    square the current image along the largest dimension

    Parameters:
    image: ndarray (w, h, ...)

    Returns:
    image: ndarray squared ndarray (w,w, ...) or (h,h, ...)
    """
    w, h, c = image.shape

    diff = abs(w - h)
    pad_before = diff // 2
    pad_after = diff - pad_before

    if w < h:
        return np.pad(image, ((pad_before, pad_after), (0, 0), (0, 0))), (w, h)
    elif h < w:
        return np.pad(image, ((0, 0), (pad_before, pad_after), (0, 0))), (w, h)
    else:
        return image, (w, h)


def compute_data(slices_path, gts_path):
    """
    Main function computing the all the necessary data for training
    Get the data -> Square it -> Center and Reduce -> Resize -> Add morphology slice

    Parameters:
    slices_path list(str, str): List of image path (FLAIR, T1)
    gts_path list(str): List of ground truth

    Returns:
    pre_processed ndarray(w, h, 3): Data for training, containing in the channels FLAIR, T1, Preprocess
    gt            ndarray(w, h, 1): Gt
    (w, h)        shape of the image along the 0 and 1 dimension

    """
    print(f'Computing for {gts_path}.')
    disk = skimage.morphology.disk(2)
    fl, _ = square(nibabel.load(slices_path[0]).get_fdata(dtype=np.float32))
    t1, _ = square(nibabel.load(slices_path[1]).get_fdata(dtype=np.float32))
    gt, (w, h) = square(nibabel.load(gts_path).get_fdata(dtype=np.float32))

    fl = (fl - fl.mean()) / fl.std()
    t1 = (t1 - t1.mean()) / t1.std()
    # Don't center and reduce gt

    _, _, t1_c = t1.shape
    _, _, fl_c = fl.shape
    _, _, gt_c = gt.shape

    t1 = skimage.transform.resize(t1, (208, 208, t1_c))
    fl = skimage.transform.resize(fl, (208, 208, fl_c))
    gt = skimage.transform.resize(gt, (208, 208, gt_c))

    slices_morph = None
    for s in range(fl_c):
        flair_slice = fl[:, :, s]
        morph = skimage.morphology.dilation(flair_slice, disk)
        top_hat = morph - flair_slice  # skimage.util.invert(morph - flair_slice)
        top_hat = top_hat.reshape((208, 208, 1))
        if slices_morph is None:
            slices_morph = top_hat
        else:
            slices_morph = np.concatenate([slices_morph, top_hat], axis=-1)

    slices_morph = np.array(slices_morph)
    slices_morph = slices_morph.reshape((208, 208, fl_c, 1))

    t1 = t1.reshape((208, 208, t1_c, 1))
    fl = fl.reshape((208, 208, fl_c, 1))

    gt = gt.reshape((208, 208, gt_c, 1))
    pre_processed = np.concatenate([t1, fl, slices_morph], axis=-1)

    # (h, w, nb_slices, channels)
    # (nb_slices, h, w, channels)
    gt = np.transpose(gt, (2, 0, 1, 3))
    gt = gt.round()
    gt[gt > 1] = 0
    gt[gt < 0] = 0
    gt = gt.astype(np.bool)
    pre_processed = np.transpose(pre_processed, (2, 0, 1, 3))

    return pre_processed, gt, (w, h)


def compute_set(slices_paths, gts_paths):
    """
    Compute the set for the given slices_path and gts_paths

    Parameters:
    slices_paths (list(str)): List containing all the gts path
    gts_paths    (list(str, str)): List containing all the slices path (FLAIR, T1)

    Returns:
    slices: Slices for each Patient
    gts   : GT for each Patient
    sizes : Resized image for each image
    """
    slices, gts, sizes = [], [], []
    for slices_path, gts_path in zip(slices_paths, gts_paths):
        result = compute_data(slices_path, gts_path)
        slices.append(result[0])
        gts.append(result[1])
        sizes.append(result[2])
    return slices, gts, sizes


def generate_data(path, save_dir):
    """
    Main function, computing the data for training
    It creates two files: save_dir/train.pickle and save_dir/test.pickle

    Parameters:
    path: Root folder of the dataset
    save_dir: Root folder of the save directory
    """
    (train_slices_paths, train_gts_paths, train_patients), (test_slices_paths, test_gts_paths, test_patients) = construct_dataset(path)
    train_slices, train_gts, train_sizes = compute_set(train_slices_paths, train_gts_paths)
    test_slices, test_gts, test_sizes = compute_set(test_slices_paths, test_gts_paths)

    with (save_dir / 'train.pickle').open('wb') as data:
        pickle.dump({
            'X': train_slices,
            'y': train_gts,
            'patients': train_patients,
            'sizes': train_sizes,
        }, data)

    with (save_dir / 'test.pickle').open('wb') as data:
        pickle.dump({
            'X': test_slices,
            'y': test_gts,
            'patients': test_patients,
            'sizes': test_sizes,
        }, data)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: ./bin data_path save_dir')
        exit(1)
    generate_data(Path(sys.argv[1]), Path(sys.argv[2]))
    print('Finished !')
