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
    # Ground Truth
    wmh = d / "wmh.nii.gz"

    # Data
    flair = d / "pre" / "FLAIR.nii.gz"
    t1 = d / "pre" / "T1.nii.gz"

    gts.append(str(wmh))
    slices.append((str(flair), str(t1)))  # Appending a tuple (flair_path, t1_path)


def construct_dataset(path):
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
    gt[gt > 1] = 1
    gt[gt < 0] = 0
    pre_processed = np.transpose(pre_processed, (2, 0, 1, 3))

    return pre_processed, gt, (w, h)


def compute_set(slices_paths, gts_paths):
    slices, gts, sizes = [], [], []
    for slices_path, gts_path in zip(slices_paths, gts_paths):
        result = compute_data(slices_path, gts_path)
        slices.append(result[0])
        gts.append(result[1])
        sizes.append(result[2])
    return slices, gts, sizes


def generate_data(path, save_dir):
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
