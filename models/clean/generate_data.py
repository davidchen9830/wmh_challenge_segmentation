import sys
import glob
import nibabel
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
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
    flair = d / "orig" / "FLAIR.nii.gz"
    t1 = d / "orig" / "T1.nii.gz"

    gts.append(str(wmh))
    slices.append((str(flair), str(t1))) # Appending a tuple (flair_path, t1_path)

def construct_dataset(path):
    root = Path(path)
    gts, slices = [], []

    singapore = root / "Singapore"
    ultrecht = root / "Utrecht"
    ge3t = root / "GE3T"

    for d in singapore.iterdir():
        fill_dataset(gts, slices, d)
    for d in ultrecht.iterdir():
        fill_dataset(gts, slices, d)
    for d in ge3t.iterdir():
        fill_dataset(gts, slices, d)
    return np.array(gts), np.array(slices)


def generate_data(path, save_dir):
    gts, slices = construct_dataset(path)
    all_gts = []
    all_imgs = []
    all_preprocessed = []
    dic = {}
    disk = skimage.morphology.disk(2.5)
    for i in range(len(gts)):
        fl = nibabel.load(slices[i][0]).get_fdata(dtype=np.float32)
        t1 = nibabel.load(slices[i][1]).get_fdata(dtype=np.float32)
        gt = nibabel.load(gts[i]).get_fdata(dtype=np.float32)

        _, _, t1_c = t1.shape
        _, _, fl_c = fl.shape
        _, _, gt_c = gt.shape

        t1 = skimage.transform.resize(t1, (200, 200, t1_c))
        fl = skimage.transform.resize(fl, (200, 200, fl_c))
        gt = skimage.transform.resize(gt, (200, 200, gt_c))

        slices_morph = None
        for s in range(fl_c):
            flair_slice = fl[:, :, s]
            morph = skimage.morphology.dilation(flair_slice, disk)
            top_hat = skimage.util.invert(morph - flair_slice)
            top_hat = top_hat.reshape((200, 200, 1))
            if slices_morph is None:
                slices_morph = top_hat
            else:
                slices_morph = np.concatenate([slices_morph, top_hat], axis=-1)

        slices_morph = np.array(slices_morph)
        slices_morph = slices_morph.reshape((200, 200, fl_c, 1))

        t1 = t1.reshape((200, 200, t1_c, 1))
        fl = fl.reshape((200, 200, fl_c, 1))

        gt = gt.reshape((200, 200, gt_c, 1))
        img = np.concatenate([t1, fl], axis=-1)
        pre_processed = np.concatenate([img, slices_morph], axis=-1)

        dic[i] = fl_c

        # 200, 200, 1
        all_gts.append(gt)
        # 200, 200, 2
        all_imgs.append(img)
        # 200, 200, 3
        all_preprocessed.append(pre_processed)
    with open(save_dir + '/gts.pickle', 'wb') as data:
        pickle.dump(all_gts, data)
    with open(save_dir + '/imgs.pickle', 'wb') as data:
        pickle.dump(all_imgs, data)
    with open(save_dir +'/preprocessed.pickle', 'wb') as data:
        pickle.dump(all_preprocessed, data)

    return dic


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: ./bin data_path save_dir')
        exit(1)
    dic = generate_data(sys.argv[1], sys.argv[2])
    indices = []
    for k, v in dic.items():
        for i in range(v):
            indices.append([k, i])
    indices = np.array(indices)
    # [[0, 1]...[0, 48]...]
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, test_indices = train_test_split(indices,
                                                   test_size=0.33,
                                                   random_state=42)
    np.save(sys.argv[2] + '/x_train_slices.npy', train_indices)
    np.save(sys.argv[2] + '/x_test_slices.npy', test_indices)
    print('Finished !')
