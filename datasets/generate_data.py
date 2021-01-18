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

path = Path('.')
patients = glob.glob(str(path / '*' / '*'))

disk = skimage.morphology.disk(2.5)

def generate_samples(save_dir, data, n0, sample_size=16, gt=False, preprocess=False):
    _, _, vol, _ = data.shape
    if vol % sample_size != 0:
        print('Error, data and sample size not divisble')
        return
    saveN0 = n0
    n0 = str(n0)
    nb_samples = vol // sample_size
    imgs = []
    for sample_idx in range(nb_samples):
        curr_sample = data[:, :, sample_idx * sample_size: (sample_idx + 1) * sample_size, :]
        if gt:
            # This lead to /labels/50_0.npy
            np.save(save_dir + '/labels/' + n0 + '_' + str(sample_idx) + '.npy', curr_sample)
        else:
            if not preprocess:
                np.save(save_dir + '/data/' + n0 + '_' + str(sample_idx) + '.npy', curr_sample)
            else:
                np.save(save_dir + '/data/' + n0 + '_' + str(sample_idx) + '_preprocessed.npy', curr_sample)
        imgs.append(sample_idx)
    return np.array(imgs)

def generate_data(save_dir):
    dic = {}
    for patient in patients:
        print(patient)
        t1 = nibabel.load(path / patient / 'pre' / 'T1.nii.gz').get_fdata()
        fl = nibabel.load(path / patient / 'pre' / 'FLAIR.nii.gz').get_fdata()
        gt = nibabel.load(path / patient / 'wmh.nii.gz').get_fdata() != 0
        
        _, _, t1_c = t1.shape
        _, _, fl_c = fl.shape
        _, _, gt_c = gt.shape
        
        if t1_c == 83:
            t1 = t1[:, :, 2:82]
            t1_c = 80
        if fl_c == 83:
            fl = fl[:, :, 2:82]
            fl_c = 80
        if gt_c == 83:
            gt = gt[:, :, 2:82]
            gt_c = 80

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
        
        n0 = patient.split('/')[1]
        
        dic[int(n0)] = (generate_samples(save_dir, img, n0, sample_size=16, gt=False, preprocess=False))
        generate_samples(save_dir, gt, n0, sample_size=16, gt=True, preprocess=False)
        generate_samples(save_dir, pre_processed, n0, sample_size=16, gt=False, preprocess=True)
    return dic

if __name__ == "__main__":
    ids = generate_data(sys.argv[1])
