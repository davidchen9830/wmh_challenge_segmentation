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

expected_shape = (200, 200, 48)
disk = skimage.morphology.disk(2.5)

def generate_data(save_dir):
    for patient in patients:
        print(patient)
        t1 = nibabel.load(path / patient / 'pre' / 'T1.nii.gz').get_fdata()
        fl = nibabel.load(path / patient / 'pre' / 'FLAIR.nii.gz').get_fdata()
        gt = nibabel.load(path / patient / 'wmh.nii.gz').get_fdata() != 0

        t1 = skimage.transform.resize(t1, expected_shape, preserve_range=True, order=1)
        fl = skimage.transform.resize(fl, expected_shape, preserve_range=True, order=1)
        gt = skimage.transform.resize(gt, expected_shape, preserve_range=True, order=1)

        slices_morph = None
        for s in range(48):
            flair_slice = fl[:, :, s]
            
    
            morph = skimage.morphology.dilation(flair_slice, disk)
            top_hat = skimage.util.invert(morph - flair_slice)
            top_hat = top_hat.reshape((200, 200, 1))
            if slices_morph is None:
                slices_morph = top_hat
            else:
                slices_morph = np.concatenate([slices_morph, top_hat], axis=-1)

        slices_morph = np.array(slices_morph)
        
        slices_morph = slices_morph.reshape((200, 200, 48, 1))

        t1 = t1.reshape((200, 200, 48, 1))
        fl = fl.reshape((200, 200, 48, 1))
        gt = gt.reshape((200, 200, 48, 1))
        
        img = np.concatenate([t1, fl], axis=-1)
        pre_processed = np.concatenate([img, slices_morph], axis=-1)
        
        n0 = patient.split('/')[1]
        
        np.save(save_dir + '/data/' + str(n0) + '.npy', img)
        np.save(save_dir + '/data/' + str(n0) + '_preprocessed' + '.npy', pre_processed)
        np.save(save_dir + '/labels/' + str(n0) + '.npy', gt)

if __name__ == "__main__":
    generate_data(sys.argv[1])
