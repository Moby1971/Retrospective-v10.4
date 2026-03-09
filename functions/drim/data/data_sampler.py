"""
Code from Kai Lonnig edited by Fabian van Stijn
This data_sampler code is called zp_sc which stands for zeropadding and scaling.
This data_sampler file is used to load the data from mat format into the DRIM model
"""

import os
import numpy as np
from torch.utils.data import Dataset
import logging
from scipy.io import loadmat
import numpy.fft as fft
import time
import math
logger = logging.getLogger(__name__)


class MRData(Dataset):
    def __init__(self, data_path):
        time0 = time.time()
        self.undersampled_data_path = data_path
        # self.subjects = os.listdir(self.undersampled_data_path)
        self.subjects = ['retroAItemp.mat']
        
        logger.info('Data loading...')

        lengths = []
        self.data = dict()
        for subject in self.subjects:
            self.data[subject] = dict()

            # Load h5 file
            corresponding_file = [file for file in self.subjects if subject[15:20] in file]

            # Get undersampled data
            file2 = loadmat(self.undersampled_data_path + corresponding_file[0])
            undersampled_kspace = file2['kData']
            mask = file2['fData']

            # Mouse data misses slice dimension because there is only one slice
            if len(undersampled_kspace.shape) == 4:
                undersampled_kspace = np.expand_dims(undersampled_kspace, axis=-1)
                mask = np.expand_dims(mask, axis=-1)

            # Data is in shape (coils, time, y, x (, z/slices))
            # transpose to (coils, time, z/slices, y, x)
            undersampled_kspace = np.transpose(undersampled_kspace, [0, 1, 4, 2, 3])
            mask = np.transpose(mask, [0, 3, 1, 2])

            # Zerofill
            height, width = undersampled_kspace.shape[3], undersampled_kspace.shape[4]
            biggest_axis = max(height, width)
            power_of_2 = 2 ** math.ceil(math.log2(biggest_axis))
            height_padding, width_padding = power_of_2 - height, power_of_2 - width
            height_padding_top = height_padding // 2
            height_padding_bottom = height_padding - height_padding_top
            width_padding_left = width_padding // 2
            width_padding_right = width_padding - width_padding_left
            padding_kspace = ((0, 0), (0, 0), (0,0), (height_padding_top, height_padding_bottom),
                       (width_padding_left, width_padding_right))
            padding_mask = ((0, 0), (0, 0), (height_padding_top, height_padding_bottom),
                       (width_padding_left, width_padding_right))
            undersampled_kspace = np.pad(undersampled_kspace, pad_width=padding_kspace, mode='constant', constant_values = 0)
            mask = np.pad(mask, pad_width=padding_mask, mode='constant', constant_values=0)

            # Calculate estimate
            estimate = fft.ifftshift(np.fft.ifft2(np.fft.fftshift(undersampled_kspace, axes=(-2, -1)), axes=(-2, -1), norm='ortho'),
                                     axes=(-2, -1))
            # Normalize estimate and kspace
            factor_estimate = np.percentile(np.abs(estimate), 95)
            estimate = estimate / factor_estimate
            undersampled_kspace = fft.fftshift(np.fft.fft2(np.fft.ifftshift(estimate, axes=(-2, -1)), axes=(-2, -1), norm='ortho'),
                                  axes=(-2, -1))

            # Save data
            self.data[subject]['kspace'] = undersampled_kspace
            self.data[subject]['mask'] = mask
            self.data[subject]['estimate'] = estimate
            # Sensitivity maps
            self.data[subject]['sense'] = np.ones_like(undersampled_kspace)
            self.data[subject]

            # Take freq dim as slices
            lengths.append(undersampled_kspace.shape[-2])

        logger.info(
            f'Finished pre-processing subject(s).')

        self.lengths = np.cumsum(lengths)
        time2 = time.time()
        print("Processing time in seconds.. ", time2-time0)
        return


    def __len__(self):
        return self.lengths[-1]


    def __getitem__(self, index):
        idx = np.digitize(index, self.lengths)
        if idx > 0:
            index -= self.lengths[idx - 1]
        subject = self.subjects[idx]

        # load data
        kspace = self.data[subject]['kspace']
        estimate = self.data[subject]['estimate']
        mask = self.data[subject]['mask']
        sense = self.data[subject]['sense'] # [C, D, H, W]

        # slice over the frequency direction (H/y)
        estimate = estimate[:, :, :, index, :] # [D, dyn, W]
        mask = mask[:, :, index, :].astype(np.int8)
        sense = sense[:, :, :, index, :] # [C, D, W]
        kspace = kspace[:, :, :, index, :]

        # Check if single or multicoil
        if estimate.shape[0] == 1:
            estimate = estimate[0]

        # TO DO coil combination? If sensitivity maps...
        # else:
            # sense = sense[np.newaxis] # [C, T, D, W]
            # imspace = imspace[np.newaxis]
            # kspace = np.fft.fft(sense * imspace, axis=-1)
            # estimate = np.sum(np.fft.ifft(kspace, axis=-1) * sense.conj(), 0)

        data = {
            'subject': subject,
            'sense': np.ascontiguousarray(sense), # [C, T, D, W]
            'mask': mask[np.newaxis, ..., np.newaxis], # [1, T, D, W, 1]
            'measurements': kspace, # [C, T, D, W]
            'estimate': estimate # [T, D, W]
        }
        return data
