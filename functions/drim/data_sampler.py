"""
This data_sampler code is called zp_sc which stands for zeropadding and scaling.
This data_sampler file is used to load the data from mat format into the DRIM model
"""

# =====================================================================================
# DATA LOADING FOR DRIM RECONSTRUCTION
# =====================================================================================
#
# This module turns the .mat file written by the MATLAB Retrospective app into the tensors
# that the DRIM network consumes. All of the work happens once, in MRData.__init__, which
# reads the file and pre-processes the entire volume into memory. __getitem__ afterwards
# only slices that in-memory array, which is why it is very cheap.
#
# The pre-processing chain in __init__ is:
#
#   1. Load 'kData' (undersampled k-space) and 'fData' (the sampling mask) from the .mat.
#   2. Add a slice axis if the data is single slice (mouse data), so the rest of the code
#      can assume a uniform number of dimensions.
#   3. Transpose into the axis order the network expects.
#   4. Zero fill k-space to a square 256 x 256 matrix (see zero_fill below).
#   5. Inverse Fourier transform to obtain a first, aliased image estimate.
#   6. Normalise that estimate to a maximum magnitude of 1, and transform the normalised
#      estimate back to k-space so that image and k-space stay mutually consistent.
#
# Axis naming used throughout this file and the network:
#   C = coils, T (or dyn) = cardiac/respiratory time frames, D = depth/slices,
#   H = phase encoding direction (y), W = readout direction (x)
#
# One important consequence of step 6: the "measurements" the network is later asked to be
# consistent with are the *normalised* k-space, not the raw scanner data. The scaling
# factor is not stored, so the reconstruction comes out in normalised units.
# =====================================================================================

import os
import numpy as np
from torch.utils.data import Dataset
import logging
from scipy.io import loadmat
import numpy.fft as fft
import time
import torch
logger = logging.getLogger(__name__)


def share(tensor):
    """Move a tensor's storage into shared memory so dataloader workers can map it."""
    # Dataloader workers started with "spawn" (the default on macOS and Windows) receive
    # the dataset by pickling it. A tensor backed by shared memory pickles as a handle to
    # the existing storage instead of as a copy of its contents, so N workers cost one
    # copy of the data rather than N. On Linux, where workers are forked, the pages would
    # already have been shared copy-on-write, and this is simply harmless.
    tensor.share_memory_()
    return tensor


def image_from_kspace(kspace):
    """Inverse Fourier transform over the last two axes, k-space centred convention."""
    # The starting image for the iterative network: a plain inverse FFT of the
    # undersampled k-space, so it still contains all the aliasing artefacts that the
    # network is trained to remove. The fftshift/ifftshift pair moves the zero frequency
    # between the corner (which is where the FFT expects it) and the centre (which is
    # where MRI convention puts it). norm='ortho' makes the forward and inverse transforms
    # exact mirror images of one another.
    #
    # Only the last two axes are transformed, so every slice is independent of every
    # other. That is what allows __getitem__ to reconstruct one slice at a time instead of
    # keeping an image estimate of the whole volume in memory.
    return fft.ifftshift(np.fft.ifft2(np.fft.fftshift(kspace, axes=(-2, -1)), axes=(-2, -1), norm='ortho'),
                         axes=(-2, -1))


def centre_slices(n, length):
    """Source and destination slices that centre an axis of size n in an axis of size length."""
    # Returns the pair (source, destination) of slice objects needed to place an axis of
    # size n into an output axis of size `length`, keeping the centre of the data at the
    # centre of the output. Exactly one of the two directions is ever active:
    #
    #   n <  length  ->  pad:  take everything, write it into the middle of the output
    #   n >  length  ->  crop: take the middle of the input, write it across the output
    #   n == length  ->  both slices span the whole axis, i.e. a straight copy
    #
    # Since k-space here has its zero frequency in the centre, cropping the outer edges
    # discards the highest spatial frequencies. That lowers the spatial resolution while
    # leaving the field of view unchanged; it does not shift or distort the image.
    if n >= length:
        # Crop: start far enough in that the same amount is removed from either side.
        start = (n - length) // 2
        return slice(start, start + length), slice(0, length)
    # Pad: leave an equal margin of zeros on either side.
    start = (length - n) // 2
    return slice(0, n), slice(start, start + n)


def zero_fill(kspace, mask, length=256):
    """Force k-space onto a square `length` x `length` matrix, cropping or zero padding each axis."""
    # Brings every scan onto the fixed matrix size the trained network expects, whatever
    # size it was acquired at. The two axes are handled independently, so a 300 x 200
    # acquisition is cropped along one axis and padded along the other.
    #
    # Zero padding in k-space is equivalent to sinc interpolation in image space: it adds
    # no real information, it only resamples onto a finer grid. Cropping is the opposite
    # and is genuinely lossy, since the discarded outer k-space carries the fine detail.
    #
    # NOTE the axis names here are the local ones: the last two axes are called x and y,
    # but they correspond to the (H, W) pair described in the module header.
    coils, time, slices, x, y = kspace.shape

    # Work out, per axis and independently, whether to crop or to pad.
    src_x, dst_x = centre_slices(x, length)
    src_y, dst_y = centre_slices(y, length)

    # Cropping throws data away, so say so rather than doing it silently: the operator
    # should know the reconstruction is running at a reduced resolution.
    if x > length or y > length:
        print(f"Note: k-space {x}x{y} is larger than {length}x{length}, "
              f"cropping to {length}x{length} (spatial resolution is reduced)..")

    # Create new kspace with the new dimensions
    # The output is allocated at exactly the target size, so it is impossible for a
    # mis-computed offset to produce anything other than length x length.
    # dtype=complex is numpy's complex128. It is cast down to complex64 in __getitem__,
    # since MPS (Apple Silicon) cannot handle 64 bit floats at all.
    new_kspace = np.zeros((coils, time, slices, length, length), dtype=complex)
    # Copy the (possibly cropped) data into the (possibly padded) destination window.
    new_kspace[:, :, :, dst_x, dst_y] = kspace[:, :, :, src_x, src_y]
    # Create new kspace with the new dimensions
    # The mask gets exactly the same treatment, so that any zero filled region stays
    # marked as "not measured" and the network's data consistency step ignores it.
    new_mask = np.zeros((time, slices, length, length))
    new_mask[:, :, dst_x, dst_y] = mask[:, :, src_x, src_y]
    return new_kspace, new_mask


class MRData(Dataset):
    # A torch Dataset serving one 2D+time slice per index. Because the app reconstructs a
    # single scan at a time, the dataset holds exactly one "subject", and its length is the
    # number of slices in that scan.
    def __init__(self, data_path, volume=True):
        # Timed so that the console shows how long reading and pre-processing took.
        time0 = time.time()
        # self.subjects = os.listdir(data_path)
        # The filename is fixed by agreement with the MATLAB app, which always writes its
        # export to retroAItemp.mat. The commented out line above is the more general
        # version that would process every file in the folder.
        self.subjects = ['retroAItemp.mat']
        logger.info('Data loading...')

        # Number of slices contributed by each subject, used by __len__ and __getitem__ to
        # map a flat index onto (subject, slice).
        lengths = []
        self.data = dict()
        for subject in self.subjects:
            self.data[subject] = {}

            # Get undersampled data
            # kData = undersampled k-space, fData = the sampling mask (which k-space
            # points were actually acquired).
            file2 = loadmat(os.path.join(data_path, subject))
            undersampled_kspace = file2['kData']
            mask = file2['fData']

            # Mouse data misses slice dimension because there is only one slice
            # MATLAB drops trailing singleton dimensions, so a single slice scan arrives
            # with one axis fewer. Appending it back keeps the axis order uniform.
            if len(undersampled_kspace.shape) == 4:
                undersampled_kspace = np.expand_dims(undersampled_kspace, axis=-1)
                mask = np.expand_dims(mask, axis=-1)

            # Data is in shape (coils, time, y, x (, z/slices))
            # transpose to (coils, time, z/slices, y, x)
            # The slice axis is moved forward so that slicing over slices in __getitem__
            # is a simple index on axis 2. The mask has no coil axis, hence its own
            # permutation: (time, y, x, slices) -> (time, slices, y, x).
            undersampled_kspace = np.transpose(undersampled_kspace, [0, 1, 4, 2, 3])
            mask = np.transpose(mask, [0, 3, 1, 2])

            # Pad to the fixed 256 x 256 matrix the trained network expects.
            undersampled_kspace, mask = zero_fill(undersampled_kspace, mask)

            # Normalize estimate and kspace
            # factor_estimate = np.percentile(np.abs(estimate), 95)
            # Scale so the brightest pixel of the image becomes 1. The network was trained
            # on data in this range, so feeding it raw scanner units would put it far
            # outside the range it knows. The commented out 95th percentile variant is
            # more robust to a single bright outlier, but is not the one used for training.
            #
            # The factor is accumulated one slice at a time. Taking the maximum of the per
            # slice maxima gives exactly the same number as a maximum over the whole
            # volume, but the complete image estimate never has to exist in memory at once.
            factor_estimate = 0.0
            for slice_index in range(undersampled_kspace.shape[2]):
                slice_estimate = image_from_kspace(undersampled_kspace[:, :, slice_index])
                factor_estimate = max(factor_estimate, np.abs(slice_estimate).max())

            # Because the Fourier transform is linear, scaling k-space scales the image by
            # the same factor. Dividing here is therefore equivalent to normalising the
            # image and transforming it back, but without a second full volume transform.
            #
            # complex64 rather than numpy's default complex128: the network consumes float
            # data at fp16/fp32, MPS cannot handle 64 bit floats at all, and this halves
            # both the memory held and the amount copied to each dataloader worker.
            undersampled_kspace = (undersampled_kspace / factor_estimate).astype(np.complex64)
            # int8 is what the mask is used as downstream, so store it that way instead of
            # keeping a float64 copy eight times larger.
            mask = mask.astype(np.int8)

            # Keep the factor so it can travel out with the reconstruction. The MATLAB app
            # calls this script once per coil and per dynamic, and each call would
            # otherwise normalise to its own maximum: a coil that barely sees the anatomy
            # would come back just as bright as one right on top of it. Sum-of-squares
            # combination assumes the opposite, namely that each coil image still carries
            # its own sensitivity weighting, so the factor has to be reported back for the
            # relative amplitudes to be restorable before the coils are combined.
            self.data[subject]['scale'] = float(factor_estimate)

            # Held as torch tensors in shared memory rather than as plain numpy arrays.
            # On macOS and Windows the dataloader starts its workers with the "spawn"
            # method, which pickles this whole dataset into every single worker: with
            # eight workers that is eight private copies of the data. A tensor whose
            # storage lives in shared memory is instead handed over as a reference, so all
            # workers map the same physical pages and startup no longer has to copy
            # hundreds of megabytes per worker.
            self.data[subject]['kspace_measurements'] = share(torch.from_numpy(undersampled_kspace))
            self.data[subject]['mask'] = share(torch.from_numpy(mask))

            # Check if image or volume
            # Two ways of cutting the data into independent samples for the network:
            if volume:
                # Take slices as slices
                # Normal case: one sample per anatomical slice, i.e. the slice axis of the
                # k-space array. The image estimate for a slice is recomputed on demand in
                # __getitem__, which is possible because the Fourier transform runs over
                # the last two axes only and so treats each slice independently.
                lengths.append(undersampled_kspace.shape[2])
            else:
                # Take freq dim as slices
                # Alternative: treat each line along the phase encoding direction as a
                # sample. Used for 2D (non volume) acquisitions.
                #
                # Here the sampling axis is one of the axes the Fourier transform runs
                # over, so a single line cannot be transformed on its own. The estimate is
                # therefore computed once for the whole volume and kept, as before. Only
                # the volume=True path benefits from deriving it on demand.
                lengths.append(undersampled_kspace.shape[-2])
                self.data[subject]['estimate'] = share(torch.from_numpy(
                    image_from_kspace(undersampled_kspace)[0].astype(np.complex64)))
            self.volume = volume


        logger.info(f'Finished pre-processing subject(s).')
        # Cumulative slice counts, so a flat index can be mapped back to a subject with a
        # single np.digitize call in __getitem__.
        self.lengths = np.cumsum(lengths)
        time2 = time.time()
        print("Processing time in seconds.. ", time2-time0)
        return


    def __len__(self):
        # The last cumulative sum is the total number of samples over all subjects.
        return self.lengths[-1]


    def __getitem__(self, index):
        # Map the flat index onto a subject, then onto a slice within that subject.
        # With only one subject (the usual case) idx is always 0 and index is unchanged.
        idx = np.digitize(index, self.lengths)
        if idx > 0:
            index -= self.lengths[idx - 1]
        subject = self.subjects[idx]

        # These are shared memory tensors; indexing them below produces views, so nothing
        # is copied until .contiguous() is called.
        kspace_measurements = self.data[subject]['kspace_measurements']
        imspace_undersampled_mask = self.data[subject]['mask']

        # Check if the input is 3D or 2D
        if self.volume:
            # Slice over slice direction
            # Pick one anatomical slice, keeping all time frames. .contiguous() copies
            # just this slice out of the shared volume, which matters for two reasons:
            # the copy is what gets handed back through the dataloader queue, and a view
            # would keep the entire underlying storage alive behind it.
            imspace_undersampled_mask = imspace_undersampled_mask[:, index, :, :].contiguous()
            kspace_measurements = kspace_measurements[:, :, index, :, :].contiguous()
            # The image estimate is derived here rather than stored, which is what keeps
            # the whole volume out of memory. Transforming this one slice gives exactly
            # the same result as transforming the volume and then indexing it, because
            # the transform acts only on the last two axes. [0] keeps the first coil,
            # matching the single coil assumption (see the note in __init__).
            # .numpy() is a view onto the same memory, so this costs no extra copy.
            estimate = torch.from_numpy(image_from_kspace(kspace_measurements.numpy())[0])
        else:
            # Slice over the frequency direction (H/y)
            # Pick one line along the phase encoding direction instead of one slice.
            # This axis is one the Fourier transform runs over, so the estimate cannot be
            # derived from a single line; it was computed once in __init__ instead.
            estimate = self.data[subject]['estimate'][:, :, index, :].contiguous()  # [D, dyn, W]
            imspace_undersampled_mask = imspace_undersampled_mask[:, :, index, :].contiguous()
            kspace_measurements = kspace_measurements[:, :, :, index, :].contiguous()

        # Placeholder coil sensitivities: all ones, i.e. a perfectly uniform coil, which
        # is consistent with the single coil assumption. Built for this slice only rather
        # than held for the whole volume, since an array of ones carries no information.
        # Real sensitivity maps (from ESPIRiT for instance) would replace this.
        coil_sense = torch.ones_like(kspace_measurements)

        # Everything is already complex64 and int8, so the conversions below are no-ops
        # kept for clarity rather than actual casts.
        data = {
            'subject': subject,
            'sense': coil_sense.to(torch.complex64), # [C, T, D, W]
            'mask': imspace_undersampled_mask.unsqueeze(0).unsqueeze(-1), # [1, T, D, W, 1]
            'measurements': kspace_measurements.to(torch.complex64), # [C, T, D, W]
            'estimate': estimate.to(torch.complex64), # [T, D, W]
            # The value everything was divided by, carried along so that the network can
            # write it out next to the reconstruction. Identical for every slice.
            # Explicitly float32: a plain Python float would be collated into a float64
            # tensor, which MPS refuses to accept when the batch is moved to the device.
            'scale': torch.tensor(self.data[subject]['scale'], dtype=torch.float32)
        }
        return data
