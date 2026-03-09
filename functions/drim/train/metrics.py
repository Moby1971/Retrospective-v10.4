import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.measure import blur_effect
import functools
import torch


def nrmse(images, recons):
    score = 0
    num_slices = images.shape[0]
    for i in range(num_slices):
        image = images[i]
        recon = recons[i]
        dim1 = image.shape[0]
        dim2 = image.shape[1]
        score += np.sqrt(np.sum((image-recon)**2)/(dim1*dim2)) / np.mean(image)
    return score / num_slices


def nmse(gt, pred):
    return np.linalg.norm(gt-pred)**2/np.linalg.norm(gt)**2


def ssim(gt: np.ndarray, pred: np.ndarray, maxval: np.ndarray = None) -> float:
    """Compute Structural Similarity Index Metric (SSIM)"""

    if gt.ndim != pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = np.max(gt) if maxval is None else maxval

    _ssim = sum(
        structural_similarity(gt[slice_num], pred[slice_num], data_range=maxval, win_size=1) for slice_num in range(gt.shape[0])
    )
    return _ssim / gt.shape[0]


def psnr(gt, pred):
    return peak_signal_noise_ratio(gt, pred)


def blur_metric(image):
    return blur_effect(image)


def haarpsi3d(gt: np.ndarray, pred: np.ndarray, maxval: np.ndarray = None) -> float:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if gt.ndim != 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if gt.ndim != pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")
    reduction= 'mean'
    scales = 3
    subsample= True
    c= 30.0
    alpha = 4.2
    maxval = np.max(gt) if maxval is None else maxval
    _haarpsi = functools.partial(piq.haarpsi, scales=scales, subsample=subsample, c=c, alpha=alpha,
                                     data_range=maxval, reduction=reduction)
    __haarpsi = sum(
       _haarpsi(torch.from_numpy(gt[slice_num]).unsqueeze(0).unsqueeze(0).float(),\
                torch.from_numpy(pred[slice_num]).unsqueeze(0).unsqueeze(0).float()) for slice_num in range(gt.shape[0])
    ).numpy()
    return __haarpsi / gt.shape[0]