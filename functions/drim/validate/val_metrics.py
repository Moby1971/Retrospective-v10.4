import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def complex_to_magnitude(data):
    """Convert complex data (..., 2) to magnitude"""
    if isinstance(data, torch.Tensor):
        return torch.abs(torch.view_as_complex(data))
    else:  # numpy
        return np.abs(data[..., 0] + 1j * data[..., 1])

def nrmse(gt, pred):
    """Normalized RMSE that handles 2D-4D inputs"""
    
    # print(gt.shape, pred.shape)
    # gt_mag = complex_to_magnitude(gt)
    # pred_mag = complex_to_magnitude(pred)
    gt_mag = gt
    pred_mag = pred
    # Handle all possible dimensions
    if gt_mag.ndim == 2:  # [H, W]
        axis = (0, 1)
    elif gt_mag.ndim == 3:  # [D, H, W]
        axis = (1, 2)
    elif gt_mag.ndim == 4:  # [T, D, H, W]
        axis = (2, 3)
    else:
        raise ValueError(f"Unsupported shape {gt_mag.shape}. Expected 2D-4D data")
    
    rmse = np.sqrt(np.mean((gt_mag - pred_mag)**2, axis=axis))
    norm = np.mean(gt_mag, axis=axis)
    return np.mean(rmse / np.where(norm > 0, norm, 1))  # avoid div by zero

def ssim(gt, pred, maxval=None, win_size=7):
    """SSIM that handles 2D–4D inputs"""
    # gt_mag = complex_to_magnitude(gt)
    # pred_mag = complex_to_magnitude(pred)
    gt_mag = gt
    pred_mag = pred
    if gt_mag.shape != pred_mag.shape:
        raise ValueError(f"Shape mismatch: {gt_mag.shape} vs {pred_mag.shape}")

    maxval = gt_mag.max() if maxval is None else maxval

    def safe_ssim(a, b):
        h, w = a.shape
        ws = min(win_size, h, w)
        if ws < 3:
            # Too small to compute SSIM meaningfully
            return 0.0  # Or np.nan, or skip this slice
        if ws % 2 == 0:
            ws -= 1  # SSIM requires odd win_size
        return structural_similarity(a, b, data_range=maxval, win_size=ws)

    if gt_mag.ndim == 2:  # [H, W]
        return safe_ssim(gt_mag, pred_mag)

    elif gt_mag.ndim == 3:  # [D, H, W]
        scores = [safe_ssim(gt_mag[d], pred_mag[d]) for d in range(gt_mag.shape[0])]

    elif gt_mag.ndim == 4:  # [T, D, H, W]
        scores = [safe_ssim(gt_mag[t, d], pred_mag[t, d])
                  for t in range(gt_mag.shape[0])
                  for d in range(gt_mag.shape[1])]

    return np.mean(scores)

def psnr(gt, pred):
    """PSNR that handles 2D-4D inputs"""
    # gt_mag = complex_to_magnitude(gt)
    # pred_mag = complex_to_magnitude(pred)
    gt_mag = gt
    pred_mag = pred
    data_range = gt_mag.max() - gt_mag.min()
    
    if gt_mag.ndim == 2:
        return peak_signal_noise_ratio(gt_mag, pred_mag, data_range=data_range)
    
    # For 3D/4D, compute PSNR slice-wise
    if gt_mag.ndim == 3:  # [D, H, W]
        psnrs = [peak_signal_noise_ratio(gt_mag[d], pred_mag[d], 
                data_range=data_range)
               for d in range(gt_mag.shape[0])]
    
    elif gt_mag.ndim == 4:  # [T, D, H, W]
        psnrs = [peak_signal_noise_ratio(gt_mag[t,d], pred_mag[t,d],
                data_range=data_range)
               for t in range(gt_mag.shape[0])
               for d in range(gt_mag.shape[1])]
    
    return np.mean(psnrs)