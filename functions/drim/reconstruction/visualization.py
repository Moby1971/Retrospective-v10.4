import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

from .reconstruction_utils import inverse_fft2_shift

def plot_results(target, reconstruction):
    """Plots ground truth, reconstructed images, and k-space in a 2x2 layout."""

    slice_idx = 0
    # for t in range(target.shape[0]):
    t = 10
    fig, axes = plt.subplots(2, figsize=(10, 10))
    axes[0].imshow(np.abs(target[t, slice_idx, :, : ]), cmap='gray')
    axes[0].set_title("Ground Truth")
    axes[0].axis('off')

    axes[1].imshow(np.abs(reconstruction[t, slice_idx,:, :]), cmap='gray')
    axes[1].set_title("Reconstructed Image")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

def save_reconstructed_data(config, metrics, target, reconstruction, file_path, kspace):
    """Saves the reconstructed data to a .mat file."""
    to_evaluate = [(reconstruction, target)]
    metric_scores = {}
    for metric in config['metrics'].split():
        # Iterate over all evaluation pairs (label, r, t)
        for r, t in to_evaluate:
            t = np.linalg.norm(t, axis=-1)
            r = np.linalg.norm(r, axis=-1)
            score = metrics[metric](t, r)
            metric_scores[metric] = score

    mat_data = {
        'reconstruction': reconstruction,
        'GT': target,
        'metrics': metric_scores,
        'kspace': kspace
    }

    file_name = os.path.basename(file_path)
    save_path = os.path.join('Reconstructed/f50-bright', file_name.replace('.mat', '') + '_DRIM.mat')
    savemat(save_path, mat_data)
    print(f"Reconstructed data saved to {save_path}")
