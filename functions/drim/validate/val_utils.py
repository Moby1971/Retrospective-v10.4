
import os
import torch
import pickle

import pandas as pd

from torch.utils.tensorboard import SummaryWriter
from validate.val_metrics import *

import logging
logger = logging.getLogger(__name__)

def define_metrics(config):
    return {
        'nrmse': nrmse,
        'ssim2d': lambda y, x: SSIM()(y, x).item(),
        'ssim4d': lambda y, x: SSIM(dim=4)(y, x).item(),
        # 'ms-ssim': lambda y, x: ms_ssim(y, x).item(),
        'ssim': lambda y, x: ssim(y, x, win_size=7),
        'psnr': psnr,
        # 'tenengrad-target': tenengrad,
        # 'tenengrad-reconstruction': lambda y, x: tenengrad(x, y),
        # 'gradient-entropy-target': gradient_entropy,
        # 'gradient-entropy-reconstruction': lambda y, x: gradient_entropy(x, y)
    }


def get_checkpoint_range(config, train_config):
    """Determine the range of checkpoints to validate."""
    if not config['min-max-checkpoint']:
        checkpoints = sorted(
            [int(f.split('checkpoint')[1].split('.')[0]) for f in os.listdir(os.path.join(config['train-dir'], 'network-parameters')) if 'checkpoint' in f]
        )
        return checkpoints[0], checkpoints[-1] + int(train_config['checkpoint-freq'])
    return map(int, config['min-max-checkpoint'].split())

def setup_logging(config):
    """Prepare logging paths and tensorboard writer."""
    log_dir = os.path.join(config['train-dir'], 'logs')
    os.makedirs(log_dir, exist_ok=True)
    csvname = os.path.join(log_dir, 'validation.csv')
    
    if not os.path.exists(csvname):
        pd.DataFrame(columns=['Checkpoint', 'T', 'Mask', 'Metric', 'Score', 'Data', 'Number of gaps', 'Temporal gaps', 'Dynamics', '4d sorted', 'Number of bins']).to_csv(csvname, index=False)
    
    writer = SummaryWriter(log_dir=os.path.join(log_dir, 'tensorboard', 'validate'))
    logger.info(f'Writing results to {csvname}')
    
    return csvname, writer

def get_random_bin_idxs(config, dataloader):
    """Generate randomized bin indices if needed."""
    # if not config['randomize-bins']:
    #     return [None]
    return [None]
    
    with open(os.path.join(dataloader.dataset.data_path, 'imspace_header'), 'rb') as f:
        nslice = pickle.load(f)[0][0]
    
    return [None, [torch.randperm(int(config['nbin']), generator=torch.manual_seed(n)) for n in range(nslice)]]
