
import torch
from torch.cuda.amp import autocast


def initialize_hidden(train_config, initrim, estimate):
    if train_config.getboolean('autocast'):
        with autocast():
            hidden = initrim(estimate.moveaxis(-1, 1))
    else:
        hidden = initrim(estimate.moveaxis(-1, 1))
    return hidden

def iterate_network(config, train_config, network, gradrim, estimate, measurements, sense, mask, hidden):
    for _ in range(int(config['niteration'])):
        gradient = gradrim(estimate, measurements, sense, mask)
        network_input = torch.cat((estimate.moveaxis(-1, 1), gradient), 1)
        if train_config.getboolean('autocast'):
            with autocast():
                estimate_step, hidden = network(network_input, hidden)
        else:
            estimate_step, hidden = network(network_input, hidden)
        estimate = estimate + estimate_step.moveaxis(1, -1)
    return estimate

def restore_random_bins(random_bin_idxs, estimate):
    for i, idx in enumerate(random_bin_idxs):
        inv_idx = torch.argsort(idx)
        newestimate = estimate[:, :, i].clone()
        estimate[:, inv_idx, i] = newestimate
    return estimate

def prepare_batch(config, batch):
    target = torch.view_as_real(batch['target'].to(device=config['device'], dtype=torch.cfloat))
    estimate = torch.view_as_real(batch['estimate'].to(device=config['device'], dtype=torch.cfloat))
    measurements = torch.view_as_real(batch['measurements'].to(device=config['device'], dtype=torch.cfloat))
    sense = torch.view_as_real(batch['sense'].to(device=config['device'], dtype=torch.cfloat))
    mask = batch['mask'].to(device=config['device'], dtype=torch.bool)
    return target, estimate, measurements, sense, mask

def adjust_random_bins(random_bin_idxs, estimate, measurements, mask):
    for i, idx in enumerate(random_bin_idxs):
        newestimate = estimate[:, :, i].clone()
        estimate[:, idx, i] = newestimate
        newmeasurements = measurements[:, :, :, i].clone()
        measurements[:, :, idx, i] = newmeasurements
        newmask = mask[:, :, :, i].clone()
        mask[:, :, idx, i] = newmask
    return estimate, measurements, mask

from reconstruction.data_processing import extract_data
def process_batches(config, train_config, dataloader, initrim, gradrim, network, random_bin_idxs):
    recons, targets, viz, trgts = [], [], [], []
    for batch in dataloader:
        data = extract_data(batch, train_config['device'])
        viz.append(batch['estimate'])
        trgts.append(batch['target'])
        
        
        with autocast():
            hidden = initrim(data['estimate'].moveaxis(-1, 1))
        
        estimate = iterate_network(config, train_config, network, gradrim, data['estimate'], data['measurements'], data['sense'], data['mask'], hidden)
        
        recons.append(estimate)
        targets.append(data['target'])
    return recons, targets, viz, trgts