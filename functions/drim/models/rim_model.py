import torch
import os
import configparser
import models.rim as rim  # Assuming rim contains the necessary model classes

def parse_kernel(kernel_str):
    """Convert kernel string into a list format."""
    return [None if k == 'None' else [int(x) for x in k] for k in kernel_str.split()]

def initialize_rim(nfeature, kernel, temporal_rnn, device):
    """Initialize RIM, InitRim, and GradRim models."""
    network = rim.RecurrentInferenceMachine(nfeature=nfeature, kernel=kernel, temporal_rnn=temporal_rnn, mute=False)
    initrim = rim.InitRim(2, 2 * [nfeature], kernel, mute=False)

    network = network.to(device=device)
    initrim = initrim.to(device=device)

    return network, initrim

def load_model(config):
    """Load a pre-trained RIM (Recurrent Inference Machine) model along with its configurations"""

    # Load training configs
    old_config = configparser.ConfigParser()
    old_config.read(os.path.join(config['train-dir'], 'config.ini'))
    old_config = old_config['train']
    
    nfeature = int(old_config['nfeature'])
    
    kernel = parse_kernel(old_config['kernel'])

    network, initrim = initialize_rim(nfeature, kernel, old_config.getboolean('temporal-rnn'), config['device'])
    
    # Parse Fourier dimensions if provided
    fourier_dim = [int(d) for d in old_config['fourier-dim'].split()] if 'fourier-dim' in old_config else -1
    gradrim = rim.GradRim(fourier_dim=fourier_dim)

    # Load model checkpoint
    load = torch.load(config['saved-model'], map_location=lambda storage, loc: storage.cpu())
    
    # Load weights
    network.load_state_dict(load['rim'])
    try:
        initrim.load_state_dict(load['initrim'])
    except KeyError:
        initrim.load_state_dict(load['init'])
    
    optimizer = torch.optim.Adam(list(network.parameters()) + list(initrim.parameters()), lr=float(old_config['lr']))
    optimizer.load_state_dict(load['optimizer'])

    # Extract step count
    step = int(config['saved-model'].split('checkpoint')[1].split('.pt')[0]) + 1

    return network, initrim, gradrim, optimizer, step