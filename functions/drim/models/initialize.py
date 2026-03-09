import torch
import models.rim as rim
import os
import re
import glob

def load_model(config, checkpoint):
    nfeature = int(config['nfeature'])
    kernel = []
    for kern in config['kernel'].split():
        if kern == 'None':
            kernel.append(None)
        else:
            kernel.append([int(k) for k in kern])
    network = rim.RecurrentInferenceMachine(
        nfeature=nfeature, kernel=kernel,
        temporal_rnn=config['temporal-rnn'])
    initrim = rim.InitRim(2, 2 * [nfeature], kernel)
    
    # if 'fourier-dim' in config:
    #     fourier_dim = [int(d) for d in config['fourier-dim'].split()]
    # else:
    #     fourier_dim = -1
    # gradrim = rim.GradRim(fourier_dim=fourier_dim)
    gradrim = rim.GradRim(fourier_dim=[config['fourier-dim']])

    checkpoint_dir = os.path.join(config['train-dir'], 'network-parameters')

    # # Find all checkpoint files in the directory
    # checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint*.pt"))
    # print(checkpoint_files)
    # # Extract the numbers from the filenames
    # def extract_checkpoint_number(filename):
    #     # Use regex to extract the number from the filename
    #     match = re.search(r"checkpoint(\d+)\.pt", filename)
    #     if match:
    #         return int(match.group(1))
    #     return -1  # Return -1 if no number is found
    #
    # # Find the file with the highest checkpoint number
    # latest_checkpoint = max(checkpoint_files, key=extract_checkpoint_number)
    load = torch.load(checkpoint, map_location=lambda storage, loc: storage.cpu())
    network.load_state_dict(load['rim'])
    initrim.load_state_dict(load['initrim'])

    network = network.to(device=config['device'])
    initrim = initrim.to(device=config['device'])
    
    return network, initrim, gradrim
