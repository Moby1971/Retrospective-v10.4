import timeit
import torch
import numpy as np
from torch.cuda.amp import autocast
import logging
import models.rim as rim

def time_model(config):
    """Times the inference process of a Recurrent Inference Machine (RIM)"""
    
    logger = logging.getLogger(__name__)

    # Extract config params
    nfeature = int(config['nfeature']) 
    ncoil = int(config['ncoil'])
    ndynamic = int(config['ndynamic'])
    nlocation = int(config['nlocation'])
    width = int(config['width'])
    batch = int(config['height'])


    kernel = []
    for kern in config['kernel'].split():
        # Handle 'None' kernel case and convert other kernels into integers
        if kern == 'None':
            kernel.append(None)
        else:
            kernel.append([int(k) for k in kern])

    # Init RIM
    network = rim.RecurrentInferenceMachine(
        nfeature=nfeature, kernel=kernel,
        temporal_rnn=config.getboolean('temporal-rnn'), mute=False)
    initrim = rim.InitRim(
        2, 2 * [nfeature], kernel, mute=False)

    network = network.to(device=config['device'])
    initrim = initrim.to(device=config['device'])

    # Init GradRim
    gradrim = rim.GradRim(
        fourier_dim=[int(d) for d in config['fourier-dim'].split()])
    network.eval()

    # Creates random data tensors for the model
    mask = torch.tensor(np.random.choice([True, False],
        size=(batch, 1, ndynamic, nlocation, width, 1))).cuda()  # Mask for the measurements
    estimate = torch.rand(
        batch, ndynamic, nlocation, width, 2, dtype=torch.float).cuda()  # Initial estimate
    measurements = torch.rand(
        batch, ncoil, ndynamic, nlocation, width, 2, dtype=torch.float).cuda()  # Measurements
    sense = torch.rand(
        batch, ncoil, 1, nlocation, width, 2, dtype=torch.float).cuda()  # Sensing matrix
    gradient = torch.zeros(batch, 2, ndynamic, nlocation, width).to(estimate)  # Initial gradient


    def inference():
        est = estimate
        grad = gradient
        with autocast():  # Enable automatic mixed precision for faster computation
            hidden = initrim(est.moveaxis(-1, 1))  
        for iteration in range(int(config['niteration'])):
            with autocast():  # Use mixed precision for further computations
                if iteration != 0:
                    grad = gradrim(est, measurements, sense, mask) 
                network_input = torch.cat(
                    (est.moveaxis(-1, 1), grad), 1)  # Prepare input for the network
                estimate_step, hidden = network(network_input, hidden) 
                est = est + estimate_step.moveaxis(1, -1)  

    # Time inference process using timeit module
    with torch.no_grad(): 
        logger.info(
            timeit.repeat(
                'inference()', 
                globals=locals(),  
                number=int(config['nbatch']),  
                repeat=int(config['nrepeat']) 
            )
        )
