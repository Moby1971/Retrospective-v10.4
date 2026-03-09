import numpy as np
import torch
from numpy.fft import fftshift, ifftshift, ifft2

def perform_iterations(device, target, estimate, hidden, initrim, gradrim, network, measurements, sense, mask, num_iterations, compute_loss, loss_weights, autocast_enabled):
    """Performs iterative updates on the estimate using the trained network."""
    
    loss = torch.tensor(0.0).to(device)

    with torch.cuda.amp.autocast(enabled=autocast_enabled):
        hidden = initrim(estimate.moveaxis(-1, 1))

    
    for iteration, weight in zip(range(num_iterations), loss_weights):
        gradient = gradrim(estimate, measurements, sense, mask)
        network_input = torch.cat((estimate.moveaxis(-1, 1), gradient), 1)
        
        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            estimate_step, hidden = network(network_input, hidden)
        
        estimate = estimate + estimate_step.moveaxis(1, -1)
        it_loss = compute_loss(estimate, target)
        loss += weight * it_loss
    
    return estimate, loss

def inverse_fft2_shift(kspace):
    """Computes the inverse FFT and shifts the result."""
    return np.abs(fftshift(ifft2(ifftshift(kspace)), axes=(-2, -1)))
