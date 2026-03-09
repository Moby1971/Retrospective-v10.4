import torch
import os
import time
import math
import logging
from torch.cuda.amp import GradScaler, autocast
from torch import nn
import numpy as np
 
logger = logging.getLogger(__name__)


def initialize_training_components(config, network, initrim):
    # compute_loss = BrightnessWeightedLoss()
    compute_loss = torch.nn.L1Loss(reduction='mean')
    optimizer = torch.optim.Adam(list(network.parameters()) + list(initrim.parameters()), lr=float(config['lr']))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(ms) for ms in config['milestones'].split()], gamma=float(config['gamma']))
    loss_weights = torch.logspace(-1, 0, steps=int(config['niteration'])).to(next(network.parameters()))
    return compute_loss, optimizer, scheduler, loss_weights


def train_one_batch(batch, network, initrim, gradrim, compute_loss, loss_weights, config, optimizer, mixed_precision_scaler):
    """Perform a single training batch step."""
    optimizer.zero_grad()
    # Move tensors to device and ensure correct dtype
    # target = torch.view_as_real(batch['target'].to(device=config['device'], dtype=torch.cfloat)).half()
    # estimate = torch.view_as_real(batch['estimate'].to(device=config['device'], dtype=torch.cfloat)).half()
    # measurements = torch.view_as_real(batch['measurements'].to(device=config['device'], dtype=torch.cfloat)).half()
    # sense = torch.view_as_real(batch['sense'].to(device=config['device'], dtype=torch.cfloat)).half()
    # mask = batch['mask'].to(device=config['device'], dtype=torch.bool)

    target = torch.view_as_real(batch['target'].to(device=config['device'], dtype=torch.cfloat))
    estimate = torch.view_as_real(batch['estimate'].to(device=config['device'], dtype=torch.cfloat))
    measurements = torch.view_as_real(batch['measurements'].to(device=config['device'], dtype=torch.cfloat))
    sense = torch.view_as_real(batch['sense'].to(device=config['device'], dtype=torch.cfloat))
    mask = batch['mask'].to(device=config['device'], dtype=torch.bool)

    # Initialize total loss tracking
    total_loss = torch.tensor(0.0, device=config['device'])
    
    # Initialize hidden state
    # with autocast(enabled=config['autocast']):
    hidden = initrim(estimate.moveaxis(-1, 1))

    # Forward iterations 8 time steps?
    for iteration, weight in zip(range(int(config['niteration'])), loss_weights):
        gradient = gradrim(estimate, measurements, sense, mask)
        network_input = torch.cat((estimate.moveaxis(-1, 1), gradient), 1)
        # Single autocast call for the entire loop (more stable precision)
        # with torch.cuda.amp.autocast(enabled=config['autocast']):
        estimate_step, hidden = network(network_input, hidden)

        # Update the estimate
        estimate = estimate + estimate_step.moveaxis(1, -1)

        # Compute loss for this iteration
        with autocast(enabled=False):
            it_loss = compute_loss(estimate.float(), target.float())

        total_loss += weight * it_loss

        # backpropagation every 4 iteration because of memory??
        # if iteration % int(config['truncate']) == int(config['truncate']) - 1:
        #     # print(torch.cuda.memory_allocated() / 1e6, "MB")
        #     total_loss = total_loss / int(config['truncate'])
        #
        #     # Try backpropagation after every time step...
        #     mixed_precision_scaler.scale(total_loss).backward()
        #     # save_loss = total_loss.detach().numpy()
        #     total_loss = torch.Tensor([0.]).to(
        #         device=config['device'], dtype=torch.float)
        #     hidden = [h.detach() for h in hidden]
        #     estimate = estimate.detach()
        #     torch.cuda.empty_cache()

    total_loss = total_loss / int(config['truncate'])
    mixed_precision_scaler.scale(total_loss).backward()
    estimate = estimate.detach()

    # total_loss.detach_()
    # batch_loss = total_loss.copy()
    # total_loss.zero_()
    # hidden = [h.detach().requires_grad_() for h in hidden]
    # Optimizer step and scaler update
    mixed_precision_scaler.step(optimizer)
    mixed_precision_scaler.update()
    # torch.cuda.empty_cache()

    return estimate, total_loss.item()


def save_checkpoint(network, initrim, optimizer, step, traindir, epoch, logger, start, config, loss):
    """Save model checkpoint at given intervals."""
    torch.save({'rim': network.state_dict(), 'initrim': initrim.state_dict(), 'optimizer': optimizer.state_dict()},
               os.path.join(traindir, f'checkpoint{step + 1}_loss:{np.round(loss, 5)}.pt'))
    return


def log_progress(logger, writer, running_loss, step, epoch, config):
    """Log training progress and loss to TensorBoard and console."""
    writer.add_scalar('Train/Loss/FinalLoss', running_loss / int(config['print-freq']), step + 1)
    logger.info(f'[{epoch + 1}, {step + 1}] train loss: {running_loss / int(config["print-freq"]):e}.')
    return 0.0  # Reset running loss
