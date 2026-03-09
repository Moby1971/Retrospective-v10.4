import torch
from models.initialize import load_model
from validate.val_utils import *
import torch.utils.data as data
from data.data_sampler import MRData
# import h5py
from scipy.io import savemat
import sys

import logging
logger = logging.getLogger(__name__)


def reconstruct(config):
    """Main function to reconstruct k-space images using DRIM model."""
    # Get data
    data_directory = sys.argv[4]
    dataset = MRData(data_directory)
    dataloader = data.DataLoader(dataset, batch_size=int(config['reconstruct']['batch-size']), drop_last=False,
        num_workers=int(config['reconstruct']['num-workers']))

    # Load model
    model_path = sys.argv[2]
    checkpoint = sys.argv[3]
    saved_model = os.path.join(model_path, 'network-parameters', f'checkpoint{checkpoint}.pt')
    network, initrim, gradrim = load_model(config['train'], saved_model)
    logger.info(f"Loaded network.")

    # Perform reconstruction
    with torch.no_grad():
        print("Reconstruction begins!")
        reconstruct_data_per_slice(config, dataloader, network, initrim, gradrim)
        print("Done with reconstruction!")
    return


def save_reconstruction_in_mat(recons, subject):
    print("saving subject...", subject)
    reconstructions = torch.cat(recons, 2)
    reconstruction = torch.view_as_complex(reconstructions).cpu().numpy().transpose((1, 0, 2, 3))
    file_name = 'retroAItemp_DRIM'
    mat_file = sys.argv[5] + file_name + '.mat'
    savemat(mat_file, {'aiData': np.abs(reconstruction)})
    recons = []
    return recons


def preprocess_input(estimate, measurements, sense, mask, device):
    # Transpose dimensions to make slices the batch dimension
    estimate = np.transpose(estimate, [2, 1, 0, 3]) # (batch/slices, time, 1, width)
    measurements = np.transpose(measurements, [3, 1, 2, 0, 4])  # (batch/slices, coil, time, 1, width)
    sense = np.transpose(sense, [3, 1, 2, 0, 4])  # (batch/slices, coils, time, 1, width)
    mask = np.transpose(mask, [3, 1, 2, 0, 4, 5])  # (batch/slices, 1, time, 1, width, 1)

    # preprocess data
    estimate = torch.view_as_real(estimate.to(device=device, dtype=torch.cfloat))
    measurements = torch.view_as_real(measurements.to(device=device, dtype=torch.cfloat))
    sense = torch.view_as_real(sense.to(device=device, dtype=torch.cfloat))
    mask = mask.to(device=device, dtype=torch.bool)
    return estimate, measurements, sense, mask


def model_steps(network, initrim, gradrim, estimate, measurements, sense, mask, config):
    hidden = initrim(estimate.moveaxis(-1, 1))
    # Data through model
    for _ in range(int(config['train']['niteration'])):
        gradient = gradrim(estimate, measurements, sense, mask)
        network_input = torch.cat((estimate.moveaxis(-1, 1), gradient), 1)
        estimate_step, hidden = network(network_input, hidden)
        estimate = estimate + estimate_step.moveaxis(1, -1)
    return estimate


def reconstruct_data_per_slice(config, dataloader, network, initrim, gradrim):
    device = config['train']['device']
    recons = []
    subject_name = 'name'
    count = 0
    for batch in dataloader:
        subject = batch['subject']
        # Save data in h5 file if new subject
        if subject != subject_name and count > 0:
            recons = save_reconstruction_in_mat(recons, subject)
        count += 1
        # Load data from batch
        estimate = batch['estimate'] # (batch, time, slices, width)
        measurements = batch['measurements']
        sense = batch['sense']
        mask = batch['mask']

        # prepare data for the model
        estimate, measurements, sense, mask = preprocess_input(estimate, measurements, sense, mask, device)

        # Put data through the model
        output_estimate = model_steps(network, initrim, gradrim, estimate, measurements, sense, mask, config)

        # Combine the slices into one matrix
        recons.append(output_estimate)
        subject_name = subject

    # Save last subject
    save_reconstruction_in_mat(recons, subject)
    return

