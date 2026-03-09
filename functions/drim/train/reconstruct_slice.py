import torch
from reconstruction.reconstruction_utils import perform_iterations, inverse_fft2_shift
from validate.val_utils import *


def get_image(im):
    # Concatenate frequency dimension
    im = torch.cat(im, 0)
    # Get back to shape (time, slices, y, x)
    image = torch.view_as_complex(im).cpu().numpy().transpose((1, 2, 3, 0))
    return image


def reconstruct_single_image(config, compute_loss, loss_weights, dataloader, network, initrim, gradrim, autocast_enabled):
    """Reconstructs a single image or batch of images."""
    device = config['device']
    recons, targets, losses = [], [], []
    rec, tar = [], []
    subject_name = 'name'
    count = 0
    for batch in dataloader:
        # data = extract_data(batch, device)
        subject = batch['subject']
        if subject != subject_name and count > 0:
            # plot_results(tar, rec)
            recons.append(get_image(rec))
            targets.append(get_image(tar))
            rec, tar = [], []

        target = torch.view_as_real(batch['target'].to(device=config['device'], dtype=torch.cfloat))
        estimate = torch.view_as_real(batch['estimate'].to(device=config['device'], dtype=torch.cfloat))
        measurements = torch.view_as_real(batch['measurements'].to(device=config['device'], dtype=torch.cfloat))
        sense = torch.view_as_real(batch['sense'].to(device=config['device'], dtype=torch.cfloat))
        mask = batch['mask'].to(device=config['device'], dtype=torch.bool)

        # with torch.cuda.amp.autocast(enabled=autocast_enabled):
        hidden = initrim(estimate.moveaxis(-1, 1))

        estimate, loss = perform_iterations(device, target,
            estimate, hidden, initrim, gradrim, network,
            measurements, sense, mask, int(config['niteration']), compute_loss, loss_weights, autocast_enabled
        )
        rec.append(estimate)
        tar.append(target)
        losses.append(loss.cpu().item())
        subject_name = subject
        count += 1
    tar = get_image(tar)
    recons.append(get_image(rec))
    targets.append(tar)
    final_loss = np.mean(losses)
    return targets, recons, final_loss


def validate_reconstruction(config, dataloader, target, reconstruction, checkpoint, metrics):
    """Validates the reconstructed images and computes metrics."""
    subject = os.path.basename(config['data-dir'])
    if not hasattr(dataloader.dataset, 'dynamic_slice'):
        dyns = 120
    log_metrics(config, metrics, target, reconstruction, subject, dyns)
    return


def log_metrics(config, metrics, target, reconstructions, subject, dyns):
    """Logs the computed metrics to TensorBoard and prints them."""
    is_sorted = "4d-sorted" if config["sorted"] else "unsorted"
    scans = len(target)
    # Iterate over all metrics specified in the config
    for metric in config['metrics'].split():
        # Iterate over all evaluation pairs (label, r, t)
        scores = []
        for i in range(scans):
            time, slices, _, _ = target[i].shape
            for t in range(time):
                for s in range(slices):
                    # print(len(target), target[i].shape)
                    tar = np.abs(target[i][t][s])
                    rec = np.abs(reconstructions[i][t][s])
                    tar = tar/np.max(tar)
                    rec = rec/np.max(rec)
                    score = metrics[metric](tar, rec)
                    scores.append(score)
        mean_score = np.mean(scores)
        # Print the metric to the logger
        logger.info(
            f"Checkpoint {subject} "
            f"mask {metric}: {mean_score}"
        )
    return

