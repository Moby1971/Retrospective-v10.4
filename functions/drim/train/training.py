import torch
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from data.data_sampler import MRData

from train.metrics import nrmse, nmse, ssim, psnr, blur_metric
from models.rim_model import *
from train.train_utils import *
from reconstruction.reconstruction import *
from train.reconstruct_slice import *
import models.rim as rim
from torch.utils.data import DataLoader, Dataset

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  # Helps reduce fragmentation
torch.backends.cudnn.benchmark = True  # Optimizes CUDA convolution kernels
torch.backends.cudnn.enabled = True


def validation_loop(config, network, initrim, gradrim, loss_weights, compute_loss, writer, val_config,
                    dataloader, val_step, metrics):
    # Perform validation periodically
    final_val_loss = []
    dataset = MRData(val_config['val-data-dir'], val_config['val-undersampled-data-dir'])
    val_loader = data.DataLoader(dataset, batch_size=int(config['batch-size']), drop_last=False,
        num_workers=int(config['num-workers']))

    # Reconstruct validation data
    with torch.no_grad():  # Disable gradient computation
        autocast = False  # config.getboolean('autocast')
        # Reconstruct validation data
        target, reconstruction, val_loss = reconstruct_single_image(config, compute_loss, loss_weights,
                                                                    val_loader, network, initrim, gradrim,
                                                                    autocast)
        final_val_loss.append(val_loss)
        # Calculate validation metrics
        validate_reconstruction(val_config, val_loader, target, reconstruction, val_step + 1, metrics)
    val_step += 1
    return final_val_loss


def train_loop(config, val_config, dataloader, step, network, initrim, gradrim, compute_loss, loss_weights, optimizer,
               mixed_precision_scaler, writer, traindir, start, scheduler):
    val_step = 0

    name_project = 'dynamicMRI'
    metrics = define_metrics(config)

    # validation_loop(config, network, initrim, gradrim, loss_weights, compute_loss, writer, val_config,
    #                 dataloader, val_step, metrics)
    best_val_loss = 10000
    for epoch in range(int(step / len(dataloader)), int(config['nepoch'])):
        running_loss = 0.0
        # for batch in tqdm(dataloader, leave=False):
        print("Training, epoch ...", epoch)
        time0 = time.time()
        for batch in dataloader:
            # metrics = define_metrics(config)

            estimate, batch_loss = train_one_batch(batch, network, initrim, gradrim, compute_loss, loss_weights, config, optimizer, mixed_precision_scaler)
            # print(batch_loss)
            running_loss += batch_loss

            # Log training loss
            writer.add_scalar('Train/Loss/FinalLoss', batch_loss, step + 1)
            # Validation after this many iterations
            if step % int(config['validation-freq']) == int(config['validation-freq']) - 1:
                val_loss = validation_loop(config, network, initrim, gradrim, loss_weights, compute_loss, writer, val_config,
                    dataloader, val_step, metrics)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(network, initrim, optimizer, step, traindir, epoch, logger, start, config,
                                    val_loss)

                logger.info(f"Running validation at step {step + 1}...")
            step += 1
        scheduler.step()
        time1 = time.time()
        print("epoch time in minutes is: ", (time1-time0)/60)
    return


def train_model(config):
    """Initialize and train the RIM model."""
    val_config = config['validate']
    config = config['train']

    logger = logging.getLogger(__name__)
    dataset = MRData(config['data-dir'], config['undersampled_data_dir'])
    # dataloader, dataloader_visualize = initialize_dataloaders(config, dataset)
    train_loader = DataLoader(dataset, batch_size=int(config['batch-size']), shuffle=True, drop_last=True,
        num_workers=int(config['num-workers']))

    if 'saved-model' in config:
        network, initrim, gradrim, optimizer, step = load_model(config)
    else: 
        nfeature = int(config['nfeature'])
        kernel = parse_kernel(config['kernel'])
        network, initrim = initialize_rim(nfeature, kernel, config['temporal-rnn'], config['device'])
        optimizer = torch.optim.Adam(list(network.parameters()) + list(initrim.parameters()), lr=float(config['lr']))
        # TO DO: wat als fourier dim meerdere waardes heeft?
        # gradrim = rim.GradRim(fourier_dim=[int(d) for d in config['fourier-dim'].split()])
        gradrim = rim.GradRim(fourier_dim=[config['fourier-dim']])
        step = 0

    # # set model to float16
    # network.half()
    # initrim.half()
    # gradrim.half()

    network.train()
    initrim.train()
    gradrim.train()
    compute_loss, optimizer, scheduler, loss_weights = initialize_training_components(config, network, initrim)

    logdir = os.path.join(config['train-dir'], 'logs', 'tensorboard', 'train')
    writer = SummaryWriter(log_dir=logdir)

    traindir = os.path.join(config['train-dir'], 'network-parameters')
    mixed_precision_scaler = GradScaler(
    init_scale=2.**10,  # Start with a higher scale (default: 2.**16)
    growth_interval=1000,  # Increase if scaling is too aggressive
    enabled=config['autocast']
    )

    start = time.time()
    logger.info('Starting training...')
    train_loop(config, val_config, train_loader, step, network, initrim, gradrim, compute_loss, loss_weights, optimizer,
               mixed_precision_scaler, writer, traindir, start, scheduler)
    return


