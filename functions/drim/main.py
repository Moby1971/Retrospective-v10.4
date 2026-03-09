import sys
import os
import yaml
import logging
from train.logger import setup as log_setup
from train.training import train_model
from reconstruction.reconstruction import reconstruct

logger = logging.getLogger(__name__)


def training(config):
    # Create new folder to save training
    traindir = config['train']['train-dir']
    if os.path.exists(traindir):
        num = 2
        traindir = config['train']['train-dir'] + f'training_{num}'
        while os.path.exists(traindir):
            num += 1
            traindir = config['train']['train-dir'] + f'training_{num}'
    config['train']['train-dir'] = traindir
    # Make a folder for the logs and one for the network parameters
    os.makedirs(os.path.join(traindir, 'logs'))
    os.mkdir(os.path.join(traindir, 'network-parameters'))

    log_setup(
        True,
        os.path.join(traindir, 'logs', 'train_run_log.txt'),
        log_level=logging.INFO)
    logger.info(f"Storing train run in {config['train']['train-dir']}")

    # start training
    train_model(config)
    logger.info("Finished training!")
    return


def testing(config):
    logger.info('Reconstructing...')
    reconstruct(config)
    return


if __name__ == "__main__":
    if '-h' in sys.argv or '--help' in sys.argv:
        print(
            "Script contains two different programs:"
            "\t1) train: train a network."
            "\t2) reconstruct: reconstruct data using a pre-trained network."
            "For more details, specify which program you need help with.")

    # Load hyperparameters in yaml file
    yaml_file = sys.argv[6]
    with open(yaml_file, 'r') as file:
        config_params = yaml.safe_load(file)

    if sys.argv[1] == 'train':
        training(config_params)
    elif sys.argv[1] == 'reconstruct':
        testing(config_params)
    else:
        print("Option not available. Options are train or reconstruct.")
