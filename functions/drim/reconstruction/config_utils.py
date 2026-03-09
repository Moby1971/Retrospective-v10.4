import os
import configparser
import shutil

def load_train_config(config):
    """Loads training configuration from config.ini."""
    config_path = os.path.join(config['DEFAULT']['train-dir'], 'config.ini')
    print("loading from", config_path)
    train_config = configparser.ConfigParser()
    train_config.read(config_path)
    return train_config['train']


def prepare_file(config, file, recon='val'):
    """Prepares directories and copies files."""
    file_path = os.path.join(config['val-data-dir'], file)
    print("CHECK ", file_path)
    if recon == 'val':
        temp_path = config['val-data-dir'][:-4] + "/save_recon/"
    else:
        temp_path = config['test-data-dir'][:-4] + "/save_test_recon/"
    temp_dir = os.path.join(temp_path, os.path.splitext(file)[0])
    os.makedirs(temp_dir, exist_ok=True)
    shutil.copy(file_path, temp_dir)
    return file_path, temp_dir
