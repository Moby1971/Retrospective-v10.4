import os
import time
import torch

def time_since(start):
    """Calculate elapsed time since a given start point."""
    now = time.time()
    s = now - start
    m = int(s // 60)
    h = int(m // 60)
    return h, m % 60, int(s % 60)

def save_checkpoint(network, initrim, optimizer, step, traindir, epoch, logger, start, config):
    """Save model checkpoint at given intervals."""
    checkpoint_path = os.path.join(traindir, f'checkpoint{step + 1}.pt')

    torch.save({
        'rim': network.state_dict(),
        'initrim': initrim.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, checkpoint_path)

    elapsed = time_since(start)
    logger.info(f'Checkpoint saved at epoch {epoch + 1}, iteration {step + 1} | Time elapsed: {elapsed[0]}h {elapsed[1]}m {elapsed[2]}s')

    if elapsed[0] + elapsed[1] / 60. > float(config['time-limit']):
        logger.info('Time limit exceeded, stopping training...')
        return True
    return False
