from torch.utils.data import DataLoader
from data.data_sampler_zp_sc import MRData

def initialize_dataloaders(config, dataset):
    """Initialize main and visualization dataloaders."""

    dataloader = DataLoader(
        dataset,  
        batch_size=int(config['batch-size']),
        shuffle=True,  
        drop_last=True,  
        num_workers=int(config['num-workers'])
    )

    dataloader_visualize = DataLoader(
        dataset,  
        batch_size=int(config['batch-size']),
        shuffle=False,  
        drop_last=True,  
        num_workers=int(config['num-workers'])
    )
    
    return dataloader, dataloader_visualize