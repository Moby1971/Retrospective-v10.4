import torch
import torch.utils.data as data
from data.data_sampler_zp_sc import MRData

def extract_data(batch, device):
    """Extract and move batch data to the specified device."""
    return {
        'target': torch.view_as_real(batch['target'].to(device=device, dtype=torch.cfloat)),
        'estimate': torch.view_as_real(batch['estimate'].to(device=device, dtype=torch.cfloat)),
        'measurements': torch.view_as_real(batch['measurements'].to(device=device, dtype=torch.cfloat)),
        'sense': torch.view_as_real(batch['sense'].to(device=device, dtype=torch.cfloat)),
        'mask': batch['mask'].to(device=device, dtype=torch.bool)
    }
