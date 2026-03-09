
import torch
import torchvision
import numpy as np
 
def plot_images(
    config, target, recon, dataset, maskname, checkpoint, writer, other_gif_name = None):
    
    def transform_image_pair(t, x):
        minval = min(t.min().item(), x.min().item(), 0)
        shifted_t = t - minval
        shifted_x = x - minval
        shifted_max = max(shifted_t.max().item(), shifted_x.max().item())
        return shifted_t / shifted_max, shifted_x / shifted_max
    def make_false_color_components(t, x):
        transformed_t, transformed_x = transform_image_pair(t, x)
        return torch.stack((transformed_t, transformed_x, transformed_t), 0)
    def normalize(x):
        return x / x.max()

    nrow = 5 if not config['volume-slices'] else len(
        config['volume-slices'].split())
    if not config['volume-slices']:
        vol_slices = np.linspace(0, recon.size(1) - 1, num=nrow).astype(int)
    else:
        vol_slices = [int(vs) for vs in config['volume-slices'].split()]
    if not config['bin-slices']:
        bin_slices = np.linspace(0, 9, num=nrow).astype(int)
    else:
        bin_slices = [int(bs) for bs in config['bin-slices'].split()]
        
    #targets_width_by_height = [torch.rot90(
    #    target[i, j].abs(), 3, [0, 1]) for i, j in zip(bin_slices, vol_slices)]
    recons_width_by_height = [torch.rot90(
        recon[i, j].abs(), 3, [0, 1]) for i, j in zip(bin_slices, vol_slices)]

    #writer.add_image(f'Width-by-Height-errors/{dataset}/{maskname}', torchvision.utils.make_grid(torch.stack([make_false_color_components(t, r) for t, r in zip(targets_width_by_height, recons_width_by_height)], 0), nrow=nrow), int(checkpoint))

    width_by_height = torchvision.utils.make_grid(torch.stack([normalize(r) for r in recons_width_by_height], 0).unsqueeze(1), nrow=nrow) #[normalize(t) for t in targets_width_by_height] +
    writer.add_image(f'Width-by-Height/{dataset}/{maskname}', width_by_height, int(checkpoint))
    #recon = torch.stack([torch.rot90(recon[:, i, :, :], 3, [1, 2]) for i in range(recon.size(1))], dim=1)
    recon = torch.abs(recon)
    recon = (recon - recon.min()) / (recon.max() - recon.min())
    recon = recon.unsqueeze(0)
    recon = recon.repeat(1, 1, 3, 1, 1)

    gif_name = 'gif'
    if other_gif_name != None:
        gif_name = other_gif_name
    writer.add_video(gif_name, recon, int(checkpoint), fps = 30)
    writer.flush()
    #writer.close()