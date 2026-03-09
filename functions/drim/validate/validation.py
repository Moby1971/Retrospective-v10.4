__author__ = 'Kai Lønning'

import os
import configparser
import pickle

import torch
import torch.utils.data as data

import numpy as np
import pandas as pd
from interval import Interval

from validate.val_utils import *
from validate.visualize import plot_images
from models.initialize import load_model
from validate.process_data import *
from data.data_sampler_zp_sc import MRData

import logging
logger = logging.getLogger(__name__)

def validate_checkpoints(config, train_config, dataloader, csvname, writer):
    """Loop through checkpoints and validate the model."""
    subject = os.path.basename(config['data-dir'])
    min_chk, max_chk = get_checkpoint_range(config, train_config)
    
    for checkpoint in map(str, range(min_chk, max_chk, int(config['checkpoint-freq']) * int(train_config['checkpoint-freq']))):
        print(checkpoint)
        network, initrim, gradrim = load_model(config, train_config, checkpoint)
        for random_idx in get_random_bin_idxs(config, dataloader):
            validate_dataset(config, train_config, subject, dataloader, checkpoint, network, initrim, gradrim, csvname, writer, random_bin_idxs=random_idx)

def validate_model(config):
    """Main validation function."""
    train_config = configparser.ConfigParser()
    train_config.read(os.path.join(config['train-dir'], 'config.ini'))
    train_config = train_config['train']
    
    dataset = MRData(config['data-dir'], config['undersampled_data_dir'])
    dataloader = data.DataLoader(dataset, batch_size=int(config['batch-size']), drop_last=False, sampler=data.SequentialSampler(dataset), num_workers=int(config['num-workers']))
    
    csvname, writer = setup_logging(config)
    validate_checkpoints(config, train_config, dataloader, csvname, writer)

def validate_dataset(
    config, train_config, subject, dataloader, checkpoint,
    network, initrim, gradrim, csvname, writer, random_bin_idxs=None):
    
    metrics = define_metrics(config)
    recons, targets, viz, trgts = process_batches(config, train_config, dataloader, initrim, gradrim, network, random_bin_idxs)

    viz = torch.cat(viz, 0).moveaxis(0, 3)
    plot_images(config, '', viz, 'Original data', 'reconstruction', 0, writer )
    trgts = torch.cat(trgts, 0).moveaxis(0, 3)
    plot_images(config, '', trgts, 'Original data', 'target', 0, writer, 'target-gif' )

    target = torch.cat(targets, 0).moveaxis(0, 3) # [T, D, H, W, 2]
    reconstruction = torch.cat(recons, 0).moveaxis(0, 3) # [T, D, H, W, 2]

    # if not config.getboolean('sorted'):
    #     tarslices = []
    #     recslices = []
    #     dynslice = dataloader.dataset.dynamic_slice
    #     iterator = zip(
    #         target.moveaxis(1, 0),
    #         reconstruction.moveaxis(1, 0),
    #         dataloader.dataset.data[''][
    #             f'respiration-{dynslice.start}-{dynslice.stop - 1}'
    #             f'dynamics-{int(config["nbin"])}bins'][0]
    #     )
    #     for tarslice, recslice, slicebreath in iterator: # [dyn, H, W, 2]
    #         tardynamics = []
    #         recdynamics = []
    #         for bin in range(int(config['nbin'])):
    #             if not slicebreath[bin]:
    #                 tardynamics.append(torch.zeros_like(tarslice[0]))
    #                 recdynamics.append(torch.zeros_like(recslice[0]))
    #             else:
    #                 tardynamics.append(tarslice[
    #                     slicebreath[bin][
    #                         int((len(slicebreath[bin]) - 1)/2)][0]])
    #                 recdynamics.append(recslice[
    #                     slicebreath[bin][
    #                         int((len(slicebreath[bin]) - 1)/2)][0]])
    #         tarslices.append(torch.stack(tardynamics)) # [T, H, W, 2]
    #         recslices.append(torch.stack(recdynamics))
    #     target = torch.stack(tarslices, 1)
    #     reconstruction = torch.stack(recslices, 1) # [T, D, H, W, 2]

    if not random_bin_idxs is None:
        subject = f'{subject}-random-bin'
    if not hasattr(dataloader.dataset, 'dynamic_slice'):
        dyns = 120
    else:
        dynrange = dataloader.dataset.dynamic_slice
        dynrange = f'{dynrange.start}-{dynrange.stop - 1}'
        subject = f'{subject}-dynamics{dynrange}'
        dyns = int(config['ndynamic'])

    plot_images(
        config, torch.view_as_complex(target),
        torch.view_as_complex(reconstruction), subject,
        'reconstruction', checkpoint, writer) #dataloader.dataset.maskname
    
    if config['test']:
        gap_mask = torch.zeros(reconstruction.shape[:2], dtype=bool)
        gap_neighbor_mask = torch.zeros(reconstruction.shape[:2], dtype=bool)
        rec_gaps = [] # Only gaps, unaltered reconstructions
        tar_gaps = [] # Only gaps, selected from best fitting slice from all dynamics
        rec_neighbors = [] # Only unaltered reconstructions of gap neibors
        tar_neighbors = [] # Only gap neighbors
        rec_gap_removed = [] # Only non-gaps, stacked in slice-bin order
        tar_gap_removed = [] # Only non-gaps, stacked in slice-bin order
        rec_gap_removed_2 = [] # Only non-gaps, stacked in bin-slice order
        tar_gap_removed_2 = [] # Only non-gaps, stacked in bin-slice order
        rec_gaps_neighbors_removed = []
        tar_gaps_neighbors_removed = []
        rec_interpolated_gaps = [] # Only gaps, interpolated from neighbor reconstructions
        rec_interpolated = torch.zeros_like(
            reconstruction).copy_(reconstruction) # Full volume with interpolations
        tar_filled_gaps = torch.zeros_like(target).copy_(target) # Full volume replaced where possible, otherwise interpolated

        data = dataloader.dataset.data['']
        breathing_data, bin_edges, neighbor_idx = data[
            f'respiration-{dynrange}dynamics-{int(config["nbin"])}bins'
        ]
        breathing_data_total, _, _ = data[
            f'respiration-{int(config["nbin"])}bins'
        ]
        
        # Make masks locating temporal gaps and their neighboring slices
        gap_idx = [[], []]
        for slice_position, breathing_bins in enumerate(breathing_data):
            for bin in range(int(config['nbin'])):
                if breathing_bins[bin]:
                    rec_gap_removed.append(reconstruction[bin, slice_position])
                    tar_gap_removed.append(target[bin, slice_position])
                else:
                    gap_idx[0].append(bin)
                    gap_idx[1].append(slice_position)
                    gap_mask[bin, slice_position] = True
                    gap_neighbor_mask[bin, slice_position] = True
                    ubin = (bin - 1) % int(config['nbin'])
                    dbin = (bin + 1) % int(config['nbin'])
                    gap_neighbor_mask[dbin, slice_position] = True
                    gap_neighbor_mask[ubin, slice_position] = True
                    if slice_position:
                        gap_neighbor_mask[bin, slice_position - 1] = True
                    if  slice_position != reconstruction.shape[1] - 1:
                        gap_neighbor_mask[bin, slice_position + 1] = True
        for bin in range(int(config['nbin'])):
            for slice_position, breathing_bins in enumerate(breathing_data):
                if breathing_bins[bin]:
                    rec_gap_removed_2.append(
                        reconstruction[bin, slice_position])
                    tar_gap_removed_2.append(target[bin, slice_position])

        bin_dict = {}
        for bin_nr in range(1, int(config['nbin']) // 2):
            bin_dict[bin_nr] = lambda x: (x[bin_nr], x[bin_nr + 1])
            bin_dict[int(config['nbin']) - bin_nr
                ] = lambda x: (x[bin_nr], x[bin_nr + 1])
        bin_dict[0] = lambda x: (x[0], x[1])
        bin_dict[int(config['nbin']) // 2] = lambda x: (
            x[int(config['nbin']) // 2], x[int(config['nbin']) // 2 + 1])

        def replace_nearest_bin_target(bin, slice_position):
            edges = bin_dict[bin](bin_edges)
            bin_center = sum(edges) / 2.
            edges = Interval(edges[0], edges[1])
            candidates = [[], []]
            for b in ( 
                (bin - 2) % int(config['nbin']), 
                (bin - 1) % int(config['nbin']),
                bin,
                (bin + 1) % int(config['nbin']),
                (bin + 2) % int(config['nbin'])
            ):
                if breathing_data_total[slice_position][b]:
                    for dynamic, signal in breathing_data_total[
                        slice_position][b]:
                        if signal in edges:
                            candidates[0].append(dynamic)
                            candidates[1].append(
                                np.abs(signal - bin_center))
            if candidates[0]:
                closest = candidates[0][np.argmin(candidates[1])]
                replacement_slice = torch.view_as_real(
                    torch.tensor(
                        dataloader.dataset.data[
                        '']['imspace'][slice_position, closest]
                    ).moveaxis(1, 0)
                )
                return torch.tensor(replacement_slice).to(target)

        def interpolate(reconstruction, neighbors):
            interpol = torch.zeros_like(reconstruction[0, 0])
            if neighbors:
                for bin, slice_position in neighbors:
                    interpol += reconstruction[bin, slice_position]
                return interpol / len(neighbors)

        
        for s in range(len(breathing_data)):
            for b in range(int(config['nbin'])):
                if not gap_neighbor_mask[b, s]:
                    rec_gaps_neighbors_removed.append(reconstruction[b, s])
                    tar_gaps_neighbors_removed.append(target[b, s])
                elif not gap_mask[b, s]:
                    rec_neighbors.append(reconstruction[b, s])
                    tar_neighbors.append(target[b, s])
                else:
                    interpolation = interpolate(
                        reconstruction, neighbor_idx[(b, s)])
                    if interpolation is not None:
                        rec_interpolated[b, s] = interpolation
                    replacement_slice = replace_nearest_bin_target(b, s)
                    if replacement_slice is not None:
                        rec_gaps.append(reconstruction[b, s])
                        tar_gaps.append(replacement_slice)
                        rec_interpolated_gaps.append(rec_interpolated[b, s])
                        tar_filled_gaps[b, s] = replacement_slice
                    else:
                        tar_filled_gaps[b, s] = torch.zeros_like(target[0, 0])
                        rec_interpolated[b, s] = torch.zeros_like(
                            reconstruction[0, 0])
        
        n_gap = len(gap_idx[0])
        rec_gap_removed = torch.stack(rec_gap_removed).unsqueeze(0)
        tar_gap_removed = torch.stack(tar_gap_removed).unsqueeze(0)
        rec_gap_removed_2 = torch.stack(rec_gap_removed_2).unsqueeze(0)
        tar_gap_removed_2 = torch.stack(tar_gap_removed_2).unsqueeze(0)
        rec_gaps_neighbors_removed = torch.stack(
            rec_gaps_neighbors_removed).unsqueeze(0)
        tar_gaps_neighbors_removed = torch.stack(
            tar_gaps_neighbors_removed).unsqueeze(0)
        to_evaluate = []
        if n_gap > 0:
            """if rec_gaps:
                rec_gaps = torch.stack(rec_gaps).unsqueeze(0)
                tar_gaps = torch.stack(tar_gaps).unsqueeze(0)
                rec_neighbors = torch.stack(rec_neighbors).unsqueeze(0)
                tar_neighbors = torch.stack(tar_neighbors).unsqueeze(0)
                to_evaluate.extend((
                    ('gaps only - left as output', rec_gaps, tar_gaps),
                    ))"""
            if rec_interpolated_gaps and (
                isinstance(tar_gaps, list) and tar_gaps):
                rec_interpolated_gaps = torch.stack(
                    rec_interpolated_gaps).unsqueeze(0)
                to_evaluate.extend((
                    ('whole volume - gaps interpolated',
                     rec_interpolated, tar_filled_gaps),))
                    #('gaps only - interpolated', 
                    # rec_interpolated_gaps, tar_gaps),
                    #('gap neighbors only', rec_neighbors, tar_neighbors)))
        to_evaluate.extend([
            (
                'whole volume - gaps left as output', 
                reconstruction, tar_filled_gaps
            ),
            (
                'gaps removed - stacked in slice-bin order',
                rec_gap_removed, tar_gap_removed
            ),
            (
                'gaps removed - stacked in bin-slice order',
                rec_gap_removed_2, tar_gap_removed_2
            ),
            #(
            #    'gaps and gap neighbors removed',
            #    rec_gaps_neighbors_removed, tar_gaps_neighbors_removed
            #)
        ])
    else:
        to_evaluate = [('N/A', reconstruction, target)]
        n_gap = 0

    is_sorted = "4d-sorted" if config.getboolean("sorted") else "unsorted"
    for metric in config['metrics'].split():    
        for label, r, t in to_evaluate:
            if config.getboolean('perslicescore'):
                for count, (r_, t_) in enumerate(zip(
                    r.reshape(-1, *r.shape[-2:]), t.reshape(-1, *t.shape[-2:]))
                ):
                    score = metrics[metric](t_, r_)
                    df = pd.DataFrame(
                        data={
                            'T': int(config['niteration']),
                            'Mask': dataloader.dataset.maskname, 
                            'Metric': metric, 
                            'Score': score, 
                            'Data': subject,
                            'Number of gaps': n_gap,
                            'Temporal gaps': label,
                            'Dynamics': dyns,
                            '4d sorted': config.getboolean('sorted'),
                            'Number of bins': int(config['nbin']),
                            'Slice': count % t.shape[1],
                            'Bin': count // t.shape[1],
                            },
                        index=[int(checkpoint)])
                    df.to_csv(csvname, header=False, mode='a')
            else:
                score = metrics[metric](t, r)
                logger.info(
                    f"Checkpoint {checkpoint} {subject} "
                    f"mask {metric} with {n_gap} " #{dataloader.dataset.maskname}
                    f"gaps and {int(config['nbin'])} bins; {label}: {score}")
                df = pd.DataFrame(
                    data={
                        'T': int(config['niteration']),
                        'Mask': 'mask', #dataloader.dataset.maskname
                        'Metric': metric, 
                        'Score': score, 
                        'Data': subject,
                        'Number of gaps': n_gap,
                        'Temporal gaps': label,
                        'Dynamics': dyns,
                        '4d sorted': config.getboolean('sorted'),
                        'Number of bins': int(config['nbin'])},
                    index=[int(checkpoint)]
                )
                df.to_csv(csvname, header=False, mode='a')
                writer.add_scalar(
                    f'Validation/{subject}/{is_sorted}/'
                    f'mask/{n_gap}_' #{dataloader.dataset.maskname}
                    f'{"_".join(label.split())}/{int(config["nbin"])}/{metric}',
                    score,
                    int(checkpoint)
                )