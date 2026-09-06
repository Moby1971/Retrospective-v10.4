"""Aangepast door Daisy van den Berg, originele code van Kai Lønning van paper Dynamic recurrent
Inference machines for accelerated MRI-guided radiotherapy of the liver."""

# =====================================================================================
# ENTRY POINT FOR DRIM RECONSTRUCTION
# =====================================================================================
#
# This script is the command line entry point that the MATLAB Retrospective app calls to
# reconstruct undersampled MRI data with a trained DRIM network (Dynamic Recurrent
# Inference Machine). It is a *reconstruction only* script: no training happens here, the
# network weights are read from a checkpoint that was trained elsewhere.
#
# The overall flow is:
#
#   1. Read the yaml configuration file given on the command line.
#   2. Build a DataLoader over the k-space data that MATLAB just wrote to disk
#      (data_sampler.MRData reads 'retroAItemp.mat').
#   3. Restore the trained network from a .ckpt checkpoint file.
#   4. Run one pass over the data with a Lightning Trainer. The network writes the
#      reconstructed images back to disk as 'retroAItemp_DRIM.mat', which MATLAB reads.
#
# Note that Lightning's terminology is used throughout: reconstruction is executed as a
# "test" pass (trainer.test), because from Lightning's point of view we are running a
# trained model over data without computing gradients. There is no test *metric* here.
#
# The heavy lifting lives in the sibling modules:
#   data_sampler.py  - reads the .mat file and serves one slice at a time
#   drim_network.py  - the LightningModule: iteration loop and saving of the result
#   drim_layers.py   - the actual network layers (conv blocks, recurrent units, gradient)
# =====================================================================================

import yaml
import torch
import lightning as L
from torch.utils.data import DataLoader, Dataset
import drim_network
from data_sampler import MRData
import sys
import os
import logging
import warnings

# -------------------------------------------------------------------------------------
# Console output clean-up.
#
# Lightning prints two promotional "tips" suggesting its paid cloud products. They are
# emitted through the logging module at INFO level, and there is no configuration flag to
# turn them off. Rather than raising the log level (which would also hide genuinely useful
# lines such as "GPU available: True (mps), used: True"), a filter is attached that drops
# only the records beginning with the light bulb marker.
# -------------------------------------------------------------------------------------
# Hide Lightning's promotional tips, but keep the informative lines (GPU available, ...)
logging.getLogger("lightning.pytorch.utilities.rank_zero").addFilter(
    lambda record: not record.getMessage().lstrip().startswith("\N{ELECTRIC LIGHT BULB} Tip:"))
# -------------------------------------------------------------------------------------
# Two warnings are filtered out because neither can be acted upon from this code base:
#   - "LeafSpec is deprecated" is raised inside Lightning itself, where it calls a torch
#     API that a newer torch version has deprecated. It disappears when Lightning updates.
#   - "Consider setting persistent_workers=True" only helps when a DataLoader is iterated
#     over multiple epochs, so that its worker processes can be reused. Reconstruction
#     runs exactly one pass, so there is nothing to reuse them for.
# -------------------------------------------------------------------------------------
# Hide warnings we cannot act on: a deprecation inside lightning itself, and a suggestion that
# does not apply here (persistent_workers only helps across epochs, reconstruction runs one).
warnings.filterwarnings("ignore", message=".*LeafSpec.*")
warnings.filterwarnings("ignore", message=".*persistent_workers.*")

# -------------------------------------------------------------------------------------
# Command line arguments (all four are required, and are passed by the MATLAB app).
# The original Dutch comments are kept below; in English they read:
#
#   argv[1] = directory holding the model checkpoint, e.g. .../functions/drim/m2/
#   argv[2] = directory holding the input data, i.e. where retroAItemp.mat was written
#   argv[3] = directory to save the reconstruction into
#   argv[4] = name (full path) of the yaml configuration file
#
# argv[2] and argv[3] are usually the same temporary folder. Note that these are read
# straight from sys.argv deep inside the functions below rather than being passed as
# parameters, so the module can only be driven from the command line, not imported and
# called with arbitrary paths.
# -------------------------------------------------------------------------------------
# argv[1] = directory waar model staat
# argv[2] = directory waar data staat
# argv[3] = directory om recon op te slaan
# argv[4] = naam yaml file


def parse_kernel(kernel_str):
    """Convert kernel string into a list format."""
    # The yaml file stores the convolution kernels as one compact string, for example
    # "333 None None". Each whitespace separated token describes one convolution layer:
    #
    #   "333"  -> a 3x3x3 kernel, given as the list [3, 3, 3] over (time, depth, width)
    #   "None" -> no convolution layer in that position
    #
    # Because each digit is read individually, only single digit kernel sizes can be
    # expressed with this notation (a kernel of 10 could not be written down).
    # The result is the list-of-lists that the network constructor expects, e.g.
    # [[3, 3, 3], None, None].
    return [None if k == 'None' else [int(x) for x in k] for k in kernel_str.split()]


def resolve_num_workers(config, n_samples):
    """Number of dataloader workers: never more than cores, never more than samples."""
    # The yaml value is treated as an upper bound rather than an exact count, so that a
    # single configuration file can be shared across machines of different sizes. PyTorch
    # itself never checks the core count: asking for more workers than there are cores
    # simply oversubscribes the CPU and slows everything down, so the value is clamped.
    requested = int(config['default']['num-workers'])
    # os.cpu_count() can return None in some virtualised environments; fall back to 1.
    available = os.cpu_count() or 1
    # The dataset serves one slice per index, so a worker beyond the number of slices has
    # nothing at all to fetch. That is not free: on macOS and Windows workers are started
    # with "spawn", so each one re-imports this module and pays the module level imports,
    # which is about 400 MB of torch and lightning per process. Single slice mouse data is
    # one sample, and the unclamped value would start twelve processes to serve it, none
    # of which does any work. The whole cost is paid again on every call, and the app calls
    # this script once per coil per dynamic.
    #
    # The data itself is not what the workers were costing: MRData puts its tensors in
    # shared memory precisely so that spawning does not copy them. It is the interpreter
    # and the imports that are per process and cannot be shared.
    return max(0, min(requested, available, int(n_samples)))


def resolve_precision(device):
    """Trainer precision for a given accelerator setting.

    "16-mixed" is what we want everywhere: on cuda it is the point of the exercise, and on
    cpu it is harmless because lightning substitutes bfloat16, fp16 autocast being
    unsupported there.

    mps is the exception, and only on older torch. Up to roughly torch 2.5 the mps autocast
    implementation cast the weights of the first Conv3d but not of the ones that followed,
    so the second convolution in any Sequential was handed a half input while its own bias
    was still float and raised

        Input type (c10::Half) and bias type (float) should be the same

    The network is built almost entirely from stacked Conv3d layers, so that made mixed
    precision unusable there. It is fixed in current torch (verified on 2.13), and falling
    back to fp32 anyway is expensive rather than merely cautious: on an Apple machine the
    GPU shares its memory with the rest of the system, so doubling the activations of a
    seven iteration RIM over a volume drives the whole machine into swap.

    So the bug is probed for rather than assumed. One tiny two-layer forward pass on the
    device answers the question in a few milliseconds, against a reconstruction measured in
    minutes, and the answer is right on both old and new torch. 'auto' is resolved here
    rather than left to lightning, because on an Apple machine lightning picks mps and this
    function has to know that is where it will land.
    """
    if device == 'mps' or (device == 'auto' and not torch.cuda.is_available()
                           and torch.backends.mps.is_available()):
        try:
            probe = torch.nn.Sequential(torch.nn.Conv3d(2, 4, 3),
                                        torch.nn.Conv3d(4, 4, 3)).to('mps')
            with torch.autocast('mps', dtype=torch.float16):
                probe(torch.randn(1, 2, 6, 8, 8, device='mps'))
        except Exception as error:
            # Anything at all going wrong here means fp32, including a torch that cannot
            # autocast on mps in some way this probe did not anticipate.
            print(f"mps half precision unusable ({error}), falling back to fp32")
            return "32-true"
    return "16-mixed"


# How much memory one reconstruction needs, and how much this machine has
#
# The network is a fixed architecture -- 128 features, seven iterations, batch of one --
# so its peak is a fixed multiple of one feature map, and a feature map is
#
#     nfeature * frames * height * width * 4 bytes
#
# Measured on this Mac over four sizes, in a fresh process each time so the allocator
# cache did not carry over, the peak is linear in that unit with a fixed offset:
#
#     unit  96 MiB -> 3.79 GiB      unit 160 MiB -> 5.31 GiB
#     unit 240 MiB -> 7.34 GiB      unit 512 MiB -> 14.66 GiB (reported by a user's OOM)
#
# A least squares fit to the first three predicts the fourth, on a different machine,
# within 4 %. The constants below are that fit. They are specific to this network: change
# nfeature, the iteration count or the kernels and they have to be measured again.
FEATURE_MAP_MULTIPLE = 25.5      # peak, in feature maps
FIXED_OVERHEAD_BYTES = 1.4 * 1024 ** 3   # weights, workspace and allocator slack

# What fraction of the device budget a reconstruction may plan to use.
#
# Not 1.0, and not close to it. The figure the estimate produces is the network's own
# allocations; the OOM that prompted this also reported 3.55 GiB of "other allocations"
# beside them, and the shared pool fragments, so the run that failed was refused a 512 MiB
# block while the reported numbers still summed to less than the limit. Two thirds leaves
# room for both and still admits everything that has ever worked here.
USABLE_FRACTION = 0.66

# The fewest cardiac frames the network can be given.
#
# The temporal convolutions are dilated, and the dilation doubles along the block: 1, 1, 2,
# 4. At dilation 4 a kernel of 3 needs four frames of wrap-around padding on each side, and
# CyclicPad1d builds that padding by SLICING -- input[:, :, -n:] -- which silently returns
# fewer than n frames when the axis is shorter than n. So below four frames the input is
# quietly under-padded and the convolution fails on a shape mismatch several layers later,
# with a message naming a kernel size the yaml never mentions.
#
# Measured rather than derived: 1, 2 and 3 frames each fail, 4 and everything above pass.
# The spatial axes have no such minimum, since they are padded by replication; 1 x 1 runs.
MIN_FRAMES = 4

# What this script writes into the log when the reconstruction cannot be attempted, as
# opposed to attempted and failed. The app looks for this line and reports the sentence
# after it on its own, instead of the tail of a traceback: not being able to run is a fact
# about the data and the machine, not a fault, and it should not read like a crash.
#
# retro.python.reportCannotReconstruct holds the same string on the MATLAB side.
CANNOT_RECONSTRUCT = 'RETRO-CANNOT-RECONSTRUCT'


class CannotReconstruct(Exception):
    """The reconstruction cannot be attempted, with the reason in the message."""


def estimate_peak_bytes(frames, height, width, nfeature=128):
    """Peak device memory one slice of this network needs, in bytes."""
    unit = nfeature * frames * height * width * 4
    return FEATURE_MAP_MULTIPLE * unit + FIXED_OVERHEAD_BYTES


def resolve_device(device):
    """The accelerator that will actually be used, with 'auto' resolved."""
    if device != 'auto':
        return device
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def device_budget_bytes(device):
    """How much memory the accelerator will let this process have, in bytes.

    Returns 0 when the answer is not knowable, which the caller reads as "do not judge".

    mps reports the same number the out of memory message calls "max allowed", and it is
    machine specific: 11.84 GiB on a 16 GB Mac against 20.13 GiB on the machine whose
    failure is quoted above, so it has to be asked on the machine that will run. cuda is
    asked for FREE memory rather than total, since the display and every other process
    have already taken their share.
    """
    try:
        if device == 'cuda' and torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            return int(free)
        if device == 'mps' and torch.backends.mps.is_available():
            return int(torch.mps.recommended_max_memory())
    except Exception as error:
        print(f"Could not read the {device} memory budget ({error}), continuing without the check")
        return 0

    # cpu: the limit is system memory, which psutil knows and the standard library does
    # not on every platform. Absent it, decline to judge rather than guess.
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        return 0


def sample_dimensions(dataset):
    """(frames, height, width) of one sample, as the network will see it."""
    # Taken from the data rather than from the yaml, because volume=False cuts the same
    # array along a different axis and the sample is then a different shape.
    estimate = dataset[0]['estimate']
    shape = tuple(estimate.shape)
    if len(shape) != 3:
        return None
    return shape


def check_memory(config, dataset):
    """Refuse the reconstruction if it will not fit on the accelerator.

    The check is here, and not in the app, because only this process knows both the shape
    the network will actually see and what the accelerator will allow. The app can only
    guess at the first and cannot see the second at all.

    Refusing rather than moving to the processor is deliberate. The processor finishes, but
    for a volume of this size it takes long enough that a user waiting on it is worse served
    than one told immediately to reconstruct fewer frames or to use a larger machine. The
    exception is a machine with no accelerator at all: there the processor is not a fallback,
    it is the only thing there is, so the estimate is reported as a warning and the
    reconstruction proceeds.
    """
    device = resolve_device(config['default']['device'])
    dims = sample_dimensions(dataset)

    if dims is None:
        return device

    frames, height, width = dims

    if frames < MIN_FRAMES:
        raise CannotReconstruct(
            f"The deep learning reconstruction needs at least {MIN_FRAMES} cardiac frames "
            f"and this one has {frames}. Its temporal filters are dilated by up to "
            f"{MIN_FRAMES}, so a shorter cycle cannot be padded. Reconstruct more frames, "
            f"or use one of the other reconstruction methods.")

    needed = estimate_peak_bytes(frames, height, width)
    budget = device_budget_bytes(device)
    gib = 1024 ** 3

    # Said whenever the processor is what will run, not only when memory is short. It is
    # the slow path however it was arrived at, and a user watching a progress gauge that
    # has barely moved is owed the reason.
    if device == 'cpu':
        print("WARNING: no accelerator in use, reconstructing on the processor. "
              "This takes considerably longer than a GPU run.")

    print(f"Reconstructing {frames} frames of {height} x {width}: "
          f"estimated peak {needed / gib:.1f} GiB")

    if budget <= 0:
        print("Device memory budget unknown, proceeding")
        return device

    usable = USABLE_FRACTION * budget
    print(f"Device budget {budget / gib:.1f} GiB, of which {usable / gib:.1f} GiB is planned for")

    if needed <= usable:
        return device

    if device == 'cpu':
        # No accelerator on this machine, so there is nothing to refuse in favour of.
        print(f"WARNING: this reconstruction needs about {needed / gib:.1f} GiB and about "
              f"{budget / gib:.1f} GiB is free. It may fail or drive the machine into swap.")
        return device

    # The message is the last line of the traceback, which is what the app shows, so it
    # carries the numbers and what to do about them rather than only the fact.
    fits = max(1, int(usable_frames(usable, height, width)))
    raise CannotReconstruct(
        f"Not enough memory for the deep learning reconstruction. {frames} frames at "
        f"{height} x {width} need about {needed / gib:.1f} GB, and this machine offers "
        f"about {budget / gib:.1f} GB to {device}, of which {usable / gib:.1f} GB can be "
        f"planned for. About {fits} frames would fit here. The matrix is not a lever: "
        f"data_sampler refills every scan onto {height} x {width} whatever was acquired, so "
        f"the frame count is what decides this, along with how much memory the machine has.")


def usable_frames(usable_bytes, height, width, nfeature=128):
    """How many frames would fit in a given budget, at this matrix size."""
    per_frame = FEATURE_MAP_MULTIPLE * nfeature * height * width * 4
    return (usable_bytes - FIXED_OVERHEAD_BYTES) / per_frame


def load_test_data(config):
    """Prepare data in dataloader"""
    # Where MATLAB wrote retroAItemp.mat (argv[2], see the header comment above).
    data_directory = sys.argv[2]
    # MRData reads and pre-processes the whole file up front in its constructor, so this
    # single line performs the zero filling, the inverse Fourier transform and the
    # normalisation. Afterwards the dataset serves one slice per index.
    dataset = MRData(data_directory, config['default']['volume'])
    num_workers = resolve_num_workers(config, len(dataset))
    # Printed so that it is visible in the log which value a given machine actually used,
    # rather than only what the yaml requested, and which of the three limits bound it.
    print(f"Dataloader workers: {num_workers} (requested {config['default']['num-workers']}, "
          f"{os.cpu_count()} cores available, {len(dataset)} sample(s) to serve)")
    # drop_last=False matters here: every slice must be reconstructed, so a final partial
    # batch may not be silently discarded the way it often is during training.
    test_loader = DataLoader(dataset, batch_size= config['default']['batch-size'], drop_last=False,
        num_workers=num_workers)
    return test_loader


def test_model(config):
    """Initialize and test the RIM model."""
    # Load data
    test_dataloader = load_test_data(config)

    # Load DRIM model
    kernel = parse_kernel(config['default']['kernel'])
    # argv[1] is the folder of the selected model (m1 ... m6), one folder per
    # undersampling factor. Each folder holds a single .ckpt file.
    checkpoint_path = sys.argv[1]
    # os.listdir returns entries in an arbitrary, filesystem dependent order, so the
    # listing is filtered by extension and sorted. Without this, a stray file synced into
    # the folder (.DS_Store on macOS, Thumbs.db or desktop.ini on Windows, a Dropbox
    # conflict copy) could be picked up and handed to the checkpoint loader.
    # Only consider checkpoint files, so that stray files (.DS_Store, Thumbs.db, ...) are ignored
    checkpoint_files = sorted(f for f in os.listdir(checkpoint_path) if f.endswith('.ckpt'))
    # Fail early and clearly. Without this check the error would surface much later and
    # far less legibly, from inside torch's unpickling machinery.
    if not checkpoint_files:
        raise FileNotFoundError(f"No .ckpt file found in {checkpoint_path}")
    # os.path.join rather than string concatenation, so that the caller does not have to
    # supply a trailing separator and so that Windows backslashes are handled correctly.
    checkpoint = os.path.join(checkpoint_path, checkpoint_files[0])
    # Number of RIM iterations: how often the network refines its own estimate. More
    # iterations generally means a better reconstruction at a proportionally higher cost.
    niterations = config['default']['niteration']
    # The architecture arguments must match those the checkpoint was trained with,
    # otherwise the stored weights will not fit the layers being created. nfeature=128
    # is therefore hard coded rather than read from the yaml file.
    model = drim_network.RecurrentInferenceMachine.load_from_checkpoint(checkpoint_path=checkpoint, nfeature=128, kernel=kernel,
                                                               temporal_rnn=True, niterations=niterations)
    # Attach the configuration to the model so the LightningModule can reach it later.
    model.config = config

    # Start reconstruction
    # accelerator comes from the yaml 'device' key: 'cuda', 'mps', 'cpu', or 'auto' to let
    # Lightning select the best device present on this machine.
    # "16-mixed" runs the convolutions in half precision for speed and memory; on CPU
    # Lightning silently substitutes bfloat16, because fp16 autocast is unsupported. On mps
    # mixed precision is broken for stacked Conv3d, so resolve_precision falls back to fp32
    # there; see the note in that function.
    # logger=False disables Lightning's metric logging, which would only write empty files.
    # The device asked for in the yaml, unless the reconstruction will not fit on it. The
    # dataset is reached through the loader rather than passed separately, so the shape
    # judged is the one the loader will actually serve.
    try:
        device = check_memory(config, test_dataloader.dataset)
    except CannotReconstruct as why:
        # Printed and returned, not raised. The app reads this line and says only the
        # sentence; raising here would give it a traceback to show instead.
        print(f"{CANNOT_RECONSTRUCT}: {why}")
        return
    precision = resolve_precision(device)
    print(f"Device: {device}, precision: {precision}")
    trainer = L.Trainer(accelerator=device, precision=precision, logger=False)
    print("Start reconstructing...")
    # Runs one pass over every slice. The reconstruction is written to disk from inside
    # the model's on_test_epoch_end hook, so nothing needs to be returned here.
    trainer.test(model, test_dataloader)
    return


# Only run when executed as a script. This guard is not merely stylistic: on Windows and
# macOS the DataLoader starts its worker processes with the "spawn" method, which re-imports
# this module in every worker. Without the guard, each worker would start a full
# reconstruction of its own.
if __name__ == "__main__":
    config_file = sys.argv[4]
    # Load hyperparameters in yaml file
    # safe_load (rather than load) parses only plain yaml types and will not instantiate
    # arbitrary Python objects named in the file.
    with open(config_file, 'r') as file:
        config_params = yaml.safe_load(file)
    test_model(config_params)
