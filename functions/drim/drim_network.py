__author__ = 'Daisy van den Berg'

# =====================================================================================
# THE DRIM MODEL: ITERATION LOOP AND OUTPUT
# =====================================================================================
#
# This module holds the LightningModule that drives the reconstruction. The actual layers
# live in drim_layers.py; what is added here is the iterative scheme that makes this a
# Recurrent Inference Machine rather than a plain feed forward network.
#
# The idea behind a RIM is that reconstruction is solved iteratively, like a classical
# optimiser, but with the update step learned instead of derived. Each iteration:
#
#   1. Compute the data consistency gradient: transform the current image estimate to
#      k-space, compare it against the measured samples, transform the difference back to
#      image space (this is self.gradrim, i.e. GradRim in drim_layers.py).
#   2. Feed the current estimate together with that gradient into the recurrent block.
#   3. Add the block's output to the estimate as an update step.
#
# The recurrent hidden state carries information between iterations, so the network can
# learn a step size and direction that depend on what it has already seen. The hidden
# state is not initialised to zero but produced by a small learned network (InitRim).
#
# Complex numbers are handled by keeping the real and imaginary parts as an extra trailing
# axis of size 2 (torch.view_as_real), because convolution layers cannot operate on
# complex tensors directly. torch.view_as_complex converts back where a Fourier transform
# is needed. The two functions are views, not copies, so this is cheap.
#
# Lightning hook order during trainer.test():
#   on_test_epoch_start  -> once, clears the output buffer
#   test_step            -> once per slice, runs the iterations and stores the result
#   on_test_epoch_end    -> once, assembles all slices and writes the .mat file
# =====================================================================================

import torch
import logging
import lightning as L
import numpy as np
from drim_layers import InitRim, GradRim, DynamicRecurrentInferenceBlock
from scipy.io import savemat
import sys
import os


class RecurrentInferenceMachine(L.LightningModule):
    """
    A Recurrent Inference Machine (RIM) for processing spatiotemporal data.

    This class implements a neural network model with convolutional blocks and
    recurrent units. It is designed for processing spatiotemporal data, with
    3D convolutions (to handle the spatial dimensions) and recurrent layers
    (to model temporal dependencies). The architecture consists of input,
    intermediate, and output convolutional layers, and two recurrent units for
    state propagation across time.

    Args:
        nfeature (int): The number of features (channels) in the intermediate layers.
        kernel (tuple of 3 tuples): Kernel sizes for the 3D convolution layers. Each tuple represents
                                     the kernel size along the (depth, height, width) dimensions.
                                     A `None` entry means no convolutional layer in that direction.
        temporal_rnn (bool): If True, uses temporal recurrent units for modeling temporal dependencies.
        mute (bool): If False, logs the initialization process. If True, disables logging.

    Attributes:
        logger (logging.Logger): Logger to record the model's initialization process.
        nfeature (int): The number of features in the intermediate layers.
        kernel (tuple): The kernel sizes for the 3D convolution layers.
        conv_in (nn.Sequential): Convolutional block for the input.
        recurrent1 (LatticeRecurrentUnit): First recurrent unit for temporal modeling.
        conv_between (nn.Sequential): Convolutional block between recurrent units.
        recurrent2 (LatticeRecurrentUnit): Second recurrent unit for temporal modeling.
        conv_out (nn.Sequential): Convolutional block for the output.
    """

    def __init__(
            self, nfeature=128, kernel=((1, 3, 3), (3, 1, 1), None),
            temporal_rnn=False, fourier_dim=(-2, -1), niterations=8):
        super().__init__()

        # Initialize initrim, Gradrim en rimblock
        # InitRim: learns the initial hidden state from the starting estimate, instead of
        # beginning from zeros. Its input has 2 channels (real and imaginary part) and it
        # produces two hidden states, one for each recurrent unit, hence 2 * [nfeature].
        self.initrim = InitRim(2, 2 * [nfeature], kernel)
        # GradRim: the data consistency gradient. It holds no learnable parameters, only
        # the Fourier transforms and the comparison against the measured k-space.
        # fourier_dim=(-2, -1) means the transform acts on the last two (spatial) axes.
        self.gradrim = GradRim([fourier_dim])
        # The learned update step: convolutions and two recurrent units.
        self.drim_block = DynamicRecurrentInferenceBlock(nfeature=nfeature, kernel=kernel,
                                                         temporal_rnn=temporal_rnn)
        # How many refinement iterations to run. Cast to int because it arrives from yaml,
        # where it may have been parsed as a string.
        self.niterations = int(niterations)

    def forward(self, x, hidden):
        """
        Forward pass through the Recurrent Inference Machine.

        Args:
            x (Tensor): Input tensor with shape [B, F, T, D, W], where:
                        B = batch size, F = features (channels), T = time steps,
                        D = depth, and W = width.
            hidden (tuple): Hidden states for the two recurrent units. Each hidden state has shape
                            [B, F, T, D, W].

        Returns:
            Tuple:
                - Output tensor with shape [B, T, D, W].
                - Updated hidden state tuple.
        """
        # A thin wrapper: one pass through the block is one RIM iteration's update step.
        x, hidden = self.drim_block(x, hidden)
        return x, hidden

    def on_test_epoch_start(self):
        # Buffer for the reconstructed slices, keyed by subject (i.e. by .mat filename).
        # Reset here rather than in __init__ so that a second run starts clean.
        self.test_outputs = {}
        # The normalisation factor the data sampler divided by, also keyed by subject, so
        # that it can be written out next to the reconstruction.
        self.test_scales = {}

    def test_step(self, batch):
        # Load data
        # view_as_real turns each complex tensor into a real one with a trailing axis of
        # size 2, which is what the convolution layers require. self.dtype follows the
        # Trainer's precision setting (fp32 with mixed precision autocasting inside).
        estimate = torch.view_as_real(batch['estimate']).to(self.dtype)
        measurements = torch.view_as_real(batch['measurements']).to(self.dtype)
        sense = torch.view_as_real(batch['sense']).to(self.dtype)
        # The mask selects which k-space points were actually measured, so it is used as a
        # boolean condition in the data consistency step.
        mask = batch['mask'].to(dtype=torch.bool)
        # Batch size is 1 during reconstruction, so element 0 is the only subject present.
        subject = batch['subject'][0]
        # Same value for every slice of a subject, so simply overwriting is fine.
        self.test_scales[subject] = float(batch['scale'][0])
        # Learned initial hidden state. moveaxis(-1, 1) moves the real/imaginary axis into
        # the channel position, giving the [B, F, T, D, W] layout the convolutions expect.
        hidden = self.initrim(estimate.moveaxis(-1, 1))

        # Forward iterations 8 time steps through RIM block
        # The core RIM loop. Note that the estimate is refined in place across iterations
        # while `hidden` carries the recurrent state along with it.
        for iteration in range(self.niterations):
            # How far the current estimate is from being consistent with the measured data,
            # expressed in image space.
            gradient = self.gradrim(estimate, measurements, sense, mask)
            # The network sees both the current estimate and that gradient: 2 channels for
            # the estimate plus 2 for the gradient makes the 4 input channels of conv_in.
            network_input = torch.cat((estimate.moveaxis(-1, 1), gradient), 1)
            estimate_step, hidden = self.forward(network_input, hidden)

            # Update the estimate
            # An additive update, exactly like a gradient descent step, except that both
            # the direction and the size of the step were produced by the network.
            # moveaxis(1, -1) puts the real/imaginary axis back at the end.
            estimate = estimate + estimate_step.moveaxis(1, -1)

        # Save lines of the same image together based on the subject name
        # Slices arrive one at a time and are collected per subject, to be concatenated in
        # on_test_epoch_end. Appending to a list is used rather than concatenating here,
        # because repeated concatenation would copy the whole array every time.
        if subject in self.test_outputs:
            list_estimate = self.test_outputs[subject]
            list_estimate.append(estimate)
            self.test_outputs[subject] = list_estimate
        else:
            self.test_outputs[subject] = [estimate]
        return

    def on_test_epoch_end(self):
        for scan in self.test_outputs:
            # Add slices together in one array
            # Stack the per slice results back into one array, move it to the CPU, read it
            # as complex again and take the magnitude: the phase is discarded, since the
            # app displays magnitude images.
            estimate = torch.abs(torch.view_as_complex(torch.cat(self.test_outputs[scan], dim=0).cpu()))
            # Transpose to dimensions (time, slices, y, x)
            # MATLAB expects the time axis first. The second np.abs is redundant, since
            # torch.abs was already applied above, but it is harmless.
            estimate = np.transpose(np.abs(estimate), [1, 0, 2, 3])
            # Save scan in matlabfile
            self.save_reconstruction_in_mat(estimate, scan)
        return

    def save_reconstruction_in_mat(self, estimate, subject):
        print("Saving subject...", subject)
        # The output filename is fixed by agreement with the MATLAB app, which looks for
        # exactly this name in the folder it passed as argv[3]. Note that the subject name
        # is not part of the filename, so reconstructing several subjects in one run would
        # have each overwrite the previous one.
        file_name = 'retroAItemp_DRIM'
        mat_file = os.path.join(sys.argv[3], file_name + '.mat')
        # 'aiData' is the variable name the app reads back from the .mat file.
        #
        # 'aiScale' is the factor the data was divided by before reconstruction. The image
        # in aiData is normalised, so multiplying by aiScale restores the original signal
        # amplitude. That matters when the app loops over coils and combines them with a
        # sum-of-squares: without rescaling, every coil arrives normalised to the same
        # peak and the sensitivity weighting that sum-of-squares relies on is lost.
        # Ignoring the field reproduces the previous behaviour exactly.
        savemat(mat_file, {'aiData': estimate,
                           'aiScale': self.test_scales.get(subject, 1.0)})
        return
