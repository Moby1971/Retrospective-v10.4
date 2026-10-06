"""Aangepast door Daisy van den Berg, originele code van Kai Lønning van paper Dynamic recurrent
Inference machines for accelerated MRI-guided radiotherapy of the liver."""

# =====================================================================================
# THE BUILDING BLOCKS OF THE DRIM NETWORK
# =====================================================================================
#
# This module contains every layer the DRIM is assembled from. drim_network.py wires them
# together and runs the iteration loop; nothing here knows about iterations.
#
# The components, in the order they appear below:
#
#   InitRim                        - learns the initial hidden state from the first estimate
#   get_num_params                 - counts parameters (diagnostics only)
#   GradRim                        - the data consistency gradient (no learnable weights)
#   set_conv_block                 - helper that assembles a padded 3D convolution block
#   DynamicRecurrentInferenceBlock - one learned update step: conv, RNN, conv, RNN, conv
#   CyclicPad1d                    - wrap-around padding along the time axis
#   LatticeRecurrentUnit           - the recurrent cell, a convolutional GRU variant
#
# Tensor layout used everywhere: [B, F, T, D, W]
#   B = batch, F = features/channels, T = time frames, D = depth (slices), W = width
# Complex data carries an extra trailing axis of size 2 holding (real, imaginary).
#
# Why 3D convolutions with unusual kernels: the data is 2D+time, and the kernel
# configuration (for example "333 None None" from the yaml, meaning a single 3x3x3 layer)
# controls whether a layer mixes information across time, across space, or both. A kernel
# of (1, 3, 3) is purely spatial; one of (3, 1, 1) is purely temporal.
#
# Padding along the time axis is *cyclic* rather than zero or replicate, because the time
# axis holds one cardiac cycle: the last frame is genuinely adjacent to the first one.
# =====================================================================================

import torch
from torch import nn
import logging
import lightning as L
import numpy as np


class InitRim(L.LightningModule):
    """
    Learned initializer for Recurrent Inference Machines (RIM), based on multi-scale context aggregation
    with dilated convolutions, which replaces the zero initializer for the RIM hidden vector.

    The module aggregates information from different scales of input data using dilated convolutions, which
    are inspired by the paper "Multi-Scale Context Aggregation by Dilated Convolutions" (https://arxiv.org/abs/1511.07122).

    Args:
        x_ch (int): The number of input channels.
        out_channels (list of int): A list of integers specifying the number of output channels (features) for the hidden states in the RIM.
        kernel (tuple): Kernel size for the convolutions used in the initializer.
        multiscale_depth (int): The number of feature layers to aggregate for the output. If 1, multi-scale context aggregation is disabled.
        mute (bool): Whether or not to suppress logging information during initialization. Default is True.

    Attributes:
        conv_blocks (nn.ModuleList): A list of convolutional blocks used for multi-scale context aggregation.
        out_blocks (nn.ModuleList): A list of convolutional blocks used for generating the output.
        multiscale_depth (int): The depth of multi-scale aggregation.
        kernel (tuple): The kernel size for convolutions.
    """

    # Rationale: a RIM needs a hidden state before its first iteration. Starting from zeros
    # wastes the first few iterations on merely building up context, so instead a small
    # network looks at the initial (aliased) estimate and produces a sensible starting
    # state. Dilated convolutions are used so that a shallow stack still sees a large area
    # of the image: the dilation doubles at each layer (1, 1, 2, 4), which widens the
    # receptive field geometrically without adding parameters or downsampling.

    def __init__(self, x_ch, out_channels, kernel, multiscale_depth=0):
        """
        Initializes the InitRim module, which learns an initializer for the hidden states of the RIM.

        Args:
            x_ch (int): The number of input channels.
            out_channels (list): A list of integers specifying the number of output channels for the hidden states.
            kernel (tuple): Kernel size for the convolutions.
            multiscale_depth (int): The number of feature layers to aggregate. If 1, multi-scale aggregation is disabled.
            mute (bool): Whether to suppress logging during initialization.
        """
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        self.out_blocks = nn.ModuleList()
        self.multiscale_depth = multiscale_depth
        self.kernel = kernel

        # Start with input channels
        # tch tracks the channel count as the stack is built, so each layer knows how many
        # channels the previous one produced.
        tch = x_ch

        # Progressively decreasing output channels
        # Widths grow towards the full feature count: [n/4, n/2, n/2, n]. The name says
        # "decreasing", but the list as written increases.
        outchannels = [
                          max(out_channels) // 4] + 2 * [max(out_channels) // 2] + [max(out_channels)]

        # Add convolutional blocks with dilated convolutions
        # Dilations pair up with the channel widths above as (1, 1), (n/2, 2), (n/2, 4)
        # and so on: [1] + [1, 2, 4] gives a geometrically growing receptive field.
        for nconv, (c, d) in enumerate(zip(outchannels, [1] + [2 ** r for r in range(3)])):
            # Set each convolution block with specific input channels (tch) and output channels (c)
            # set_conv_block attaches the block as an attribute named conv1, conv2, ...
            # so that it registers as a proper submodule and its parameters are tracked.
            set_conv_block(self, f'conv{nconv + 1}', tch, c, d)
            self.conv_blocks.append(getattr(self, f'conv{nconv + 1}'))
            tch = c  # Update number of channels for next convolution block

        # Define the output blocks by aggregating the features from the last layers
        # With multiscale_depth = 0 (the default used here) the slice [-0:] takes the whole
        # list, so tch becomes the sum of every layer width. This matches the forward pass,
        # where all collected features are concatenated.
        tch = np.sum(outchannels[-multiscale_depth:])
        for ch in out_channels:
            # Define output block using 3D convolution followed by ReLU activation
            # One output head per recurrent unit; each maps the aggregated features onto
            # that unit's hidden state size.
            outblock = nn.Sequential(
                nn.Conv3d(tch, ch, 1),  # 1x1 convolution to match output channels
                nn.ReLU(inplace=True))
            self.out_blocks.append(outblock)

    def forward(self, x):  # Input shape: [B, F, T, D, W] (Batch, Features, Time, Depth, Width)
        """
        Forward pass for the learned initializer for the RIM.

        The forward pass applies the multi-scale context aggregation by passing the input through the
        convolutional blocks and aggregating the features based on the specified multiscale depth.

        Args:
            x (Tensor): Input tensor of shape [B, F, T, D, W] where:
                        - B: Batch size
                        - F: Number of input features
                        - T: Temporal dimension
                        - D: Depth dimension
                        - W: Width dimension

        Returns:
            list: A list of output tensors generated by the output blocks, where each output has the shape [B, F, T, D, W].
        """
        features = []
        # Pass input through convolutional blocks
        # Each block's output is kept, so that the final representation can combine coarse
        # and fine scales rather than only the deepest one.
        for block in self.conv_blocks:
            x = block(x)
            if self.multiscale_depth != 1:
                features.append(x)  # Collect features for multi-scale aggregation

        # If multi-scale aggregation is enabled, concatenate last features based on depth
        # Concatenation happens along the channel axis, which is why tch above had to be
        # the sum of the individual widths.
        if self.multiscale_depth != 1:
            x = torch.cat(features[-self.multiscale_depth:], dim=1)  # Concatenate features along channel dimension

        # Pass aggregated features through output blocks to get final outputs
        # One tensor per recurrent unit; drim_network indexes them as hidden[0], hidden[1].
        outlst = []
        for block in self.out_blocks:
            outlst.append(block(x))  # Collect output from each output block

        return list(outlst)


def get_num_params(module):
    """Calculate the total number of parameters in a PyTorch module, including the sizes of each parameter"""
    # Diagnostics only; called from the logging branches that are disabled by default.
    # Note this function is defined twice in this file, here and again at the bottom. The
    # later definition is the one that is actually in effect at import time.

    n = 0

    for na, p in module.named_parameters():
        # Log the name and size of each parameter (for debugging)
        module.logger.debug(f"{na}: {p.size()}")

        n += p.numel()

    return n


class GradRim(L.LightningModule):
    """
    A gradient-based reconstruction model for MRI-like data.

    This model computes the gradient of the data consistency term in an iterative
    reconstruction algorithm. It operates on Fourier-transformed data, specifically
    leveraging a sense matrix for coil sensitivity encoding and handling k-space data.

    Args:
        fourier_dim (int): The dimensionality of the Fourier transform. Default is -1,
                            which means the transform will be applied to all spatial dimensions.

    Attributes:
        fourier_dim (int): The dimensionality of the Fourier transform used in the model.

    Methods:
        forward(x, measurements, sense, mask):
            Computes the gradient of the reconstruction with respect to the k-space data and the measurements.
            This is performed by computing the error between the k-space and the measurements and using
            this error to calculate the gradient with respect to the input image.
    """

    # This is the only part of the network that knows about MRI physics, and it contains no
    # learnable parameters at all. It answers the question: given the current image guess,
    # how wrong is it where we actually measured data, and in which direction should the
    # image move to fix that?
    #
    # In operator terms the forward model is  y = M . F . S . x  (sensitivity encoding,
    # Fourier transform, then sampling mask), and what is computed here is the gradient of
    # ||M F S x - y||^2, namely  S^H . F^H . M . (M F S x - y).

    def __init__(self, fourier_dim=-1):
        """Initializes the GradRim model"""
        super().__init__()

        self.fourier_dim = fourier_dim

    def forward(self, x, measurements, sense, mask):
        """Compute the gradient of the reconstruction error with respect to the k-space data"""

        # Unsqueeze x to add an additional channel dimension
        # A coil axis is inserted so the image can broadcast against the per coil
        # sensitivity maps and measurements.
        x = x.unsqueeze(1)  # Shape: [B, 1, T, D, W, 2]

        # Get seperate complex coil images
        # Multi-coil case: multiply the image by each coil's sensitivity. The multiplication
        # is written out by hand because the tensors are stored as real pairs rather than as
        # complex tensors, so this is (a+bi)(c+di) expanded into real and imaginary parts.
        if measurements.shape[1] > 1:
            x = torch.view_as_complex(torch.stack((
                sense[..., 0] * x[..., 0] - sense[..., 1] * x[..., 1],  # Real part
                sense[..., 0] * x[..., 1] + sense[..., 1] * x[..., 0]  # Imaginary part
            ), -1))
        else:
            # Single coil: the sensitivity map is all ones, so the multiplication is skipped.
            x = torch.view_as_complex(x)

        # Perform FFT to get k-space data
        # The ifftshift/fftshift sandwich converts between the centred k-space convention
        # used in MRI and the corner convention the FFT itself uses. norm='ortho' keeps the
        # forward and inverse transforms exact adjoints, which matters for a gradient.
        kspace = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), dim=(-2, -1), norm='ortho'),
            dim=(-2, -1))
        kspace = torch.view_as_real(kspace)

        # Calculate the error between k-space and measurements, apply the mask to ignore invalid points
        # Only sampled positions contribute; everywhere else the residual is forced to zero,
        # since there is no measurement to disagree with. This is the M operator.
        error = torch.where(
            mask,
            kspace - measurements,
            torch.Tensor([0]).to(measurements)
        )

        # Perform inverse FFT to transform error back to image space
        # The residual is carried back to image space, where the network operates.
        error_kspace = torch.view_as_complex(error)
        error_imspace = torch.fft.ifftshift(
            torch.fft.ifft2(torch.fft.fftshift(error_kspace, dim=(-2, -1)), dim=(-2, -1), norm='ortho'),
            dim=(-2, -1))
        error = torch.view_as_real(error_imspace)

        # Compute gradient w.r.t. input image using error and sense map
        # Get seperate complex coil images
        # Multi-coil: combine the per coil errors using the conjugate of the sensitivity
        # maps (note the sign pattern, which differs from the forward multiplication above)
        # and sum over the coil axis. This is the S^H operator.
        if measurements.shape[1] > 1:
            gradient = torch.stack((
                torch.sum(
                    sense[..., 0] * error[..., 0] + sense[..., 1] * error[..., 1],  # Gradient in the real direction
                    1),  # Sum over coil dimension
                torch.sum(
                    sense[..., 0] * error[..., 1] - sense[..., 1] * error[..., 0],
                    # Gradient in the imaginary direction
                    1)  # Sum over coil dimension
            ), 1)  # Stack the real and imaginary gradients
        else:
            # Single coil: no combination needed, only a rearrangement that moves the
            # real/imaginary axis into the channel position and drops the coil axis.
            gradient = torch.permute(error, (0, -1, 2, 3, 4, 1))[..., 0]

        return gradient  # Return computed gradient [B, T, D, W]


def set_conv_block(module, name, n_chan, n_feat, dilation):
    """Set up a convolutional block with padding and 3D convolutional layers"""
    # Builds one nn.Sequential of padding + convolution + activation, and attaches it to
    # `module` under `name`. Attaching by attribute (rather than returning it) is what
    # registers the parameters with PyTorch so they are saved and loaded properly.
    #
    # The block's shape is dictated by module.kernel, the (up to) three kernel entries from
    # the yaml file. Every non-None entry becomes one convolution layer.

    # Create a new sequential container for the conv block
    setattr(module, name, nn.Sequential())
    n_layer = -1

    # Determine number of layers in convolutional block
    # Counts the layers ahead of time, so the loop below can recognise the last one and
    # leave the final activation off the output block.
    for kernel in module.kernel:
        if kernel is not None:
            n_layer += 1

    # Loop through each kernel size to build convolutional block
    for i, kernel in enumerate(module.kernel):
        if kernel is not None:
            # If kernel height is greater than 1, apply cyclic padding
            # kernel[0] is the temporal extent. Padding along time wraps around, because
            # the frames cover one closed cardiac cycle. The padding width accounts for
            # dilation, so that the output keeps the same size as the input.
            if int(kernel[0]) > 1:
                padt = int((
                                   int(kernel[0]) + (int(kernel[0]) - 1) * (dilation - 1)) / 2)
                getattr(module, name).append(CyclicPad1d(padt))

            # If kernel width or depth is greater than 1, apply replication padding
            # Spatial edges are padded by replicating the border pixels. Unlike time, space
            # does not wrap around, and replication avoids the dark border that zero
            # padding would introduce.
            if kernel[1] > 1 or kernel[2] > 1:
                padw = int((
                                   kernel[2] + (kernel[2] - 1) * (dilation - 1)) / 2)
                padd = int((
                                   kernel[1] + (kernel[1] - 1) * (dilation - 1)) / 2)
                getattr(module, name).append(
                    nn.ReplicationPad3d((padw, padw, padd, padd, 0, 0))
                )

            # Add 3D convolutional layer to the block
            # No padding argument here: all padding was applied by the explicit layers
            # above, which is what allows the cyclic variant along the time axis.
            getattr(module, name).append(nn.Conv3d(
                n_chan, n_feat, kernel, dilation=dilation, bias=True))

            # Optionally add ReLU activation after each layer (except the last one)
            # The output block must be able to produce negative values, since it predicts
            # an additive update to a complex image, so its final ReLU is omitted.
            if not (name == 'conv_out' and i == n_layer):
                getattr(module, name).append(nn.ReLU(inplace=True))

            # Update number of input channels for the next layer
            n_chan = n_feat
    return


class DynamicRecurrentInferenceBlock(L.LightningModule):
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

    # One invocation of this block is one RIM iteration's update step. The layout is
    # conv_in -> recurrent1 -> conv_between -> recurrent2 -> conv_out.

    def __init__(
            self, nfeature=128, kernel=((1, 3, 3), (3, 1, 1), None),
            temporal_rnn=False, mute=True):
        super().__init__()
        self.nfeature = nfeature

        # Check kernel configuration validity
        # The kernel entries must be filled from the front: a third layer without a second
        # one, or a second without a first, is not a configuration set_conv_block can build.
        if kernel[2] is not None:
            assert kernel[1] is not None
        if kernel[1] is not None:
            assert kernel[0] is not None
        self.kernel = kernel

        # Set up convolutional blocks
        # 4 input channels: 2 for the current estimate (real, imaginary) and 2 for the
        # data consistency gradient, concatenated by the caller in drim_network.test_step.
        set_conv_block(self, 'conv_in', 4, self.nfeature, 1)

        # Initialize first recurrent unit
        self.recurrent1 = LatticeRecurrentUnit(
            self.nfeature, self.nfeature, temporal=temporal_rnn, mute=mute)

        # Set up the intermediate convolutional block
        set_conv_block(self, 'conv_between', self.nfeature, self.nfeature, 1)

        # Initialize second recurrent unit
        self.recurrent2 = LatticeRecurrentUnit(
            self.nfeature, self.nfeature, temporal=temporal_rnn, mute=mute)

        # Set up the output convolutional block
        # Back down to 2 channels: the real and imaginary parts of the update step. The
        # name 'conv_out' is significant, since set_conv_block checks it to decide whether
        # to leave off the final ReLU.
        set_conv_block(self, 'conv_out', self.nfeature, 2, 1)

        # Logging initialization details if not muted
        # Disabled by default (mute=True). Note that self.logger on a LightningModule is
        # the Trainer's experiment logger, not a logging.Logger, so this branch would fail
        # when logger=False is passed to the Trainer, as main.py does.
        if not mute:
            self.logger.info("Initialized a recurrent inference machine of "
                             f"{get_num_params(self)} parameters. Convolutional blocks "
                             f"are conv_in: {self.conv_in}, conv_between: "
                             f"{self.conv_between}, and conv_out: {self.conv_out}.")

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

        # The two hidden states are updated in place in the list, so the caller's `hidden`
        # is carried forward into the next RIM iteration automatically.
        x = self.conv_in(x)
        x, hidden[0] = self.recurrent1(x, hidden[0])
        x = self.conv_between(x)
        x, hidden[1] = self.recurrent2(x, hidden[1])
        x = self.conv_out(x)

        return x, hidden


class CyclicPad1d(L.LightningModule):
    """
    A 1D cyclic padding layer that pads the input tensor cyclically (i.e., wraps around).

    This module pads the input tensor along the third dimension (T) by appending
    the last `n` elements from the end to the front, and the first `n` elements
    from the beginning to the end, effectively creating a cyclic padding effect.

    Args:
        n (int): The number of elements to pad at both the beginning and the end of the input tensor.

    Attributes:
        n (int): The number of elements to pad at the start and the end along the third dimension (T).

    Methods:
        forward(input):
            Applies cyclic padding to the input tensor along the third dimension (T).
    """

    # Wrap-around padding for the time axis. The frames span one cardiac cycle, so the last
    # frame is a genuine neighbour of the first. Zero padding would tell the network that
    # the heart stops at the end of the cycle; replication would tell it the motion freezes.

    def __init__(self, n):
        """Initializes the CyclicPad1d layer with the specified padding size"""
        super().__init__()

        self.n = n

    def __repr__(self):
        """Returns a string representation of the CyclicPad1d layer """
        # Printed in the same style as PyTorch's own padding layers, so that a printed
        # model summary reads consistently.

        return f"{type(self).__name__}: ({self.n}, {self.n}, 0, 0, 0, 0)"

    def forward(self, input):
        """Applies cyclic padding to the input tensor along the third dimension (T)"""
        # Axis 2 is the time axis in the [B, F, T, D, W] layout. The last n frames are
        # placed in front and the first n frames behind, then everything is concatenated.

        return torch.cat(
            (input[:, :, -self.n:], input, input[:, :, :self.n]), 2)


class LatticeRecurrentUnit(nn.Module):
    """
    A Lattice Recurrent Unit (LRU) implementation for temporal data processing, which extends the idea
    of a Gated Recurrent Unit (GRU) with convolutional operations for both the input and hidden states.

    The LatticeRecurrentUnit uses 3D convolutions to process the input and hidden states across different
    dimensions, allowing it to capture both spatial and temporal dependencies. The unit also uses cyclic padding
    to handle boundary conditions for the temporal dimension.

    Args:
        input_size (int): The number of input features.
        hidden_size (int): The number of hidden units in the recurrent unit.
        temporal (bool): Whether or not to use temporal convolutions. Defaults to False.
        mute (bool): Whether to suppress logging. Defaults to True.

    Attributes:
        conv_ih (nn.Conv3d): 3D convolution layer for input-to-hidden connections.
        conv_hh (nn.Conv3d): 3D convolution layer for hidden-to-hidden connections.
        bias_h (nn.parameter.Parameter): Bias parameter for hidden state.
        input_size (int): The number of input features.
        hidden_size (int): The number of hidden units.
        temporal (bool): Whether temporal convolutions are used.
        logger (logging.Logger): Logger to record information about the initialization of the unit.

    Methods:
        forward(input, hidden):
            Perform the forward pass for the LatticeRecurrentUnit, including the calculations for reset, update,
            and candidate hidden states based on input and previous hidden state.
    """

    # A convolutional GRU with a twist: where a standard GRU keeps one running state, this
    # "lattice" variant maintains two interleaved streams, one flowing along the input path
    # and one along the hidden path, each with its own reset and update gate. That is why
    # the convolutions below produce 6 x hidden_size channels (three gate quantities for
    # each of the two streams) instead of the usual 3 x for a GRU.
    #
    # Note this class derives from nn.Module rather than L.LightningModule, unlike the
    # others in this file, which is why self.logger here is a real logging.Logger.

    def __init__(self, input_size, hidden_size, temporal=False, mute=True):
        """
        Initializes the LatticeRecurrentUnit.

        Args:
            input_size (int): The number of input features.
            hidden_size (int): The number of hidden units in the recurrent unit.
            temporal (bool): Whether to use temporal convolutions (default is False).
            mute (bool): Whether to suppress logging (default is True).
        """
        super().__init__()
        self.logger = logging.getLogger(type(self).__name__)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.temporal = temporal

        # Input-to-hidden convolution: Conv3d to transform input to hidden space
        # The kernel is (3, 1, 1) when temporal mixing is enabled, so each frame sees its
        # two neighbours in time but no neighbouring pixels in space; (1, 1, 1) otherwise,
        # which makes it a per pixel linear map. Spatial context is supplied by the
        # convolution blocks around this unit, not by the unit itself.
        self.conv_ih = nn.Conv3d(
            input_size, 6 * hidden_size,
            (3 if temporal else 1, 1, 1), bias=True)  # 3D kernel with size depending on temporal flag

        # Hidden-to-hidden convolution: Conv3d to propagate previous hidden state
        # bias=False because conv_ih above already contributes a bias to the same sums,
        # and a second one would be redundant.
        self.conv_hh = nn.Conv3d(
            hidden_size, 6 * hidden_size,
            (3 if temporal else 1, 1, 1), bias=False)  # 3D kernel for hidden state propagation

        # Bias for the hidden state
        # A separate learnable bias applied only inside the candidate hidden state, shaped
        # to broadcast over batch, time, depth and width.
        self.bias_h = nn.parameter.Parameter(
            torch.zeros(self.hidden_size).view(1, self.hidden_size, 1, 1, 1))  # Bias parameter

        # Log initialization information if not muted
        if not mute:
            self.logger.info(
                f'{type(self).__name__} '
                f'{"with temporal convolutions" if temporal else ""}'
                f' initialized with input size {input_size}, '
                f'hidden size {hidden_size}, containing '
                f'{get_num_params(self)} parameters.')

    def forward(self, input, hidden):
        """
        Perform the forward pass for the LatticeRecurrentUnit.

        The forward pass includes computing the reset and update gates, and the candidate hidden state.
        If temporal convolutions are enabled, the input and hidden states are padded using cyclic padding.

        Args:
            input (Tensor): The input tensor of shape [B, F, T, D, W] where:
                             - B: Batch size
                             - F: Number of input features
                             - T: Temporal dimension
                             - D: Depth
                             - W: Width
            hidden (Tensor): The previous hidden state of shape [B, H, T, D, W] where:
                             - H: Number of hidden units

        Returns:
            Tuple: A tuple containing the updated output and hidden state:
                - output (Tensor): The updated output tensor of shape [B, H, T, D, W]
                - hidden (Tensor): The updated hidden state tensor of shape [B, H, T, D, W]
        """
        # print(torch.cuda.memory_summary())
        # Apply cyclic padding if temporal convolutions are enabled
        # Padding by 1 on each side compensates for the (3, 1, 1) kernel, so the time axis
        # keeps its length. A fresh CyclicPad1d is constructed per call; it holds no
        # parameters, so this costs nothing but an object allocation.
        if self.temporal:
            padded_input = CyclicPad1d(1)(input)  # Apply cyclic padding to input
            padded_hidden = CyclicPad1d(1)(hidden)  # Apply cyclic padding to hidden state
        else:
            padded_input = input  # No padding if no temporal convolutions
            padded_hidden = hidden

        # Perform input-to-hidden convolution and split the results into gates and candidate values
        # One convolution computes all six quantities at once, then chunk splits them along
        # the channel axis. Naming: i/h prefix = which stream, second letter = which path,
        # suffix r/z/c = reset gate, update gate, candidate.
        ii_r, ih_r, ii_z, ih_z, ii_c, ih_c = self.conv_ih(padded_input).chunk(6, 1)

        # Perform hidden-to-hidden convolution and split the results into gates and candidate values
        hi_r, hh_r, hi_z, hh_z, hi_c, hh_c = self.conv_hh(padded_hidden).chunk(6, 1)

        # Reset and update gates using sigmoid activation
        # Each gate combines an input-derived and a hidden-derived term. Sigmoid keeps them
        # in [0, 1], so they act as soft switches.
        reset_i = torch.sigmoid(ii_r + hi_r)
        reset_h = torch.sigmoid(ih_r + hh_r)
        update_i = torch.sigmoid(ii_z + hi_z)
        update_h = torch.sigmoid(ih_z + hh_z)

        # Candidate hidden states using tanh activation
        # The reset gates control how much of the other stream leaks in before the
        # nonlinearity. Note the crossover: the input candidate is gated by reset_h and the
        # hidden candidate by reset_i, which is what couples the two streams together.
        candidate_i = torch.tanh(ih_c + reset_h * (hh_c + self.bias_h))
        candidate_h = torch.tanh(reset_i * ii_c + hi_c)

        # Calculate the final output and hidden state using the update gates and candidate values
        # Standard GRU style interpolation: each update gate decides how much of the new
        # candidate to accept versus how much of the previous value to keep. Because the
        # output path retains (1 - update_i) * input, the unit can pass its input straight
        # through, which keeps gradients healthy across many iterations.
        output = update_i * candidate_h + (1 - update_i) * input
        hidden = update_h * candidate_i + (1 - update_h) * hidden

        return output, hidden


def get_num_params(module):
    """Calculate the total number of parameters in a PyTorch module, including the sizes of each parameter"""
    # Identical to the definition near the top of this file. Being the later of the two, it
    # is the one that is actually bound to the name after the module is imported.

    n = 0

    for na, p in module.named_parameters():
        # Log the name and size of each parameter (for debugging)
        module.logger.debug(f"{na}: {p.size()}")

        n += p.numel()

    return n
