import torch
import torch.nn as nn
from torch.nn import functional as F

import spconv.pytorch as spconv
from spconv.core import ConvAlgo
# from spconv.pytorch import SparseConvTensor, SpatialGroupConv3d

import math
# from model_spconv.utils import sparse_reduce_dim, sparse_split_dim, get_conv
# from .modules import SparseBatchNorm


class Squash(nn.Module):
    def __init__(self, eps=10e-21):
        super(Squash, self).__init__()
        self.eps = eps
    
    def forward(self, caps, dim=2):
        norm = torch.linalg.norm(caps, dim=dim,keepdim=True)
        return (1 - 1/(torch.exp(norm)+self.eps))*(caps/(norm+self.eps))

    
class DepthwiseConv3d(nn.Module):
    """
    Performs 2D convolution given a 5D input tensor.
    This layer given an input tensor of shape
    `[batch, in_num_caps, in_caps_dim, input_height, input_width]` squeezes the
    first two dimmensions to get a 4D tensor as the input of torch.nn.Conv2d. Then
    splits the first dimmension and the second dimmension and returns the 6D
    convolution output.
    Args:
        kernel_size: scalar or tuple, convolutional kernels are [kernel_size, kernel_size, kernel_size].
        in_num_caps: scalar, number of capsules in the layer below.
        out_num_caps: scalar, number of capsules in this layer.
        in_caps_dim: scalar, number of units in each capsule of input layer.
        out_caps_dim: scalar, number of units in each capsule of output layer.
        stride: scalar or tuple, stride of the convolutional kernel.
        padding: scalar or tuple, zero-padding added to both sides of the input
        dilation: scalar or tuple, spacing between kernel elements
        share_weight: share transformation weight matrices between capsules in lower layer or not
    Returns:
        6D SparseTensor output of a 3D convolution with feats shape
        `[n, in_num_caps, out_num_caps, out_caps_dim]`.
    """
    def __init__(
        self,
        kernel_size,
        in_num_caps,
        out_num_caps,
        in_caps_dim=8,
        out_caps_dim=8,
        stride=2,
        dilation=1,
        padding=0,
        share_weight=True,
        padding_mode='zeros' 
    ):
        super(DepthwiseConv3d, self).__init__()
        self.in_num_caps = in_num_caps
        self.out_num_caps = out_num_caps
        self.in_caps_dim = in_caps_dim
        self.out_caps_dim = out_caps_dim
        self.share_weight = share_weight
        
        if self.share_weight:
            self.conv2d = nn.Sequential(
                nn.Conv2d(
                    in_caps_dim,
                    out_num_caps * out_caps_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding,
                    padding_mode=padding_mode
                ),
            )
        else:
            self.conv2d = nn.Sequential(
               nn.Conv2d(
                    in_num_caps * in_caps_dim,
                    in_num_caps * out_num_caps * out_caps_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding,
                    groups=in_num_caps,
                    padding_mode=padding_mode
                ),
            )
            # raise 'Currently not support the depthwise convolution (share_weight==False)!'
        # torch.nn.init.normal_(self.conv3d.weight, std=0.1)

    def forward(self, input_tensor):
        # input_shape = input_tensor.size()
        
        input_shape = input_tensor.size()

        if self.share_weight:
            input_tensor_reshaped = input_tensor.view(
                input_shape[0] * self.in_num_caps, self.in_caps_dim, input_shape[-2], input_shape[-1]
            )
        else:
            input_tensor_reshaped = input_tensor.view(
                input_shape[0], self.in_num_caps * self.in_caps_dim, input_shape[-2], input_shape[-1]
            )

        conv = self.conv2d(input_tensor_reshaped)
        conv_shape = conv.size()

        conv_reshaped = conv.view(
            input_shape[0], self.in_num_caps, self.out_num_caps, self.out_caps_dim, conv_shape[-2], conv_shape[-1]
        )
        return conv_reshaped
    
class DepthwiseConv4d(nn.Module):
    """
    Performs 3D convolution given a 6D input tensor.
    This layer given an input tensor of shape
    `[batch, in_num_caps, in_caps_dim, input_height, input_width, input_depth]` squeezes the
    first two dimmensions to get a 5D tensor as the input of torch.nn.Conv3d. Then
    splits the first dimmension and the second dimmension and returns the 7D
    convolution output.
    Args:
        kernel_size: scalar or tuple, convolutional kernels are [kernel_size, kernel_size, kernel_size].
        in_num_caps: scalar, number of capsules in the layer below.
        out_num_caps: scalar, number of capsules in this layer.
        in_caps_dim: scalar, number of units in each capsule of input layer.
        out_caps_dim: scalar, number of units in each capsule of output layer.
        stride: scalar or tuple, stride of the convolutional kernel.
        padding: scalar or tuple, zero-padding added to both sides of the input
        dilation: scalar or tuple, spacing between kernel elements
        share_weight: share transformation weight matrices between capsules in lower layer or not
    Returns:
        7D SparseTensor output of a 3D convolution with feats shape
        `[n, in_num_caps, out_num_caps, out_caps_dim]`.
    """

    def __init__(
        self,
        kernel_size,
        in_num_caps,
        out_num_caps,
        in_caps_dim=8,
        out_caps_dim=8,
        stride=1,
        dilation=1,
        padding=0,
        share_weight=True,
    ):
        super(DepthwiseConv4d, self).__init__()
        self.in_num_caps = in_num_caps
        self.out_num_caps = out_num_caps
        self.in_caps_dim = in_caps_dim
        self.out_caps_dim = out_caps_dim
        self.share_weight = share_weight
        
        if self.share_weight:
            self.conv3d = nn.Sequential(
                get_conv(
                    in_caps_dim,
                    out_num_caps * out_caps_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding,
                ),
                # SparseBatchNorm(out_num_caps * out_caps_dim),
            )
        else:
            self.conv3d = nn.Sequential(
                get_conv(
                    in_num_caps * in_caps_dim,
                    in_num_caps * out_num_caps * out_caps_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding,
                    groups=in_num_caps,
                    algo = ConvAlgo.Native,
                ),
                SparseBatchNorm(in_num_caps * out_num_caps * out_caps_dim),
            )
            # raise 'Currently not support the depthwise convolution (share_weight==False)!'
        # torch.nn.init.normal_(self.conv3d.weight, std=0.1)

    def forward(self, input_tensor):
        # input_shape = input_tensor.size()
        
        if self.share_weight:
            input_tensor_reshaped, batch_size = sparse_reduce_dim(
                input_tensor,
                n_caps = self.in_num_caps, 
                d_caps = self.in_caps_dim,
            )
            conv = self.conv3d(input_tensor_reshaped)
            caps, feats = sparse_split_dim(
                conv,
                self.in_num_caps,
                self.out_num_caps,
                self.out_caps_dim,
                batch_size
            )  # u shape = (B, N_in, N_out, d_out)
        else:
            # raise 'Currently not support the depthwise convolution (share_weight==False)!'
            caps = self.conv3d(input_tensor)
            feats = caps.features.view(-1, self.in_num_caps, self.out_num_caps, self.out_caps_dim)

        # print(input_tensor_reshaped.feats.shape)

        
        return caps, feats

class ConvCapsule2d(nn.Module):
    def __init__(
        self,
        kernel_size,
        in_num_caps,
        in_caps_dim,
        out_num_caps,
        out_caps_dim,
        stride=1,
        padding=0,
        dilation=1,
        # num_routing=3,
        share_weight=False,
        padding_mode='circular',
        squash=True,
    ):
        super(ConvCapsule2d, self).__init__()

        self.routing = AttentionRouting(in_num_caps, out_num_caps, out_caps_dim, squash)

        self.depthwise_conv3d = DepthwiseConv3d(
            kernel_size=kernel_size,
            in_num_caps=in_num_caps,
            out_num_caps=out_num_caps,
            in_caps_dim=in_caps_dim,
            out_caps_dim=out_caps_dim,
            stride=stride,
            padding=padding,
            dilation=dilation,
            share_weight=share_weight,
            padding_mode=padding_mode 
        )

    def forward(self, input_tensor):
        # print(input_tensor.feats.shape)
        caps = self.depthwise_conv3d(input_tensor)
        return self.routing(caps)


class SparseConvCapsule3d(nn.Module):
    """
    Builds a slim convolutional capsule layer.
    This layer performs 3D convolution given 6D input tensor of shape
    `[batch, in_num_caps, in_caps_dim, input_height, input_width, input_depth]`. Then refines
    the votes with routing and applies Squash non linearity for each capsule.
    Each capsule in this layer is a convolutional unit and shares its kernel over
    the position grid and different capsules of layer below. Therefore, number
    of trainable variables in this layer is:
        kernel: [kernel_size, kernel_size, kernel_size, in_caps_dim, out_num_caps * out_caps_dim]
        bias: [out_num_caps, out_caps_dim]
    Output of a conv3d layer is a single capsule with channel number of atoms.
    Therefore conv_slim_capsule_3d is suitable to be added on top of a conv3d layer
    with num_routing=1, in_num_caps=1 and in_caps_dim=conv_channels.
    Args:
        kernel_size: scalar or tuple, convolutional kernels are [kernel_size, kernel_size, kernel_size].
        in_num_caps: scalar, number of capsules in the layer below.
        out_num_caps: scalar, number of capsules in this layer.
        in_caps_dim: scalar, number of units in each capsule of input layer.
        out_caps_dim: scalar, number of units in each capsule of output layer.
        stride: scalar or tuple, stride of the convolutional kernel.
        padding: scalar or tuple, zero-padding added to both sides of the input
        dilation: scalar or tuple, spacing between kernel elements
        num_routing: scalar, number of routing iterations.
        share_weight: share transformation weight matrices between capsules in lower layer or not
    Returns:
        Tensor of activations for this layer of shape
        `[batch, out_num_caps, out_caps_dim, out_height, out_width, out_depth]`
    """

    def __init__(
        self,
        kernel_size,
        in_num_caps,
        out_num_caps,
        in_caps_dim=8,
        out_caps_dim=8,
        stride=2,
        padding=0,
        dilation=1,
        # num_routing=3,
        share_weight=False,
    ):
        super(SparseConvCapsule3d, self).__init__()

        self.routing = SparseAttentionRouting(in_num_caps, out_num_caps, out_caps_dim)

        self.depthwise_conv4d = DepthwiseConv4d(
            kernel_size=kernel_size,
            in_num_caps=in_num_caps,
            out_num_caps=out_num_caps,
            in_caps_dim=in_caps_dim,
            out_caps_dim=out_caps_dim,
            stride=stride,
            padding=padding,
            dilation=dilation,
            share_weight=share_weight,
        )

    def forward(self, input_tensor):
        # print(input_tensor.feats.shape)
        caps, feats = self.depthwise_conv4d(input_tensor)
        return self.routing(caps, feats)
    
        
class AttentionRouting(nn.Module):
    def __init__(
        self, 
        in_num_caps,
        out_num_caps,
        out_caps_dim,
        squash=True,
    ):
        super().__init__()
        self.in_num_caps = in_num_caps
        self.out_num_caps = out_num_caps
        self.out_caps_dim = out_caps_dim

        self.b = nn.Parameter(torch.zeros(
            1,
            self.in_num_caps,
            self.out_num_caps,
            1, 1, 1
        ))
        if squash:
            self.squash = Squash()
        else:
            self.squash = None

    def forward(self, u):
        # print(input_caps.shape)
        # u shape (B, N_in, N_out, d_out, H, W)
        c = torch.einsum('...injhw,...knjhw->...inhw', u, u).unsqueeze(-3)  #shape=(B, N_in, N_out, 1, H, W)
        
        c = c/math.sqrt(self.out_caps_dim)
        c = F.softmax(c, dim=2)
        
        c = c + self.b #shape=(B, N_in, N_out, 1, H, W)
        s = torch.sum(torch.mul(u, c),dim=1)         # s shape=(B, N_out, d_out, H, W)
        if self.squash is not None:
            s = self.squash(s, dim=2)       # s shape=(B, N_out, d_out, H, W)
        
        return s
