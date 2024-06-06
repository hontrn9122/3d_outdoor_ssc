from typing import List, Tuple, Union
import torch
import torch.nn as nn
from spconv.core import ConvAlgo
from timm.models.layers import trunc_normal_
import spconv.pytorch as spconv
from spconv.pytorch import SparseConv3d, SparseInverseConv3d, SubMConv3d, SpatialGroupConv3d, SparseModule

class SpatialGroupConv(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, indice_key=None, bias=False, _type='A'):
        super(SpatialGroupConv, self).__init__()
        self.kernel_size = kernel_size
        self.indice_key = indice_key
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block = spconv.SubMConv3d(
                                        in_channels,
                                        out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=int(kernel_size//2),
                                        bias=bias,
                                        indice_key=indice_key,
                                        #algo=ConvAlgo.Native
                                    )

        self.conv3x3_1 = spconv.SubMConv3d(
                                        in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=stride,
                                        padding=1,
                                        bias=bias,
                                        dilation=3,
                                        indice_key=indice_key+'conv_3x3_1' if indice_key is not None else None,
                                        #algo=ConvAlgo.Native
                                    )

        self._indice_list = []

        if kernel_size==7:
            _list = [0, 3, 4, 7]
        elif kernel_size==5:
            _list = [0, 2, 3, 5]
        elif kernel_size==3:
            _list = [0, 1, 2]
        else:
            raise ValueError('Unknown kernel size %d'%kernel_size)
        for i in range(len(_list)-1):
            for j in range(len(_list)-1):
                for k in range(len(_list)-1):
                    a = torch.zeros((kernel_size, kernel_size, kernel_size)).long()
                    a[_list[i]:_list[i+1], _list[j]:_list[j+1], _list[k]:_list[k+1]] = 1
                    b = torch.range(0, kernel_size**3-1, 1)[a.reshape(-1).bool()]
                    self._indice_list.append(b.long())

    def _convert_weight(self, weight):
        weight_reshape = self.block.weight.permute(3, 4, 0, 1, 2).reshape(self.out_channels, self.in_channels, -1).clone()
        weight_return = self.block.weight.permute(3, 4, 0, 1, 2).reshape(self.out_channels, self.in_channels, -1).clone()
        for _indice in self._indice_list:
            _mean_weight = torch.mean(weight_reshape[:, :, _indice], dim=-1, keepdim=True)
            weight_return[:, :, _indice] = _mean_weight
        return weight_return.reshape(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, self.kernel_size).permute(2, 3, 4, 0, 1)

    def forward(self, x_conv):
        if self.training:
            self.block.weight.data = self._convert_weight(self.block.weight.data).contiguous()
        x_conv_block = self.block(x_conv)

        x_conv_conv3x3_1 = self.conv3x3_1(x_conv)

        x_conv_block = x_conv_block.replace_feature(x_conv_block.features + x_conv_conv3x3_1.features)
        return x_conv_block

class SpatialGroupConvV2(SparseModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, indice_key=None, bias=False, dilation=1, _type='A'):
        super(SpatialGroupConvV2, self).__init__()
        self.kernel_size = kernel_size
        self.indice_key = indice_key
        self.in_channels = in_channels
        self.out_channels = out_channels

        if kernel_size==3:
            kernel_size = 7
        _list = [0, int(kernel_size//2), int(kernel_size//2)+1, 7]
        self.group_map = torch.zeros((3**3, int(kernel_size//2)**3)) - 1
        _num = 0
        for i in range(len(_list)-1):
            for j in range(len(_list)-1):
                for k in range(len(_list)-1):
                    a = torch.zeros((kernel_size, kernel_size, kernel_size)).long()
                    a[_list[i]:_list[i+1], _list[j]:_list[j+1], _list[k]:_list[k+1]] = 1
                    _pos = a.sum()
                    self.group_map[_num][:_pos] = torch.range(0, kernel_size**3-1, 1)[a.reshape(-1).bool()]
                    _num += 1
        self.group_map = self.group_map.int()
        position_embedding = True
        self.block = SpatialGroupConv3d(
                                        in_channels,
                                        out_channels,
                                        kernel_size, 3,
                                        stride=stride,
                                        padding=int(kernel_size//2),
                                        bias=bias,
                                        dilation=dilation,
                                        indice_key=indice_key,
                                        algo=ConvAlgo.Native,
                                        position_embedding=position_embedding,
                                    )
        if position_embedding:
            trunc_normal_(self.block.position_embedding, std=0.02)

    def forward(self, x_conv):
        x_conv = self.block(x_conv, group_map=self.group_map.to(x_conv.features.device))
        return x_conv

def get_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, List[int], Tuple[int, ...]],
    stride: Union[int, List[int], Tuple[int, ...]] = 1,
    padding: Union[int, List[int], Tuple[int, ...]] = 0,
    dilation: int = 1,
    groups: int = 1,
    transposed: bool = False,
    sgc = False,
    **kwargs):
    if transposed:
        return SparseInverseConv3d(
            in_channels,
            out_channels,
            kernel_size,
            # padding=padding,
            # stride=stride,
            # dilation=dilation, 
            **kwargs
        )
    if sgc:
        return SpatialGroupConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            # dilation=dilation,
            # algo=ConvAlgo.Native,
            # position_embedding=True,
            **kwargs
        )
    if stride == 1:
        layer = SubMConv3d
    else:
        layer = SparseConv3d
    
    return layer(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        dilation=dilation,
        groups=groups,
        padding=padding,
        **kwargs
    )