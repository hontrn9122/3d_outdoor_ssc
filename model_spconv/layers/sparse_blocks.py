from typing import List, Tuple, Union

import numpy as np
from torch import nn

from spconv.pytorch import SparseConvTensor, Identity
from model_spconv.utils import get_conv, cat, sparse_add
from .modules import SparseSequential, SparseReLU, SparseBatchNorm, SparseLeakyReLU, SparseGELU, SparseReLU


class SparseConvBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]],
        indice_key: str,
        stride: Union[int, List[int], Tuple[int, ...]] = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__(
            get_conv(
                in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, indice_key=indice_key,
            ),
            # SparseBatchNorm(out_channels),
            # SparseReLU(True),
            SparseReLU(True),
            SparseBatchNorm(out_channels),
        )


class SparseConvTransposeBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]],
        indice_key: str,
        stride: Union[int, List[int], Tuple[int, ...]] = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__(
            get_conv(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                transposed=True,
                indice_key=indice_key,
            ),
            # SparseBatchNorm(out_channels),
            # SparseReLU(True),
            SparseReLU(True),
            SparseBatchNorm(out_channels),
        )


class SparseResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]],
        stride: Union[int, List[int], Tuple[int, ...]] = 1,
        dilation: int = 1,
        act=SparseReLU(True),
        sgc=False,
    ) -> None:
        super().__init__()
        
        self.main = nn.Sequential(
            # get_conv(in_channels, out_channels, kernel_size, dilation=dilation, stride=stride),
            get_conv(in_channels, out_channels, 7, dilation=dilation, sgc=sgc),
            # SparseAsymBlock(in_channels, out_channels, kernel_size, dilation=dilation, stride=stride),
            # SparseMultiBlock(in_channels, out_channels, act),
            # SparseBatchNorm(out_channels),
            # SparseReLU(True),
            act,
            SparseBatchNorm(out_channels),
            # get_conv(out_channels, out_channels, kernel_size, dilation=dilation),
            get_conv(out_channels, out_channels, 7, sgc=sgc),
            # SparseAsymBlock(out_channels, out_channels, kernel_size, dilation=dilation),
            # SparseMultiBlock(out_channels, out_channels),
            SparseBatchNorm(out_channels),
        )

        if in_channels != out_channels or np.prod(stride) != 1:
            self.shortcut = nn.Sequential(
                get_conv(in_channels, out_channels, 1, stride=stride),
                SparseBatchNorm(out_channels),
            )
        else:
            self.shortcut = Identity()

        self.fuse = nn.Sequential(
            # SparseReLU(True),
            act,
            # SparseBatchNorm(out_channels)
        )

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        x = self.fuse(sparse_add(self.main(x), self.shortcut(x)))
        return x
    
class SparseAsymBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]],
        stride: Union[int, List[int], Tuple[int, ...]] = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.branch1 = nn.Sequential(
            get_conv(
                in_channels, out_channels,
                kernel_size = [kernel_size, 1, kernel_size],
                padding = [(kernel_size-1)//2, 1, (kernel_size-1)//2],
                dilation=dilation, stride=stride
            ),
            # SparseReLU(True),
            SparseReLU(True),
            SparseBatchNorm(out_channels),
            get_conv(
                out_channels, out_channels,
                kernel_size = [1, kernel_size, kernel_size],
                padding = [0, (kernel_size-1)//2, (kernel_size-1)//2],
                dilation=dilation, stride=stride,
            ),
            # SparseReLU(True),
            # SparseBatchNorm(out_channels),
        )
        
        self.branch2 = nn.Sequential(
            get_conv(
                in_channels, out_channels,
                kernel_size = [1, kernel_size, kernel_size],
                padding = [0, (kernel_size-1)//2, (kernel_size-1)//2],
                dilation=dilation, stride=stride,
            ),
            # SparseReLU(True),
            SparseReLU(True),
            SparseBatchNorm(out_channels),
            get_conv(
                out_channels, out_channels,
                kernel_size = [kernel_size, 1, kernel_size],
                padding = [(kernel_size-1)//2, 0, (kernel_size-1)//2],
                dilation=dilation, stride=stride,
            ),
            # SparseReLU(True),
            # SparseBatchNorm(out_channels),
        )

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        # x = self.relu(self.main(x) + self.shortcut(x))
        return sparse_add(self.branch1(x), self.branch2(x))
    

class SparseMultiBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        act=SparseReLU(True),
    ):
        super(SparseMultiBlock, self).__init__()
        self.convH = get_conv(
            in_channels, out_channels//2,
            kernel_size=[3,3,1],
            padding=[1,1,0],
        )
        self.conv3 = get_conv(
            in_channels, out_channels//8,
            kernel_size=3, padding=1,
        )
        self.sgc5 = get_conv(
            in_channels, out_channels//8,
            kernel_size=5, padding=2, sgc=False,
        )
        self.sgc7 = get_conv(
            in_channels, out_channels//4,
            kernel_size=7, padding=3, sgc=False,
        )
        self.fuse = nn.Sequential(
            get_conv(out_channels, out_channels, 1),
            act,
            SparseBatchNorm(out_channels),
        )
            
    
    def forward(self, x):
        xs = [
            self.convH(x),
            self.conv3(x),
            self.sgc5(x),
            self.sgc7(x),
        ]
        return self.fuse(cat(xs))