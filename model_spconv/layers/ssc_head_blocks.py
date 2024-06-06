from typing import List
import torch
import torch.nn as nn
from spconv.pytorch import SubMConv3d, SparseConv3d


class SSCHeadBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        act = nn.ReLU(True),
    ):
        super(SSCHeadBlock, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bot_channels = in_channels//2
        
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding=1),
            act,
            nn.Conv3d(32, num_classes, 3, padding=1),
        )
        
    def forward(self, x):
        return self.block(x.dense())
        

# class SSCHeadBlock(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         num_classes: int,
#         # bot_channels: int,
#     ):
#         super(SSCHeadBlock, self).__init__()
#         self.in_channels = in_channels
#         self.num_classes = num_classes
#         self.bot_channels = in_channels//2
        
#         self.block = nn.ModuleDict()
#         for i in range(1, 8, 2):
#             if i == 1:
#                 self.block[f'Conv{i}x{i}']= nn.Conv3d(
#                     self.in_channels,
#                     self.num_classes,
#                     1,
#                 )
#             else:
#                 self.block[f'Conv{i}x{i}']= nn.Sequential(
#                     nn.Conv3d(
#                         in_channels,
#                         self.bot_channels,
#                         1,
#                     ),
#                     nn.Conv3d(
#                         self.bot_channels,
#                         self.num_classes,
#                         i,
#                         padding=(i-1)//2
#                     )
#                 )
#         self.mlp = nn.Sequential(
#             nn.Conv3d(
#                 self.num_classes*4,
#                 self.num_classes,
#                 1
#             ),
#             # nn.Softmax(dim=1)
#         )
        
#     def forward(self, input):
#         input = input.dense()
#         feats = []
#         for l in self.block.values():
#             feats.append(l(input))
#         return self.mlp(torch.cat(feats, dim=1))

    
class SSCHeadBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
    ):
        super(SSCHeadBlock, self).__init__()
        self.shortcut1 = SubMConv3d(
            in_channels = in_channels,
            out_channels = in_channels//4,
            kernel_size=1,
        )
        
        self.shortcut3 = SparseConv3d(
            in_channels = in_channels,
            out_channels = in_channels//4,
            kernel_size=3,
            padding=1
        )
        
        self.main = nn.ModuleDict({
            'start_conv': SparseConv3d(
                in_channels = in_channels,
                out_channels = in_channels//2,
                kernel_size=3,
                padding=1,
            ),
            'sparse_block': nn.ModuleList([
                SparseConv3d(
                    in_channels = in_channels//2,
                    out_channels = in_channels//c,
                    kernel_size=i,
                    padding=(i-1)//2
                ) for i, c in [(1, 2), (3, 4), (5, 4)]
            ]),
            'bottleneck': nn.Sequential(
                nn.Conv3d(
                    in_channels = in_channels,
                    out_channels = in_channels//4,
                    kernel_size = 1
                ),
                # nn.BatchNorm3d(in_channels//4),
                # nn.ReLU(True),
                nn.GELU(),
                nn.BatchNorm3d(in_channels//4),
            ),
            'dense_block': nn.ModuleList([
                nn.Conv3d(
                    in_channels=in_channels//4,
                    out_channels=in_channels//(2*c),
                    kernel_size=i,
                    padding = (i-1)//2,
                    dilation=1,
                ) for i, c in [(1, 2), (3, 4), (5, 4)]
            ]),
            'fuse': nn.Sequential(
                nn.Conv3d(
                    in_channels = in_channels,
                    out_channels = in_channels,
                    kernel_size=1,
                ),
                # nn.BatchNorm3d(in_channels),
                # nn.ReLU(True),
                nn.GELU(),
                nn.BatchNorm3d(in_channels),
            ),
            'fc': nn.Sequential(
                nn.Conv3d(
                    in_channels = in_channels,
                    out_channels = num_classes,
                    kernel_size = 1
                ),
                nn.Softmax(dim=1)
            )
        })
        
    def forward(self, input):
        start_conv = self.main['start_conv'](input)
        xs = [l(start_conv).dense() for l in self.main['sparse_block']]
        # x = sum(xs)
        x = torch.cat(xs, dim=1)
        x = self.main['bottleneck'](x)
        xs = [l(x) for l in self.main['dense_block']]
        # x = sum(xs)
        x = torch.cat(xs, dim=1)
        return self.main['fc'](self.main['fuse'](torch.cat(
            [self.shortcut1(input).dense(), self.shortcut3(input).dense(), x], dim=1
        )))