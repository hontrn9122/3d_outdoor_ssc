import math
from typing import List, Tuple

import torch
from torch import nn

import spconv.pytorch as spconv
from spconv.pytorch import SparseConvTensor

from model_spconv.layers import SparseConvCapsule3d, SparseConvBlock, SparseConvTransposeBlock, SparseResBlock, SparseSequential, SparseBatchNorm, SparseGELU, SparseReLU, SparseLeakyReLU

from model_spconv.utils import feat_cat, get_conv, cat

from KPConv.models.blocks import KPConv

import torch_geometric
from torch_geometric.nn import knn, radius, MLP
from torch_geometric.utils import to_dense_batch


class PointVoxelEncoder(nn.Module):
    def __init__(
        self,
        point_in_channels: int,
        voxel_in_channels: int,
        stem_channels: int,
        feature_channels: List[int],
        point_channels: List[int],
        voxel_size: int,
    ): 
        super(PointVoxelEncoder, self).__init__()
        self.stem = nn.ModuleDict({
            'feat_extractor': nn.Sequential(
                get_conv(voxel_in_channels, stem_channels//2, 5, padding=2),
                # SparseBatchNorm(stem_channels//2),
                # nn.ReLU(True),
                SparseReLU(True),
                SparseBatchNorm(stem_channels//2),
                get_conv(stem_channels//2, stem_channels, 5, padding=2),
                # SparseBatchNorm(stem_channels//2),
                # nn.ReLU(True),
                SparseReLU(True),
                SparseBatchNorm(stem_channels),
            ),
            # 'fuse': get_conv(stem_channels, stem_channels, 1)
            'fuse': nn.Sequential(
                get_conv(stem_channels, stem_channels, 5, padding=2),
                # SparseBatchNorm(stem_channels//2),
                # nn.ReLU(True),
                SparseReLU(True),
                SparseBatchNorm(stem_channels),
            )
        })

        self.voxel_encoders = nn.ModuleList()
        feature_channels = [stem_channels] + feature_channels
        for k in range(1, len(feature_channels)):
            self.voxel_encoders.append(
                nn.ModuleDict({
                    'downsample': SparseConvBlock(
                        feature_channels[k-1],
                        feature_channels[k-1],
                        2,
                        stride=2,
                        indice_key=f"conv_{k}"
                    ),
                    'fuse': nn.Sequential(
                        SparseResBlock(feature_channels[k-1], feature_channels[k], 3, sgc=True),
                        SparseResBlock(feature_channels[k], feature_channels[k], 3, sgc=True),
                    )
                })
            )

        self.voxel_size = [voxel_size * 2**i for i in range(len(feature_channels))]
        self.radius = [math.sqrt(3*vs**2) for vs in self.voxel_size]
        self.point_encoders = nn.ModuleList()
        for k in range(len(point_channels)):
            # kernel_size
            self.point_encoders.append(nn.ModuleDict({
                'feat_extractor': nn.Sequential(
                    nn.Linear(point_in_channels, point_channels[k]//4),
                    # nn.BatchNorm1d(point_channels[k]),
                    nn.ReLU(True),
                    nn.BatchNorm1d(point_channels[k]//4),
                    nn.Linear(point_channels[k]//4, point_channels[k]//2),
                    # nn.BatchNorm1d(point_channels[k]),
                    nn.ReLU(True),
                    nn.BatchNorm1d(point_channels[k]//2),
                    nn.Linear(point_channels[k]//2, point_channels[k]),
                    # nn.BatchNorm1d(point_channels[k]),
                    nn.ReLU(True),
                    nn.BatchNorm1d(point_channels[k]),
                ),
                'point_conv': KPConv(
                    kernel_size = 60,
                    p_dim = 3,
                    in_channels=point_channels[k],
                    out_channels=point_channels[k],
                    KP_extent = self.radius[k]//60,
                    radius=self.radius[k]
                ),
                'act': nn.Sequential(
                    # nn.BatchNorm1d(point_channels[k]),
                    nn.ReLU(True),
                    nn.BatchNorm1d(point_channels[k]),
                ),
            }))

    def forward(
        self, 
        x: SparseConvTensor, 
        point: torch_geometric.data.Data,
    ):
        voxel_feats, pvoxel_feats, point_feats = [], [], []
        for i in range(len(self.voxel_encoders) + 1):
            # voxel future extraction
            if i == 0:
                x = self.stem['feat_extractor'](x)
                fuse = self.stem['fuse']
            else:
                x = self.voxel_encoders[i-1]['downsample'](x)
                fuse = self.voxel_encoders[i-1]['fuse']
            
            # center points query
            _, center_indices = knn(
                torch.floor(point.pos / self.voxel_size[i]).float(),
                x.indices[:, 1:].contiguous().float(),
                1,
                batch_x=point.batch,
                batch_y=x.indices[:, 0].contiguous(),
            )

            # query point in voxel using radius point query
            batch, ids = radius(
                point.pos,
                point.pos[center_indices],
                self.radius[i],
                point.batch,
                point.batch[center_indices],
            )
            neighbors, _ = to_dense_batch(
                ids, batch, fill_value=-1
            )

            # point feature extraction
            pfeats = self.point_encoders[i]['feat_extractor'](point.x)
            
            # point convolution
            num_centers = neighbors.size(0)
            feat_dim = pfeats.size(1)
            neighb_idx = neighbors.view(-1)
            neighb_x = torch.cat((pfeats, torch.zeros_like(pfeats[:1, :])), 0)[neighb_idx, :].view(num_centers, -1, feat_dim)

            point_feat = self.point_encoders[i]['point_conv'](
                point.pos[center_indices],
                point.pos,
                neighbors,
                pfeats[center_indices],
                neighb_x
            )
            point_feat = self.point_encoders[i]['act'](point_feat)
            if torch.all(torch.isnan(point_feat)).item():
                raise Exception('NaN values found!')
            
                
            # fuse voxel and point features
            # x = fuse(feat_cat(x, point_feat))
            x = fuse(x)
            voxel_feats.append(x)
            pvoxel_feats.append(point_feat)
            point_feats.append(pfeats)
            # point_feats.append(spconv.SparseConvTensor(
            #     point_feat,
            #     indices = x.indices,
            #     spatial_shape = x.spatial_shape,
            #     batch_size = x.batch_size
            # ))

        return voxel_feats, point_feats, pvoxel_feats


class SparseResUNet(nn.Module):
    def __init__(
        self,
        point_in_channels: int,
        voxel_in_channels: int,
        stem_channels: int,
        encoder_channels: List[int],
        point_channels: List[int],
        capsule_channels: List[Tuple[int]],
        decoder_channels: List[int],
        voxel_size: int,
    ):
        super(SparseResUNet, self).__init__()
        self.encoder = PointVoxelEncoder(
            point_in_channels,
            voxel_in_channels,
            stem_channels,
            encoder_channels,
            point_channels,
            voxel_size,
        )

        self.capsnet = nn.Sequential()
        for k in range(1, len(capsule_channels)):
            self.capsnet.append(nn.Sequential(
                SparseConvCapsule3d(
                    kernel_size=3,
                    in_num_caps=capsule_channels[k-1][0],
                    out_num_caps=capsule_channels[k][0],
                    in_caps_dim=capsule_channels[k-1][1],
                    out_caps_dim=capsule_channels[k][1],
                    stride=1,
                    padding=1,
                    dilation=1,
                    # num_routing=3,
                    share_weight=True,
                ),
                # SparseBatchNorm(capsule_channels[k][0] * capsule_channels[k][1])
            ))
            
        self.decoders = nn.ModuleList()
        encoder_channels = [stem_channels] + encoder_channels
        decoder_channels = [capsule_channels[-1][0] * capsule_channels[-1][1]] + decoder_channels
        # decoder_channels = [encoder_channels[-1]] + decoder_channels
        for k in range(1, len(decoder_channels)):
            self.decoders.append(
                nn.ModuleDict(
                    {
                        "upsample": SparseConvTransposeBlock(
                            decoder_channels[k - 1],
                            decoder_channels[k],
                            2,
                            stride=2,
                            indice_key=f"conv_{len(decoder_channels) - k}"
                        ),
                        "fuse": nn.ModuleList([
                            SparseResBlock(
                                decoder_channels[k] + encoder_channels[-1-k],
                                decoder_channels[k],
                                3,
                            ),
                            SparseResBlock(
                                decoder_channels[k] + point_channels[-1-k],
                                decoder_channels[k],
                                3,
                            ),
                        ]),
                    }
                )
            )

    def forward(self, x: SparseConvTensor, point: torch_geometric.data.Data):
        encoded_feats, point_feats, pvoxel_feats = self.encoder(x, point)
        decoded_feats = [self.capsnet(encoded_feats[-1])]
        for i, d in enumerate(self.decoders):
            decoded_feats.append(d["fuse"][1](
                feat_cat(
                    d['fuse'][0](cat([
                        d["upsample"](decoded_feats[-1]),
                        encoded_feats[-2-i]
                    ])),
                    pvoxel_feats[-2-i]
                )
            ))
        
        return decoded_feats[0], decoded_feats[1:], point_feats




# class SparseResUNet(nn.Module):
#     def __init__(
#         self,
#         stem_channels: int,
#         encoder_channels: List[int],
#         decoder_channels: List[int],
#         capsule_channels: List[Tuple[int]],
#         *,
#         in_channels: int = 4,
#     ) -> None:
#         super().__init__()
#         # assert len(encoder_channels) == len(decoder_channels) - 1 , 'The number of encoder layers must equal to the number of decoder layers - 1!'
#         self.stem_channels = stem_channels
#         self.encoder_channels = encoder_channels
#         self.decoder_channels = decoder_channels
#         self.capsule_channels = capsule_channels
#         self.in_channels = in_channels


#         self.stem = nn.ModuleDict({
#             'feat_extractor': SparseSequential(
#                 get_conv(in_channels, stem_channels//2, 3),
#                 SparseBatchNorm(stem_channels//2),
#                 nn.ReLU(True),
#                 get_conv(stem_channels//2, stem_channels//2, 3),
#                 SparseBatchNorm(stem_channels//2),
#                 nn.ReLU(True),
#             ),
#             'fuse': get_conv(stem_channels, stem_channels, 1)
#         })

#         self.encoders = nn.ModuleList()
#         encoder_channels = [stem_channels] + encoder_channels
#         for k in range(1, len(encoder_channels)):
#             self.encoders.append(
#                 nn.Sequential(
#                     SparseConvBlock(
#                         encoder_channels[k-1],
#                         encoder_channels[k-1],
#                         2,
#                         stride=2,
#                         indice_key=f"conv_{k}"
#                     ),
#                     SparseResBlock(encoder_channels[k-1], encoder_channels[k], 3),
#                     SparseResBlock(encoder_channels[k], encoder_channels[k], 3),
#                 )
#             )

#         self.capsnet = nn.Sequential()
#         for k in range(1, len(capsule_channels)):
#             self.capsnet.append(
#                 SparseConvCapsule3d(
#                     kernel_size=3,
#                     in_num_caps=capsule_channels[k-1][0],
#                     out_num_caps=capsule_channels[k][0],
#                     in_caps_dim=capsule_channels[k-1][1],
#                     out_caps_dim=capsule_channels[k][1],
#                     stride=1,
#                     padding=0,
#                     dilation=1,
#                     # num_routing=3,
#                     share_weight=True,
#                 )
#             )
            
#         self.decoders = nn.ModuleList()
#         decoder_channels = [capsule_channels[-1][0] * capsule_channels[-1][1]] + decoder_channels
#         # decoder_channels = [encoder_channels[-1]] + decoder_channels
#         for k in range(1, len(decoder_channels)):
#             self.decoders.append(
#                 nn.ModuleDict(
#                     {
#                         "upsample": SparseConvTransposeBlock(
#                             decoder_channels[k - 1],
#                             decoder_channels[k],
#                             2,
#                             stride=2,
#                             indice_key=f"conv_{len(decoder_channels) - k}"
#                         ),
#                         "fuse": nn.Sequential(
#                             SparseResBlock(
#                                 decoder_channels[k] + encoder_channels[-1-k],
#                                 decoder_channels[k],
#                                 3,
#                             ),
#                             SparseResBlock(
#                                 decoder_channels[k],
#                                 decoder_channels[k],
#                                 3,
#                             ),
#                         ),
#                     }
#                 )
#             )
        
        

#     def forward(self, x: SparseConvTensor, point_feats: SparseConvTensor):
#         # feature extraction
#         # print(x.features.shape)
#         encoded_feats = [
#             self.stem['fuse'](feat_cat(
#                 self.stem['feat_extractor'](x), 
#                 point_feats
#             ))
#         ]
        
#         # encoding
#         for e in self.encoders:
#             encoded_feats.append(e(encoded_feats[-1]))
        
#         # print(encoded_feats[-1].feats.shape, encoded_feats[-2].feats.shape)
#         # decoding
#         decoded_feats = [self.capsnet(encoded_feats[-1])]
#         for i, d in enumerate(self.decoders):
#             decoded_feats.append(d["fuse"](
#                 cat([
#                     d["upsample"](decoded_feats[-1]),
#                     encoded_feats[-2-i]
#                 ])
#             ))
        
#         return encoded_feats[:-1], decoded_feats[0], decoded_feats[1:]

#         # return self._unet_forward(encoded_feats[0], self.encoders, self.decoders)
