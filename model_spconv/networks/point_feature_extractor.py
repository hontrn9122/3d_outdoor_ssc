# from typing import List
# import torch
# import torch.nn as nn
# from functools import partial
# from KPConv.models.blocks import KPConv
# import math


# class LocalFeatsExtractor(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: List[int],
#         radius: List[int],
#     ):
#         super(LocalFeatsExtractor, self).__init__()
#         assert len(out_channels) != len(radius)
#         channels = [in_channels] + out_channels
#         self.layers = nn.ModuleList()
#         for i in range(1, len(channels)):
#             self.layers.append(
#                 RadiusPointTransConv(
#                     radius = radius[i-1]
#                     in_channels = channels[i-1]
#                     out_channels = channels[i]
#                 )
#             )
        
#     def forward(self, point_input):
#         x = [point_input.x]
#         for l in self.layers:
#             x.append(l(x, pos, batch))
#         return x[1:]
    

# # class LocalFeatsExtractor(nn.Module):
# #     def  __init__(
# #         self,
# #         in_channels,
# #         out_channels,
# #         mlp_layer=nn.Linear
# #     ):
# #         super(LocalFeatsExtractor, self).__init__()
# #         self.mlp = mlp_layer(
# #             in_channels,
# #             out_channels,
# #         )
# #         radius = math.sqrt(0.03)
# #         self.point_conv = KPConv(
# #             kernel_size = 30,
# #             p_dim = 3,
# #             in_channels=out_channels,
# #             out_channels=out_channels,
# #             KP_extent = radius//2.5,
# #             radius=radius
# #         )
        
# #     def forward(self, point_data):
# #         center_indices = point_data['center_indices']
# #         point_feats = self.mlp(point_data['feats'])

# #         # for k, v in point_data.items():
# #         #   print(f'{k}_shape: ', v.shape)
# #         # print(f'feats_shape', point_feats.shape)
# #         # print(torch.unique(point_data['neighbors']))
# #         # print(torch.unique(center_indices))

# #         # print(point_data['center_coords'].shape)
# #         num_centers = point_data['neighbors'].size(0)
# #         feat_dim = point_feats.size(1)
# #         neighb_idx = point_data['neighbors'].view(-1)
# #         neighb_x = torch.cat((point_feats, torch.zeros_like(point_feats[:1, :])), 0)[neighb_idx, :].view(num_centers, -1, feat_dim)

# #         # print(point_feats.shape)
# #         # print(neighb_x.shape)
        
# #         voxel_feats = self.point_conv(
# #             point_data['center_coords'],
# #             point_data['coords'],
# #             point_data['neighbors'],
# #             point_feats[center_indices],
# #             neighb_x
# #         )
# #         # print(voxel_feats.shape)
# #         # sparse_voxel_feats = SparseTensor(
# #         #     feats=voxel_feats, 
# #         #     coords=point_data['voxel_coords'], 
# #         #     spatial_range=(point_data['voxel_batches'].size(0), 256, 256, 32)
# #         # )
# #         return voxel_feats
        
        
        
        