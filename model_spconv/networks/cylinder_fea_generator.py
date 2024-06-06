# -*- coding:utf-8 -*-
# author: Xinge

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numba as nb
import multiprocessing
import torch_scatter
from torch_geometric.utils import scatter
import spconv.pytorch as spconv
import torch_scatter
from model_spconv.utils import scatter_max

class cylinder_fea(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        compr_channels,
    ):
        super(cylinder_fea, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm1d(in_channels),

            nn.Linear(in_channels, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),

            nn.Linear(256, out_channels)
        )
        if compr_channels is not None:
            self.cyl_fea_compression = nn.Sequential(
                nn.Linear(out_channels, compr_channels),
                nn.LeakyReLU(inplace=True)
            )
            self.vox_fea_compression = nn.Sequential(
                nn.Linear(out_channels, compr_channels),
                nn.LeakyReLU(inplace=True)
            )
        else:
            self.fea_conpression = nn.Indentity()
    
    def forward(self, point, cylinder_coord, batch_size, qinfor, stage):
        # print(torch.min(point.coord, dim=0))
        feat = self.layer(point.x)
        cfeat = scatter_max(feat, point.p2v, dim=0)
        cfeat = self.cyl_fea_compression(cfeat)
        
        cylinder = spconv.SparseConvTensor(
            cfeat, cylinder_coord.int(),
            qinfor['cyl_voxel_dims'][0].tolist(), batch_size,
        )
        
        res = [cylinder, feat]
        if stage != 1:
            vccoord, p2v_map = torch.unique(
                torch.cat([point.batch.unsqueeze(1), point.q_coord], dim=1), sorted=True, 
                return_inverse=True, dim=0
            )
            vfeat = scatter_max(feat, p2v_map, dim=0)
            vfeat = self.vox_fea_compression(vfeat)

            voxel = spconv.SparseConvTensor(
                vfeat, vccoord.int(),
                qinfor['voxel_dims'], batch_size,
            ).dense()
            
            res.append(voxel)
        return res


# class cylinder_fea(nn.Module):

#     def __init__(self, grid_size, fea_dim=3,
#                  out_pt_fea_dim=64, max_pt_per_encode=64, fea_compre=None):
#         super(cylinder_fea, self).__init__()

#         self.PPmodel = nn.Sequential(
#             nn.BatchNorm1d(fea_dim),

#             nn.Linear(fea_dim, 64),
#             nn.BatchNorm1d(64),
#             nn.LeakyReLU(inplace=True),

#             nn.Linear(64, 128),
#             nn.BatchNorm1d(128),
#             nn.LeakyReLU(inplace=True),

#             nn.Linear(128, 256),
#             nn.BatchNorm1d(256),
#             nn.LeakyReLU(inplace=True),

#             nn.Linear(256, out_pt_fea_dim)
#         )

#         self.max_pt = max_pt_per_encode
#         self.fea_compre = fea_compre
#         self.grid_size = grid_size
#         kernel_size = 3
#         # self.local_pool_op = torch.nn.MaxPool2d(kernel_size, stride=1,
#         #                                         padding=(kernel_size - 1) // 2,
#         #                                         dilation=1)
#         self.pool_dim = out_pt_fea_dim

#         # point feature compression
#         if self.fea_compre is not None:
#             self.fea_compression = nn.Sequential(
#                 nn.Linear(self.pool_dim, self.fea_compre),
#                 nn.LeakyReLU(inplace=True))
#             self.pt_fea_dim = self.fea_compre
#         else:
#             self.pt_fea_dim = self.pool_dim

#     def forward(self, pt_fea, xy_ind):
#         cur_dev = pt_fea[0].get_device()

#         # concate everything
#         cat_pt_ind = []
#         for i_batch in range(len(xy_ind)):
#             cat_pt_ind.append(F.pad(xy_ind[i_batch], (1, 0), 'constant', value=i_batch))

#         cat_pt_fea = torch.cat(pt_fea, dim=0)
#         cat_pt_ind = torch.cat(cat_pt_ind, dim=0)
#         pt_num = cat_pt_ind.shape[0]

#         # shuffle the data
#         shuffled_ind = torch.randperm(pt_num, device=cur_dev)
#         cat_pt_fea = cat_pt_fea[shuffled_ind, :]
#         cat_pt_ind = cat_pt_ind[shuffled_ind, :]

#         # unique xy grid index
#         unq, unq_inv, unq_cnt = torch.unique(cat_pt_ind, return_inverse=True, return_counts=True, dim=0)
#         unq = unq.type(torch.int64)

#         # process feature
#         processed_cat_pt_fea = self.PPmodel(cat_pt_fea)
#         pooled_data = torch_scatter.scatter_max(processed_cat_pt_fea, unq_inv, dim=0)[0]

#         if self.fea_compre:
#             processed_pooled_data = self.fea_compression(pooled_data)
#         else:
#             processed_pooled_data = pooled_data

#         return unq, processed_pooled_data
