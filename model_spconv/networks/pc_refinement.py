import torch
import torch.nn as nn
import spconv.pytorch as spconv
import math
from torch_geometric.nn import knn
from torch_geometric.utils import scatter
from model_spconv.utils import ravel_hash, scatter_max


class PCRefinement(nn.Module):
    def __init__(self):
        super(PCRefinement, self).__init__()
        self.pointMLP = nn.ModuleList([
            nn.Sequential(nn.Linear(256 + 4 * 32, 4 * 32), nn.ReLU()),
            nn.Sequential(nn.Linear(256 + 8 * 32, 8 * 32), nn.ReLU()),
            nn.Sequential(nn.Linear(256 + 16 * 32, 16 * 32), nn.ReLU())
        ])
        
    def polar2cube(self, polar, point_feat, point, mlp, pvox_dim, cvox_dim, batch_size, qinfor):
        device = point_feat.device
        pvox_size = (qinfor['cyl_max_extent'] - qinfor['cyl_min_extent']) / pvox_dim
        cvox_size = (qinfor['max_extent'] - qinfor['min_extent']) / cvox_dim
        polar_quantized = torch.floor((point.p_coord - qinfor['cyl_min_extent']) / pvox_size)
        
        p2p = knn(
            polar.indices[:,1:].float().contiguous(), 
            polar_quantized, 
            1, 
            polar.indices[:,0].contiguous(), point.batch
        )[1]
        point_feat = torch.cat([point_feat, polar.features[p2p]], dim=1)
        cube_quantized = torch.cat([
            point.batch.unsqueeze(1),
            torch.floor((point.coord - qinfor['min_extent'].to(device)) / cvox_size)
        ], dim=1)
        
        center_coord, idx_map = torch.unique(
            cube_quantized, sorted=True,
            return_inverse=True, dim=0
        )
        vfeat = scatter_max(point_feat, idx_map, dim=0)
        return spconv.SparseConvTensor(
            mlp(vfeat),
            center_coord.int(),
            cvox_dim[0].tolist(), batch_size,
        )    
        
    def forward(self, point_feat, point, feats, batch_size, qinfor):
        cyl_voxel_dims = qinfor['cyl_voxel_dims']
        voxel_dims = torch.tensor([qinfor['voxel_dims']], device=point_feat.device)
        res = []
        for i in range(len(feats)):
            x = self.polar2cube(
                feats[i], point_feat,
                point, self.pointMLP[i],
                (cyl_voxel_dims-1)/2**(i+1), voxel_dims/2**(i+1),
                batch_size, qinfor
            )
            x = x.dense().permute(0,1,4,2,3)
            x = x.reshape(-1, x.shape[1]*x.shape[2], x.shape[3], x.shape[4])
            res.append(x)
            
        return res