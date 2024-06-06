import torch
import torch.nn as nn
import spconv.pytorch as spconv
from torch_geometric.utils import scatter


def batch_split(input):
    v, batch_idx = torch.unique(input.indices[:,0], sorted=True, return_inverse=True)
    outputs = []
    spatial_shape = input.spatial_shape
    for i in range(v.size(0)):
        feat = input.features[batch_idx==i]
        indice = input.indices[batch_idx==i]
        indice = torch.cat([indice[:, 0:1]*0, indice[:,1:]], dim=1)
        outputs.append(spconv.SparseConvTensor(feat, indice, spatial_shape, 1))
    return outputs


def batch_merge(inputs):
    out_feats = []
    out_idx = []
    batch_size = len(inputs)
    for i in range(batch_size):
        out_feats.append(inputs[i].features)
        idx = inputs[i].indices
        idx = torch.cat([idx[:,0:1] + i, idx[:,1:]], dim=1)
        out_idx.append(idx)
    return spconv.SparseConvTensor(
        torch.cat(out_feats, dim=0),  
        torch.cat(out_idx, dim=0),
        inputs[0].spatial_shape,
        batch_size
    )
        
        