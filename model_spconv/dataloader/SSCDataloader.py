import torch
from torch.utils.data import DataLoader
from model_spconv.utils import from_sparse_info
import math
import KPConv.cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors
from torch_geometric.data import Batch
import spconv.pytorch as spconv
import torch.nn.functional as F

def custom_collate_fn(input):
    file_info, data, qinfor = zip(*input)
    batch_size = len(data)
    # collection = {k: torch.stack([d[k] for d in data]) for k in data[0].keys() if k not in ['point']}
    collection = {k: [] for k in data[0].keys() if k != 'p2v'}
    p2v = []
    for k in data[0].keys():
        for i in range(len(data)): 
            if k == 'cylinder_coord':
                collection[k].append(
                    F.pad(data[i][k], (1, 0), 'constant', value=i)
                )
            elif k == 'p2v':
                p2v.append(
                    data[i][k] if len(p2v) == 0 else
                    data[i][k] + torch.max(p2v[-1]) + 1
                )
            else:
                collection[k].append(data[i][k])
        if k == 'point':
            collection[k] = Batch.from_data_list(collection[k])
        elif k in ['cylinder_coord', 'cylinder_label']:
            collection[k] = torch.cat(collection[k], dim=0)
        elif k != 'p2v':
            collection[k] = torch.stack(collection[k], dim=0)
            
    collection['point'].p2v = torch.cat(p2v, dim=0)
    if 'cylinder_label' in collection:
        collection['cylinder_label'] = spconv.SparseConvTensor(
            collection['cylinder_label'].unsqueeze(1),
            collection['cylinder_coord'].int(),
            qinfor[0]['cyl_voxel_dims'][0].tolist(), batch_size
        ).dense().squeeze(1)

    return list(file_info), collection, batch_size, qinfor[0]