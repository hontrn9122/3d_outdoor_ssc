from itertools import repeat
from typing import List, Tuple, Union
import torch
import torch.nn.functional as F

def ravel_hash(x, xmax = torch.tensor([256, 256, 32])):
    assert x.ndim == 2, x.shape

    h = torch.zeros(x.shape[0], device=x.device)
    
    for k in range(x.shape[1] - 1):
        h += x[:, k]
        h *= xmax[k + 1]
    h += x[:, -1]
    return h

def unique_idx(input):
    _, inverse_indices, counts = torch.unique(
        input, return_inverse=True, return_counts=True, sorted=True
    )
    _, ind_sorted = torch.sort(inverse_indices, stable=True)
    cum_sum = torch.cat((torch.tensor([0]), counts.cumsum(0)[:-1]))
    indices = ind_sorted[cum_sum]
    return indices
    

def sparse_quantize(
    coords,
    sample_coords,
    voxel_size = 0.2,
    *,
    min_extent: List[float] = [0, -25.6, -2]
) -> List[torch.Tensor]:

    min_extent = torch.tensor([min_extent])

    # quantized and hash the point coords for processing
    raw_quantized_coords = (coords - min_extent) / voxel_size
    quantized_coords = torch.floor(raw_quantized_coords)
    hashed_coords = ravel_hash(quantized_coords)

    # sort the indice to choose the point closest to the voxel centroid to be the center point
    sorted_indices = F.pairwise_distance(quantized_coords, raw_quantized_coords, p=2, eps=0).argsort(descending=True)
    sorted_hashed_coords = hashed_coords[sorted_indices]

    # get centroid indices
    center_indices = unique_idx(sorted_hashed_coords)
    indices = sorted_indices[center_indices] # update indices
    
    # sample the nearest point for missing point and drop invalid points
    hashed_sample_coords = ravel_hash(sample_coords)

    # tmp_coords = quantized_coords[indices].unsqueeze(0).expand(sample_coords.size(0), -1, -1)
    tmp_coords = quantized_coords[indices]
    tmp_sample_coords = sample_coords.unsqueeze(1).expand(-1 , tmp_coords.size(0), -1)
    nearest_sampled_id = torch.cat(
      [F.pairwise_distance(sample_coord, tmp_coords, p=2, eps=0).min(dim=0, keepdim=True).indices
      for sample_coord in tmp_sample_coords]
    , dim=0)
    # nearest_sampled_id = F.pairwise_distance(tmp_sample_coords, tmp_coords, p=2, eps=0).min(dim=1).indices
    indices = indices[nearest_sampled_id] # update indices

    coords = coords[indices]
    quantized_coords = quantized_coords[indices]

    return coords, quantized_coords.int(), indices