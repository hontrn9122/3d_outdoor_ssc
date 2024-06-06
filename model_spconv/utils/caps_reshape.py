import torch
from spconv.pytorch import SparseConvTensor

def sparse_reduce_dim(input, n_caps, d_caps):
    '''
        This fuction will transform the sparse tensor with features of shape (n, n_caps * d_caps) and
        batch_size = B to a sparse tensor with feature of shape(n*n_caps, d_caps) and batch_size = B*n_caps
    '''
    feats, coords = input.features, input.indices
    n, _ = list(feats.shape)
    batch_size = torch.max(coords[:,0]).item() +1
    batch_const = torch.arange(0, n_caps*batch_size, step=batch_size, device=feats.device, dtype=torch.int).repeat(n)
    feats = feats.view(-1, d_caps)
    coords = coords.repeat_interleave(n_caps, dim=0)
    coords = torch.cat([
        (coords[:,0] + batch_const).unsqueeze(1), 
        coords[:,1:]
      ], dim=1)
    
    output = SparseConvTensor(
        feats, coords, input.spatial_shape,
        batch_size*n_caps, input.grid, 
        input.voxel_num, input.indice_dict
    )
    output.benchmark = input.benchmark
    output.benchmark_record = input.benchmark_record
    output.thrust_allocator = input.thrust_allocator
    output._timer = input._timer
    
    return output, batch_size

def sparse_split_dim(input, in_n_caps, out_n_caps, out_d_caps, batch_size):
    '''
        This fuction will transform the sparse tensor with features of shape (n*in_n_caps, out_n_caps * out_d_caps) and
        batch_size = B*in_n_caps to a sparse tensor with feature of shape(n, in_n_caps, out_n_caps, out_d_caps) and batch_size = B
    '''
    feats, coords = input.features, input.indices
    feats = feats.view(-1, in_n_caps, out_n_caps, out_d_caps)
    coords = coords[coords[:,0] < batch_size]
    
    output = SparseConvTensor(
        input.features, coords, input.spatial_shape,
        batch_size, input.grid, 
        input.voxel_num, input.indice_dict
    )
    output.benchmark = input.benchmark
    output.benchmark_record = input.benchmark_record
    output.thrust_allocator = input.thrust_allocator
    output._timer = input._timer
    return output, feats