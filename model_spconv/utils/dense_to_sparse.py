import torch
import spconv.pytorch as spconv
from spconv.pytorch import SparseConvTensor


def dense_to_sparse_info(x: torch.Tensor, channel_last=False):
    """
        create sparse tensor fron pytorch dense tensor
        Parameters:
            x (torch.Tensor): input tensor,
            channel_last (bool): True if the input tensor is channel last, otherwise, False
        Return:
            (torchsparse.SparseTensor): output torchsparse sparse tensor
    """
    if not channel_last:
        permute_id = [0] + [i for i in range(2, x.ndim)] + [1]
        x = x.permute(*permute_id)
    
    x_sp = x.to_sparse(x.ndim-1)
    
    output = {}
    output['batch_size'] = x_sp.shape[0]
    output['spatial_shape'] = x_sp.shape[1:-1]
    output['indices'] = x_sp.indices().transpose(1, 0).contiguous().int()
    output['feats'] = x_sp.values()

    return output

def from_sparse_info(sparse_info):
    return SparseConvTensor(
        sparse_info['feats'],
        sparse_info['indices'],
        sparse_info['spatial_shape'],
        sparse_info['batch_size']
    )
