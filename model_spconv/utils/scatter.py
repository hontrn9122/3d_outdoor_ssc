import torch
from torch import Tensor

def broadcast(src: Tensor, ref: Tensor, dim: int) -> Tensor:
    dim = ref.dim() + dim if dim < 0 else dim
    size = ((1, ) * dim) + (-1, ) + ((1, ) * (ref.dim() - dim - 1))
    return src.view(size).expand_as(ref)

def scatter_max(src, index, dim, dim_size=None):
    if isinstance(index, Tensor) and index.dim() != 1:
            raise ValueError(f"The `index` argument must be one-dimensional "
                             f"(got {index.dim()} dimensions)")

    dim = src.dim() + dim if dim < 0 else dim

    if isinstance(src, Tensor) and (dim < 0 or dim >= src.dim()):
        raise ValueError(f"The `dim` argument must lay between 0 and "
                         f"{src.dim() - 1} (got {dim})")
        
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() > 0 else 0
        
    
    size = src.size()[:dim] + (dim_size, ) + src.size()[dim + 1:]
    
    index = broadcast(index, src, dim)
    return src.new_zeros(size).scatter_reduce_(
        dim, index, src, reduce='amax',
        include_self=False)