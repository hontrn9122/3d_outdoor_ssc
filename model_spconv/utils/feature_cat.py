from typing import List

import torch
from spconv.pytorch import SparseConvTensor

def feat_cat(input: SparseConvTensor, feat: torch.Tensor) -> SparseConvTensor:
    feats = torch.cat([input.features, feat], dim=1)
    return input.replace_feature(feats)

def cat(inputs: List[SparseConvTensor]) -> SparseConvTensor:
    feats = torch.cat([input.features for input in inputs], dim=1)
    return inputs[0].replace_feature(feats)