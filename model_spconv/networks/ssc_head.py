from typing import List

import torch
import torch.nn as nn

import torch.nn.functional as F

from model_spconv.layers import SSCHeadBlock

class DenseSSCHead(nn.Module):
    def __init__(
        self,
        feature_channels: List[int],
        num_classes: int,
    ):
        super(DenseSSCHead, self).__init__()
        self.num_heads = len(feature_channels)
        self.num_classes = num_classes
        self.heads = nn.ModuleList()
        
        for c in feature_channels:
            self.heads.append(SSCHeadBlock(
                c,
                num_classes,
                # c//2,
            ))
            
    def decode(self, feat, layer_idx):
        return self.heads[layer_idx](feat)
    
    def decode_all(self, feats):
        outputs = []
        for i, feat in enumerate(feats):
            outputs.append(self.heads[i](feat))
        return outputs[::-1]
    
    def forward(self, input):
        return F.softmax(self.heads[-1](input), dim=1)
            
        
class PointTrainHead(nn.Module):
    def __init__(
        self,
        feature_channels: List[int],
        num_classes: int,
    ):
        
        super(PointTrainHead, self).__init__()
        self.num_heads = len(feature_channels)
        self.num_class = num_classes
        self.heads = nn.ModuleList()
        
        for c in feature_channels:
            self.heads.append(
                nn.Sequential(
                    nn.Linear(c, 32),
                    nn.ReLU(True),
                    nn.Linear(32, num_classes),
                    # nn.Softmax(dim=1),
                )
            )
        
    def forward(self, feats):
        outputs = []
        for i, feat in enumerate(feats):
            outputs.append(self.heads[i](feat))
        return outputs
    
    
        