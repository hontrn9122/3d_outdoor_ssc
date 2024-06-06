import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    MLP
    PointTransformerConv,
    radius_graph,
)


class RadiusPointTransConv(nn.Module):
    def __init__(
        self,
        radius,
        in_channels,
        out_channels,
    ):
        super(RadiusPointTransConv, self).__init__()
        self.radius = radius
        self.lin_in = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LeakyReLU(True),
        )
        self.lin_out = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.LeakyReLU(True),
            nn.BathcNorm1d(out_channels),
        )
        
        self.pos_nn = MLP([3, 64, out_channels], norm=None, plain_last=False)

        self.attn_nn = MLP([out_channels, 64, out_channels], norm=None,
                           plain_last=False)
        
        self.transformer = PointTransformerConv(in_channels, out_channels,
                                                pos_nn=self.pos_nn,
                                                attn_nn=self.attn_nn)
        
    def forward(self, x, pos, batch):
        edge_index = radius_graph(pos, self.radius, batch=batch, loop=False)
        x = self.lin_in(x)
        x = self.transformer(x, pos, edge_index)
        x - self.lin_out(x)
        return x
    
    


    
