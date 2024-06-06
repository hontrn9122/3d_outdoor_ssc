import torch
from torch import nn
from torch.nn import functional as F

class MarginLoss(nn.Module):
    def __init__(self, margin=0.4, downweight=0.5, class_weight=None, reduction="mean",):
        super(MarginLoss, self).__init__()
        self.margin = margin
        self.downweight = downweight
        if class_weight is not None:
            self.register_buffer("class_weight", class_weight)
        else:
            self.class_weight = class_weight
        self.reduction = reduction
        
    def forward(self, raw_logits, labels, invalid):
        raw_logits_shape = raw_logits.size()
        num_dims = len(raw_logits_shape)
        if num_dims > 2:
            invalid = invalid.unsqueeze(1)
            invalid = invalid.expand(*list(raw_logits_shape)).view(raw_logits_shape[0], raw_logits_shape[1], -1)
            mask = invalid==0
            raw_logits = raw_logits.view(raw_logits_shape[0], raw_logits_shape[1], -1)[mask]
            labels = labels.view(raw_logits_shape[0], raw_logits_shape[1], -1)[mask]
        logits = raw_logits - 0.5
        positive_cost = labels * F.relu(self.margin - logits) ** 2
        negative_cost = (1 - labels) * F.relu(logits + self.margin) ** 2
        if self.class_weight is not None:
            if num_dims > 2:
                loss = (
                    torch.sum(
                        self.class_weight[None, :, None]
                        * (0.5 * positive_cost + self.downweight * 0.5 * negative_cost),
                        dim=1,
                    )
                    / torch.sum(self.class_weight)
                )
            else:
                loss = torch.sum(
                    self.class_weight[None, :] * (0.5 * positive_cost + self.downweight * 0.5 * negative_cost), dim=1
                ) / torch.sum(self.class_weight)
        else:
            loss = torch.sum(0.5 * positive_cost + self.downweight * 0.5 * negative_cost, dim=1)

        if self.reduction == "mean":
            return torch.mean(loss)
        else:
            pass