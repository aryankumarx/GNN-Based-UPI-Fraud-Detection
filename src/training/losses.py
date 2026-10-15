"""
Custom Loss Functions for Imbalanced Graph Classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss addresses class imbalance by down-weighting easy negative examples.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    For fraud detection, alpha represents the weight for the positive (fraud) class.
    """

    def __init__(self, alpha: float = 0.80, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: logits [N, C] or probabilities
        # targets: class labels [N]
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        
        # alpha for minority fraud class (target==1), (1 - alpha) for majority normal class
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        focal_loss = alpha_t * ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
