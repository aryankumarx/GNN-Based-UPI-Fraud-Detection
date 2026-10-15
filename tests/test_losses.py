import pytest
import torch

def test_focal_loss_forward():
    from src.training.losses import FocalLoss
    criterion = FocalLoss(alpha=0.85, gamma=2.0)
    
    # Batch of 4 predictions
    logits = torch.tensor([[2.5, -2.5], [-1.0, 1.0], [0.5, -0.5], [-2.0, 2.0]], requires_grad=True)
    targets = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    
    loss = criterion(logits, targets)
    assert loss.dim() == 0  # Scalar loss
    assert loss.item() > 0
    
    # Backward pass check
    loss.backward()
    assert logits.grad is not None
