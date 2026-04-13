"""Reusable custom layers 
"""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Custom Dropout layer.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()

        if not 0 <= p < 1 :
            raise ValueError(f"Dropout probability must be in the range [0, 1). Got: {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dropout to the input tensor.
        Args:
            x: Input tensor of shape [B, C, H, W].
        Returns:
            Tensor of the same shape as input with dropout applied."""
        
        if not self.training or self.p == 0.0:
            return x
        
        # Creating a dropout mask
        mask = (torch.rand_like(x) > self.p).float()

        # inverted dropout scaling
        return mask * x / (1.0 - self.p)


#Checking
# drop = CustomDropout(p=0.5)
# drop.train()

# x = torch.ones(1000)
# out = drop(x)

# print(out.mean())  # should be ≈ 1