"""Localization modules
"""

import torch
import torch.nn as nn
from .layers import CustomDropout
from .vgg11 import VGG11Encoder


class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4),  # Output: (x_center, y_center, width, height)
            # nn.Sigmoid()  # Ensuring outputs are in [0, 1] range
        )        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format.
        """
        features = self.encoder(x)
        bbox_coords = self.regressor(features)
        return bbox_coords


#Checking
# if __name__ == "__main__":
#     # x = torch.randn(1, 3, 224, 224)

#     model = VGG11Localizer()
#     # model.eval()

#     # out = model(x)

#     # print("Output shape:", out.shape)
#     # print("Predicted bbox:", out)
#     from data.pets_dataset import OxfordIIITPetDataset

#     dataset = OxfordIIITPetDataset(root="./data")

#     img, _, bbox, _ = dataset[0]

#     img = img.unsqueeze(0)  # add batch dimension

#     pred_bbox = model(img)

#     print("Ground truth:", bbox)
#     print("Prediction:", pred_bbox)