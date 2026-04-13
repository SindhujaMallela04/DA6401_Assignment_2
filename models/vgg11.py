"""VGG11 encoder
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
# from .layers import CustomDropout


class VGG11Encoder(nn.Module):
    """VGG11-style encoder with optional intermediate feature returns.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        
        #conv 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #conv 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        #conv 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            #Adding dropout
            # CustomDropout(p=0.5),

            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        #conv 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            #Adding dropout
            # CustomDropout(p=0.5),

            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        #conv 5
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            #Adding dropout
            # CustomDropout(p=0.5),

            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: input image tensor [B, 3, H, W].
            return_features: if True, also return skip maps for U-Net decoder.

        Returns:
            - if return_features=False: bottleneck feature tensor.
            - if return_features=True: (bottleneck, feature_dict).
        """
        features = {}

        x = self.conv1(x)
        features["block1"] = x
        x = self.pool1(x)

        x = self.conv2(x)
        features["block2"] = x
        x = self.pool2(x)

        x = self.conv3(x)
        features["block3"] = x
        x = self.pool3(x)

        x = self.conv4(x)
        features["block4"] = x
        x = self.pool4(x)

        #bottleneck
        x = self.conv5(x)
        features["block5"] = x
        x = self.pool5(x)

        if return_features:
            return x, features
        return x
    
VGG11 = VGG11Encoder
#Checking
# x = torch.randn(1, 3, 224, 224)
# model = VGG11Encoder()

# out, feats = model(x, return_features=True)

# print(out.shape)
# for k in feats:
#     print(k, feats[k].shape)