"""Segmentation model
"""

import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder

def conv_block(in_channels, out_channels):
    # Helper function to create a convolutional block.
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),

        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

class VGG11UNet(nn.Module):
    """U-Net style segmentation network.
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3):
        super().__init__()

        self.encoder = VGG11Encoder(in_channels=in_channels)

        #Upsampling layers
        self.upsamp5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = conv_block(512 + 512, 512)

        self.upsamp4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec4 = conv_block(512 + 512, 256)

        self.upsamp3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(256 + 256, 128)

        self.upsamp2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(128 + 128, 64)

        self.upsamp1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(64 + 64, 64)

        # Final Layer
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        bottleneck, features = self.encoder(x, return_features=True)

        #Encoder features
        enc_f1 = features["block1"] # 224x224
        enc_f2 = features["block2"] # 112x112
        enc_f3 = features["block3"] # 56x56
        enc_f4 = features["block4"] # 28x28
        enc_f5 = features["block5"] # 14x14

        #Decoder path
        x = self.upsamp5(bottleneck) # 7 -> 14
        x = torch.cat([x, enc_f5], dim=1) # Skip connection
        x = self.dec5(x)

        x = self.upsamp4(x) # 14 -> 28
        x = torch.cat([x, enc_f4], dim=1) # Skip connection
        x = self.dec4(x)

        x = self.upsamp3(x) # 28 -> 56
        x = torch.cat([x, enc_f3], dim=1) # Skip connection
        x = self.dec3(x)

        x = self.upsamp2(x) # 56 -> 112
        x = torch.cat([x, enc_f2], dim=1) # Skip connection
        x = self.dec2(x)

        x = self.upsamp1(x) # 112 -> 224
        x = torch.cat([x, enc_f1], dim=1) # Skip connection
        x = self.dec1(x)

        logits = self.final_conv(x)

        return logits
