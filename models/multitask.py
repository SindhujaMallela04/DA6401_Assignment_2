"""Unified multi-task model
"""

import torch
import torch.nn as nn
import gdown
from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet

class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, classifier_path: str = "classifier.pth", localizer_path: str = "localizer.pth", unet_path: str = "unet.pth"):
        super().__init__()

        gdown.download(id="1bRel3MX-GAFydQO8lFCuAp5YeziI8nx6", output=classifier_path, quiet=False)
        gdown.download(id="1nedveiXu8LqMc7_F-bg7_qfUw-Krkx73", output=localizer_path, quiet=False)
        gdown.download(id="1wo4W1KtkgvW-SVEgyuPSRvWjUxqEd6Xv", output=unet_path, quiet=False)

        #Loading models
        self.classifier_model = VGG11Classifier(num_classes = num_breeds)
        self.localizer_model = VGG11Localizer()
        self.unet_model = VGG11UNet(num_classes = seg_classes)

        classifier_ckpt = torch.load(classifier_path, map_location="cpu")
        localizer_ckpt = torch.load(localizer_path, map_location="cpu")
        unet_ckpt = torch.load(unet_path, map_location="cpu")

        classifier_weights = classifier_ckpt["state_dict"] if "state_dict" in classifier_ckpt else classifier_ckpt
        localizer_weights = localizer_ckpt["state_dict"] if "state_dict" in localizer_ckpt else localizer_ckpt
        unet_weights = unet_ckpt["state_dict"] if "state_dict" in unet_ckpt else unet_ckpt

        self.classifier_model.load_state_dict(classifier_weights)
        self.localizer_model.load_state_dict(localizer_weights)
        self.unet_model.load_state_dict(unet_weights)


        #Shared Encoder
        self.encoder = self.classifier_model.encoder

        #Using trained heads
        self.classifier = self.classifier_model.classifier
        self.localizer = self.localizer_model.regressor
        # self.unet = self.unet_model       
        self.unet_upsamp5 = self.unet_model.upsamp5
        self.unet_dec5 = self.unet_model.dec5
        self.unet_upsamp4 = self.unet_model.upsamp4
        self.unet_dec4 = self.unet_model.dec4
        self.unet_upsamp3 = self.unet_model.upsamp3
        self.unet_dec3 = self.unet_model.dec3
        self.unet_upsamp2 = self.unet_model.upsamp2
        self.unet_dec2 = self.unet_model.dec2
        self.unet_upsamp1 = self.unet_model.upsamp1
        self.unet_dec1 = self.unet_model.dec1
        self.unet_final_conv = self.unet_model.final_conv

    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """
        #Encoder
        bottleneck, features = self.encoder(x, return_features=True)

        #Classification
        class_out = self.classifier(bottleneck)

        #Localization
        localization_out = self.localizer(bottleneck) * 224.0
        # bbox = self.localizer(bottleneck)

        enc_f1 = features["block1"]   # 224×224
        enc_f2 = features["block2"]   # 112×112
        enc_f3 = features["block3"]   #  56×56
        enc_f4 = features["block4"]   #  28×28
        enc_f5 = features["block5"]   #  14×14
 
        s = self.unet_upsamp5(bottleneck)           # 7  → 14
        s = torch.cat([s, enc_f5], dim=1)
        s = self.unet_dec5(s)
 
        s = self.unet_upsamp4(s)                    # 14 → 28
        s = torch.cat([s, enc_f4], dim=1)
        s = self.unet_dec4(s)
 
        s = self.unet_upsamp3(s)                    # 28 → 56
        s = torch.cat([s, enc_f3], dim=1)
        s = self.unet_dec3(s)
 
        s = self.unet_upsamp2(s)                    # 56 → 112
        s = torch.cat([s, enc_f2], dim=1)
        s = self.unet_dec2(s)
 
        s = self.unet_upsamp1(s)                    # 112 → 224
        s = torch.cat([s, enc_f1], dim=1)
        s = self.unet_dec1(s)
 
        segmentation_out = self.unet_final_conv(s)


        return {
            "classification": class_out,
            "localization": localization_out,
            "segmentation": segmentation_out
        }


# if __name__ == "__main__":
#     model = MultiTaskPerceptionModel()

#     x = torch.randn(1, 3, 224, 224)
#     out = model(x)

#     print(out["classification"].shape)  # [1, 37]
#     print(out["localization"].shape)   # [1, 4]
#     print(out["segmentation"].shape)   # [1, 1, 224, 224]