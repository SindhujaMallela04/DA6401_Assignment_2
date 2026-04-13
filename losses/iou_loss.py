"""Custom IoU loss 
"""

import torch
import torch.nn as nn

class IoULoss(nn.Module):
    """IoU loss for bounding box regression.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        self.eps = eps

        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Invalid reduction: {reduction}")
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.
        Args:
            pred_boxes: [B, 4] predicted boxes in (x_center, y_center, width, height) format.
            target_boxes: [B, 4] target boxes in (x_center, y_center, width, height) format."""
        
        
        pred_xmin = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        pred_ymin = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        pred_xmax = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        pred_ymax = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

        target_xmin = target_boxes[:, 0] - target_boxes[:, 2] / 2
        target_ymin = target_boxes[:, 1] - target_boxes[:, 3] / 2
        target_xmax = target_boxes[:, 0] + target_boxes[:, 2] / 2
        target_ymax = target_boxes[:, 1] + target_boxes[:, 3] / 2

        #Intersection
        inter_xmin = torch.max(pred_xmin, target_xmin)
        inter_ymin = torch.max(pred_ymin, target_ymin)
        inter_xmax = torch.min(pred_xmax, target_xmax)
        inter_ymax = torch.min(pred_ymax, target_ymax)

        inter_w = torch.clamp(inter_xmax - inter_xmin, min=0)
        inter_h = torch.clamp(inter_ymax - inter_ymin, min=0)

        intersection = inter_w * inter_h

        #Areas
        pred_area = (pred_xmax - pred_xmin) * (pred_ymax - pred_ymin)
        target_area = (target_xmax - target_xmin) * (target_ymax - target_ymin)

        #Union
        union = pred_area + target_area - intersection

        #Calculating IOU
        iou = intersection / (union + self.eps)
        iou = torch.clamp(iou, min=0.0, max=1.0)

        #Calculating Loss
        loss = 1 - iou

        #Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
        