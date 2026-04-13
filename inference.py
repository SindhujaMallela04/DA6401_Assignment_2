"""Inference and evaluation
"""

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from data.pets_dataset import OxfordIIITPetDataset
from models.multitask import MultiTaskPerceptionModel



def compute_iou(pred, target):
    pred_x1 = pred[:,0] - pred[:,2]/2
    pred_y1 = pred[:,1] - pred[:,3]/2
    pred_x2 = pred[:,0] + pred[:,2]/2
    pred_y2 = pred[:,1] + pred[:,3]/2

    target_x1 = target[:,0] - target[:,2]/2
    target_y1 = target[:,1] - target[:,3]/2
    target_x2 = target[:,0] + target[:,2]/2
    target_y2 = target[:,1] + target[:,3]/2

    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)

    union = pred_area + target_area - inter_area
    return inter_area / (union + 1e-6)


def dice_score_multiclass(pred, target, num_classes=3):
    pred = torch.argmax(pred, dim=1)

    dice = 0
    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()

        dice += (2 * intersection) / (union + 1e-6)

    return dice / num_classes



def evaluate():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset = OxfordIIITPetDataset(root="./data", split="test")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    # Model
    model = MultiTaskPerceptionModel().to(device)
    model.eval()

    all_preds, all_labels = [], []
    all_ious = []
    all_dice = []

    with torch.no_grad():
        for images, labels, bboxes, masks in dataloader:

            images = images.to(device)
            labels = labels.to(device)
            bboxes = bboxes.to(device)
            masks = masks.to(device)

            outputs = model(images)

            # -------------------------
            # Classification
            # -------------------------
            preds = outputs["classification"].argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # -------------------------
            # Localization
            # -------------------------
            iou = compute_iou(outputs["localization"], bboxes)
            all_ious.extend(iou.cpu().numpy())

            # -------------------------
            # Segmentation
            # -------------------------
            dice = dice_score_multiclass(outputs["segmentation"], masks)
            all_dice.append(dice.item())


    f1 = f1_score(all_labels, all_preds, average='macro')
    mean_iou = sum(all_ious) / len(all_ious)
    mean_dice = sum(all_dice) / len(all_dice)

    print("\n===== FINAL METRICS =====")
    print(f"Macro F1 Score: {f1:.4f}")
    print(f"Mean IoU (proxy mAP): {mean_iou:.4f}")
    print(f"Dice Score: {mean_dice:.4f}")


if __name__ == "__main__":
    evaluate()