import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from losses.iou_loss import IoULoss

IMAGE_SIZE = 224.0

def train(task="classification", epochs=10, batch_size=8, lr=1e-4, freeze_encoder=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    dataset = OxfordIIITPetDataset(root="./data", split="trainval")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

   
    if task == "classification":
        model = VGG11Classifier().to(device)
        loss_fn = nn.CrossEntropyLoss()

        #Loading existing checkpoint for resuming training
        if os.path.exists("classifier.pth"):
            print("Resuming from checkpoint...")
            ckpt = torch.load("classifier.pth", map_location="cpu", weights_only = False)
            weights = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            model.load_state_dict(weights)

    elif task == "localization":
        model = VGG11Localizer().to(device)
        
        # Loading pretrained encoder from classifier
        if os.path.exists("classifier.pth"):
            print("Loading pretrained encoder from classifier.pth")
            classifier = VGG11Classifier()
            ckpt = torch.load("classifier.pth", map_location="cpu", weights_only = False)
            classifier_weights = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            classifier.load_state_dict(classifier_weights)
            model.encoder.load_state_dict(classifier.encoder.state_dict())

        #RESUME LOCALIZATION TRAINING
        if os.path.exists("localizer.pth"):
            print("Resuming localization from checkpoint...")
            ckpt = torch.load("localizer.pth", map_location="cpu", weights_only = False)
            loc_weights = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            model.load_state_dict(loc_weights)

        # Freeze encoder if needed
        if freeze_encoder:
            print("Freezing encoder...")
            for param in model.encoder.parameters():
                param.requires_grad = False

        loss_fn_iou = IoULoss()
        loss_fn_mse = nn.MSELoss()

    elif task == "segmentation":
        model = VGG11UNet(num_classes=3).to(device)

        # Loading pretrained encoder
        if os.path.exists("classifier.pth"):
            print("Loading pretrained encoder from classifier...")
            classifier = VGG11Classifier()
            ckpt = torch.load("classifier.pth", map_location="cpu", weights_only = False)
            classifier_weights = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            classifier.load_state_dict(classifier_weights)
            model.encoder.load_state_dict(classifier.encoder.state_dict())

        if os.path.exists("unet.pth"):
            print("Resuming segmentation from checkpoint")
            ckpt = torch.load("unet.pth", map_location = "cpu", weights_only = False)
            seg_weights = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            model.load_state_dict(seg_weights)

        if freeze_encoder:
            print("Freezing encoder...")
            for param in model.encoder.parameters():
                param.requires_grad = False

        loss_fn = nn.CrossEntropyLoss()

    else:
        raise ValueError("Invalid task. Choose from classification | localization | segmentation")

    # Optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay = 1e-4)
    

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        correct = 0
        total = 0

        for images, labels, bboxes, masks in dataloader:

            images = images.to(device)
            labels = labels.long().to(device)
            bboxes = bboxes.to(device)
            masks = masks.to(device).long()

            optimizer.zero_grad()

            
            if task == "classification":
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                #Tracking accuracy
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            elif task == "localization":
                outputs = model(images)

                # outputs_norm = outputs / IMAGE_SIZE
                # bboxes_norm = bboxes / IMAGE_SIZE
                
                mse_loss = loss_fn_mse(outputs, bboxes)
                iou_loss = loss_fn_iou(outputs, bboxes)
                loss = 0.5 * mse_loss / (224.0 ** 2) + iou_loss

            elif task == "segmentation":
                outputs = model(images)   # [B, 3, H, W]
                loss = loss_fn(outputs, masks)

            # Backprop
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if task == "classification":
            acc = correct / total
            print(f"[{task.upper()}] Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f} | Acc: {acc:.4f}")
        else:
            print(f"[{task.upper()}] Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")

    save_path = f"{task}.pth"

    checkpoint = {
        "state_dict" : model.state_dict(),
        "epoch" : epochs,
        "best_metric" : total_loss / len(dataloader)
    }

    torch.save(checkpoint, save_path)
    print(f"Saved {save_path}")

#Main
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True,
                        choices=["classification", "localization", "segmentation"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze_encoder", action="store_true")

    args = parser.parse_args()

    train(
        task=args.task,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        freeze_encoder=args.freeze_encoder
    )