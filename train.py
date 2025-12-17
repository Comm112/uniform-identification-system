# train.py
import os
import time
import copy
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="dataset", help="dataset folder with train/val/test")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_path", type=str, default="model.pt")
    p.add_argument("--img_size", type=int, default=224)
    return p.parse_args()

def build_model(num_classes, device):
    # EfficientNet-B0
    model = models.efficientnet_b0(pretrained=True)
    # Replace classifier
    in_f = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_f, num_classes)
    model = model.to(device)
    return model

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs, save_path):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total = 0

            loader = dataloaders[phase]
            for inputs, labels in tqdm(loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                total += inputs.size(0)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), save_path)
                print(f"Saved best model to {save_path} (val acc {best_acc:.4f})")

    print(f"Training complete. Best val acc: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model

def main():
    args = get_args()
    data_dir = Path(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # transforms
    train_transforms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    train_dataset = datasets.ImageFolder(data_dir / "train", transform=train_transforms)
    val_dataset = datasets.ImageFolder(data_dir / "val", transform=val_transforms)
    test_dataset = datasets.ImageFolder(data_dir / "test", transform=val_transforms)

    num_classes = len(train_dataset.classes)
    print("Classes:", train_dataset.classes)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}

    model = build_model(num_classes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    model = train_model(model, dataloaders, criterion, optimizer, device, args.epochs, args.save_path)

    # Evaluate on test set
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.4f}")

    # save final model (already saved best)
    torch.save(model.state_dict(), args.save_path)
    print("Saved final model to", args.save_path)

if __name__ == "__main__":
    main()
