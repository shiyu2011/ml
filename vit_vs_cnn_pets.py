#!/usr/bin/env python3
"""
Compare ViT vs CNN (ResNet50) with timm on Oxford-IIIT Pets.

- Downloads the Oxford-IIIT Pet dataset via torchvision
- Trains a linear classifier (default) or full fine-tune (optional) for each backbone
- Reports Top-1 accuracy for both models

Usage:
    python vit_vs_cnn_pets.py --epochs 3 --batch-size 64 --finetune linear    # fast baseline (default)
    python vit_vs_cnn_pets.py --epochs 10 --batch-size 32 --finetune full     # slower, full fine-tune
"""
import argparse
import os
from pathlib import Path
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import timm
import numpy as np
import glob, json
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def get_files(data_dir, subFolder, run_dir, seed=0, val_ratio=0.2):
        #get all file names of images and labels
    imgs = sorted(glob.glob(os.path.join(data_dir, subFolder, "*.jpg")))
    
    rng = np.random.RandomState(seed)
    idx = np.arange(len(imgs))
    rng.shuffle(idx)
    # Reserve a portion for validation
    v = int(len(imgs) * val_ratio)
    val_idx, train_idx = idx[:v], idx[v:]
    train_files = [{"image": imgs[i]} for i in train_idx]
    val_files = [{"image": imgs[i]} for i in val_idx]
    
    split = {
        "seed": seed,
        "train": [Path(imgs[i]).name for i in train_idx],
        "val": [Path(imgs[i]).name for i in val_idx]
    }
    
    with open(run_dir / "split.json", "w") as f:
        json.dump(split, f, indent=2)
    print(f"save split to {run_dir / 'split.json'}, train {len(train_files)}, val {len(val_files)}")
    return train_files, val_files

class PetDataset(Dataset):
    def __init__(self, files, class_to_idx, transform=None):
        self.files = files
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = self.files[idx]["image"]
        label_name = Path(image_path).name.split("_")[0]
        label = self.class_to_idx[label_name]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

def get_dataloaders(data_root: str, img_size: int = 224, batch_size: int = 64, num_workers: int = 4, val_split: float = 0.15) -> Tuple[DataLoader, DataLoader, int]:
    tfm_train = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tfm_eval = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_files, val_files = get_files(data_root, "images", Path("./"), seed=0, val_ratio=val_split)

    # Derive class mapping from file names (e.g., Abyssinian_1.jpg -> class Abyssinian)
    class_names = sorted({Path(f["image"]).name.split("_")[0] for f in train_files + val_files})
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    num_classes = len(class_names)

    train_dataset = PetDataset(train_files, class_to_idx, transform=tfm_train)
    val_dataset = PetDataset(val_files, class_to_idx, transform=tfm_eval)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, num_classes

def build_model(model_name: str, num_classes: int, pretrained: bool = True, finetune: str = 'linear') -> nn.Module:
    """
    finetune: 'linear' (freeze backbone) or 'full' (fine-tune all)
    """
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    if finetune == 'linear':
        # Freeze all except classifier head
        for name, param in model.named_parameters():
            param.requires_grad = False
        # Unfreeze classifier parameters
        for p in model.get_classifier().parameters(): #last layer is fc for both vit and resnet
            p.requires_grad = True
    elif finetune == 'full':
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError("finetune must be 'linear' or 'full'")
    return model

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    amp_enabled = device.type == 'cuda'
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = model(images)
        pred = outputs.argmax(dim=1)
        correct += (pred == targets).sum().item()
        total   += targets.size(0)
    return 100.0 * correct / total

def train_one(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, epochs: int = 3, lr: float = 5e-4, weight_decay: float = 1e-4, amp: bool = True) -> float:
    if any(p.requires_grad for p in model.parameters()):
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        # Should not happen, but fallback
        params = model.get_classifier().parameters()
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay) #decay is applied on momentum terms
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    amp_enabled = amp and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_val = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0
        t0 = time.time()
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward() #scales the loss, and calls backward() on the scaled loss
            scaler.step(optimizer) #unscales gradients and calls or skips optimizer.step()
            scaler.update() #updates the scale for next iteration
            running += loss.item() * targets.size(0)
            n += targets.size(0)
        train_loss = running / max(1, n)
        val_acc = evaluate(model, val_loader, device)
        best_val = max(best_val, val_acc)
        dt = time.time() - t0
        print(f"Epoch {epoch:02d} | train_loss: {train_loss:.4f} | val@1: {val_acc:.2f}% | time: {dt:.1f}s")
    return best_val

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='data', help='Where to download Oxford-IIIT Pets')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--finetune', type=str, default='linear', choices=['linear', 'full'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--vit', type=str, default='vit_base_patch16_224')
    parser.add_argument('--cnn', type=str, default='resnet50')
    args = parser.parse_args()

    print("Config:", vars(args))
    train_loader, val_loader, num_classes = get_dataloaders(args.data_root, img_size=224, batch_size=args.batch_size, num_workers=args.num_workers)

    # Build models
    vit = build_model(args.vit, num_classes=num_classes, pretrained=True, finetune=args.finetune)
    cnn = build_model(args.cnn, num_classes=num_classes, pretrained=True, finetune=args.finetune)

    device = torch.device(args.device)
    amp_enabled = device.type == 'cuda'
    print(f"AMP: {'enabled' if amp_enabled else 'disabled'}")

    print(f"\nTraining ViT ({args.vit}) [{args.finetune}] ...")
    vit_val = train_one(vit, train_loader, val_loader, device, epochs=args.epochs, amp=amp_enabled)

    print(f"\nTraining CNN ({args.cnn}) [{args.finetune}] ...")
    cnn_val = train_one(cnn, train_loader, val_loader, device, epochs=args.epochs, amp=amp_enabled)

    print("\n=== Results (Val Top-1) ===")
    print(f"ViT  ({args.vit}): {vit_val:.2f}%")
    print(f"CNN  ({args.cnn}): {cnn_val:.2f}%")

if __name__ == '__main__':
    main()
