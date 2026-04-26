#!/usr/bin/env python3
import torch
import torchvision.transforms as T
import torchvision.datasets as datasets
from torchvision import models
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import os

DATASET_DIR = os.path.expanduser('~/cat_dataset/train')
VAL_DIR     = os.path.expanduser('~/cat_dataset/val')
OUTPUT_PATH = os.path.expanduser('~/cat_classifier.pth')
EPOCHS      = 30
BATCH_SIZE  = 32
LR          = 0.0005

train_transform = T.Compose([
    T.Resize((256, 256)),
    T.RandomCrop(224),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.RandomRotation(30),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(DATASET_DIR, transform=train_transform)
val_dataset   = datasets.ImageFolder(VAL_DIR,     transform=val_transform)

class_names = train_dataset.classes
print(f'Classes: {class_names}')
print(f'Train samples: {len(train_dataset)}')
print(f'Val samples:   {len(val_dataset)}')

targets       = [s[1] for s in train_dataset.samples]
class_counts  = np.bincount(targets)
class_weights = 1.0 / class_counts
sample_weights = [class_weights[t] for t in targets]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Training on: {device}')

model = models.mobilenet_v2(weights='IMAGENET1K_V1')
model.classifier[1] = nn.Linear(1280, len(class_names))
model = model.to(device)

weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=weight_tensor)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    train_loss, train_correct = 0.0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        train_loss    += loss.item() * imgs.size(0)
        train_correct += (out.argmax(1) == labels).sum().item()

    model.eval()
    val_correct = 0
    per_class_correct = np.zeros(len(class_names))
    per_class_total   = np.zeros(len(class_names))

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out   = model(imgs)
            preds = out.argmax(1)
            val_correct += (preds == labels).sum().item()
            for i in range(len(class_names)):
                mask = labels == i
                per_class_correct[i] += (preds[mask] == labels[mask]).sum().item()
                per_class_total[i]   += mask.sum().item()

    train_acc = train_correct / len(train_dataset)
    val_acc   = val_correct   / len(val_dataset)
    scheduler.step()

    print(f'Epoch {epoch+1:02d}/{EPOCHS} | train_acc={train_acc:.3f} | val_acc={val_acc:.3f}')
    for i, name in enumerate(class_names):
        if per_class_total[i] > 0:
            print(f'  {name}: {per_class_correct[i]/per_class_total[i]:.3f}')

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), OUTPUT_PATH)
        print(f'  ✅ Saved best model (val_acc={val_acc:.3f})')

print(f'\nDone! Best val_acc={best_val_acc:.3f}')
print(f'Class order: {class_names}')
