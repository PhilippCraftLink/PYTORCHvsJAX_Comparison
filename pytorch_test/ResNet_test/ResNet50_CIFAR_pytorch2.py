import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import wandb
from tqdm import tqdm

# --- 1) Konfiguration ---
BATCH_SIZE = 128
LEARNING_RATE = 1e-3  # 0.001
WEIGHT_DECAY = 5e-4
EPOCHS = 20
NUM_CLASSES = 10
SEED = 42

# --- 2) Modell-Architektur ---
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        # Conv1: 1x1
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # Conv2: 3x3
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # Conv3: 1x1 
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet50CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50CIFAR, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # ResNet Layers
        self.layer1 = self._make_layer(Bottleneck, 64, 3, stride=1)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        self.linear = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # Global Average Pooling 
        out = F.avg_pool2d(out, 4) 
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# --- 3) Data Pipeline ---

def get_dataloaders(batch_size):
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    
    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])
    
    # Im JAX Skript wurde KEIN RandomFlip/Crop für CIFAR verwendet, nur Resize & Norm.
    # Um exakt vergleichbar zu bleiben, nutzen wir hier auch keine Augmentation.
    # (Normalerweise würde man RandomCrop/Flip nutzen).

    os.makedirs("./data", exist_ok=True)
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    return trainloader

# --- 4) Main ---

def seed_everything(seed):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def main():
    seed_everything(SEED)
    
    # Setup WandB
    wandb.init(
        project="cifar10-resnet50_FINAL",
        name=f"pytorch_resnet50_bs{BATCH_SIZE}",
        config={
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "framework": "pytorch",
            "model": "ResNet50-CIFAR"
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader = get_dataloaders(BATCH_SIZE)
    
    # Model
    model = ResNet50CIFAR(NUM_CLASSES).to(device)
    
    # Optimizer
    # (L2 Penalty auf Gradients).
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    criterion = nn.CrossEntropyLoss()

    print("Starte Training...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        processed_imgs = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        
        for images, labels in pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            processed_imgs += images.size(0)
            
        epoch_time = time.time() - start_time
        avg_loss = running_loss / len(train_loader)
        throughput = processed_imgs / epoch_time
        
        # Memory Stats
        mem_alloc = torch.cuda.memory_allocated()
        mem_reserved = torch.cuda.memory_reserved()

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Zeit: {epoch_time:.2f}s | Throughput: {throughput:.1f} img/s")
        
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": avg_loss,
            "epoch/time_s": epoch_time,
            "system/throughput_img_per_sec": throughput,
            "gpu/mem_allocated": mem_alloc,
            "gpu/mem_reserved": mem_reserved
        })

    wandb.finish()
    print("Training abgeschlossen.")

if __name__ == "__main__":
    main()