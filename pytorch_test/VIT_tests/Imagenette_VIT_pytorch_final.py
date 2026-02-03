import os
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from transformers import ViTForImageClassification, ViTConfig, get_cosine_schedule_with_warmup
import wandb

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def get_args():
    parser = argparse.ArgumentParser(description="PyTorch ViT Training (Aligned)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project_name", type=str, default="vit-benchmark")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--log_freq", type=int, default=5)
    parser.add_argument("--compile_mode", type=str, default="default")
    return parser.parse_args()

def get_dataloaders(data_dir, batch_size, num_workers=4):
    # Imagenette Transform (Standard)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)), # Resize fix auf 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    os.makedirs(data_dir, exist_ok=True)
    try:

        train_dataset = torchvision.datasets.Imagenette(root=data_dir, split='train', size='full', download=True, transform=train_transform)
    except:
        # Fallback
        train_dataset = torchvision.datasets.Imagenette(root=data_dir, split='train', download=True, transform=train_transform)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True, 
        drop_last=True, 
        persistent_workers=True
    )
    return train_loader

def main():
    args = get_args()
    seed_everything(args.seed)
    
    torch.set_float32_matmul_precision('high')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    wandb.init(
        project=args.project_name, 
        name=f"PyTorch_ViT_BS{args.batch_size}",
        config=vars(args),
        tags=["pytorch", "vit"]
    )

    # 1. Dataset
    train_loader = get_dataloaders(args.data_dir, args.batch_size)

    # 2. Model (Hugging Face)
    config = ViTConfig(
        image_size=224,
        patch_size=16,
        num_channels=3,
        num_labels=10, # Imagenette hat 10 Klassen
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        dropout_rate=0.0,
        attention_probs_dropout_rate=0.0,
        initializer_range=0.02,
        return_dict=True
    )
    
    # Init Weights random, -> like JAX
    model = ViTForImageClassification(config)
    model.to(device)

    print(f"Compiling model with mode: {args.compile_mode}...")
    model = torch.compile(model, mode=args.compile_mode)
    
    # 3. Optimizer 
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    
    # Hugging Face Scheduler 
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    scaler = GradScaler()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    global_step = 0
    total_start_time = time.time()
    
    print("Starting training...")
    
    for epoch in range(args.epochs):
        model.train()
        step_times = []
        
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                # HF Model Output ist ein Objekt, wir brauchen .logits
                outputs = model(pixel_values=images)
                loss = nn.CrossEntropyLoss()(outputs.logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            
            step_duration = t1 - t0

            if global_step == 0:

                wandb.log({
                    "system/compilation_time_seconds": step_duration,
                    "global_step": global_step
                })
                print(f"\n[Info] Step 0 (Compilation) Time: {step_duration:.4f}s")
            else:
                step_times.append(step_duration)
            
            if global_step > 0:
                step_times.append(step_duration)

            if global_step % args.log_freq == 0:
                avg_exec = np.mean(step_times) if step_times else step_duration
                step_times = [] 
                
                loss_val = loss.item()
                lr_val = optimizer.param_groups[0]['lr']
                mem_alloc = torch.cuda.memory_allocated()
                max_mem_alloc = torch.cuda.max_memory_allocated()
                
                _, pred = outputs.logits.max(1)
                acc_val = pred.eq(labels).sum().item() / labels.size(0)

                wandb.log({
                    "train/loss": loss_val,
                    "train/accuracy": acc_val,
                    "train/learning_rate": lr_val,
                    "system/execution_time_seconds": avg_exec,
                    "gpu/mem_allocated": mem_alloc,
                    "gpu/max_mem_allocated": max_mem_alloc, # Vergleichswert
                    "global_step": global_step,
                    "epoch": epoch + 1
                }, step=global_step)
            
            global_step += 1

    total_time = time.time() - total_start_time
    avg_throughput = (global_step * args.batch_size) / total_time
    print(f"Training finished in {total_time:.2f} seconds.")
    wandb.log({"system/throughput_img_per_sec": avg_throughput})
    wandb.finish()

if __name__ == "__main__":
    main()