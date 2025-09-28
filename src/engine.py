import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple
from torch.amp import autocast


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    mixed_precision: bool,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for images, labels in progress_bar:
        images, labels = (
            images.to(device),
            labels.to(device),
        )

        with autocast(
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            enabled=mixed_precision,
        ):
            # forward pass
            ouputs = model(images)
            loss = loss_fn(ouputs, labels)

        # backward pass and optim
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        # loss.backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler and isinstance(
            scheduler, torch.optim.lr_scheduler.CosineAnnealingLR
        ):
            scheduler.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(ouputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        progress_bar.set_postfix(loss=running_loss / total, acc=correct / total)

    avg_loss = running_loss / total
    avg_acc = correct / total

    return avg_loss, avg_acc


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    mixed_precision: bool,
):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Evaluating")

    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = (
                images.to(device),
                labels.to(device),
            )
            with autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu",
                enabled=mixed_precision,
            ):
                outputs = model(images)
                loss = loss_fn(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            progress_bar.set_postfix(loss=running_loss / total, acc=correct / total)
    avg_loss = running_loss / total
    avg_acc = correct / total
    return avg_loss, avg_acc
