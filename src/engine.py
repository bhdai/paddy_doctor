import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple, Optional
from torch.amp import autocast
import timm
from timm.scheduler.cosine_lr import CosineLRScheduler


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    mixed_precision: bool,
    scheduler: Optional[CosineLRScheduler],
    num_updates: int,
    mixup_fn=None,
) -> Tuple[float, float, int]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = (
            images.to(device),
            labels.to(device),
        )

        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        with autocast(
            device_type=device.type,
            enabled=mixed_precision,
        ):
            # forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

        # backward pass and optim
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        # loss.backward()
        scaler.step(optimizer)
        scaler.update()

        num_updates += 1
        if scheduler is not None:
            scheduler.step_update(num_updates=num_updates)

        if batch_idx % 200 == 0:
            try:
                lrs = [pg["lr"] for pg in optimizer.param_groups]
            except Exception:
                lrs = None
            grad_norms = []
            for pg in optimizer.param_groups:
                total_norm_sq = 0.0
                for p in pg["params"]:
                    if p.grad is not None:
                        total_norm_sq += float(p.grad.data.norm(2).item()) ** 2
                grad_norms.append(total_norm_sq**0.5)
            print(f" batch {batch_idx}: lrs={lrs}, grad_norms={grad_norms}")

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        if labels.ndim == 2:  # if label are soft fine the argmax
            true_labels = labels.argmax(dim=1)
        else:
            true_labels = labels
        correct += (preds == true_labels).sum().item()
        total += labels.size(0)

        progress_bar.set_postfix(loss=running_loss / total, acc=correct / total)

    avg_loss = running_loss / total
    avg_acc = correct / total

    return avg_loss, avg_acc, num_updates


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
            images, labels = images.to(device), labels.to(device)
            with autocast(
                device_type=device.type,
                enabled=mixed_precision,
            ):
                outputs = model(images)
                loss = loss_fn(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            progress_bar.set_postfix(loss=running_loss / total, acc=correct / total)
    return running_loss / total, correct / total


def evaluate_with_tta(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    mixed_precision: bool,
    num_tta: int = 5,
    eps: float = 1e-12,
) -> Tuple[float, float]:
    model.eval()
    sum_probs = None
    targets = None

    with torch.no_grad():
        for t in tqdm(range(num_tta), desc=f"TTA Evaluation ({num_tta} runs)"):
            run_preds = []
            run_targets = []

            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)

                with autocast(
                    device_type=device.type,
                    enabled=mixed_precision,
                ):
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)

                run_preds.append(probs.float())
                if targets is None:
                    run_targets.append(labels)

            run_preds = torch.cat(run_preds, dim=0)

            if sum_probs is None:
                sum_probs = run_preds
            else:
                sum_probs = sum_probs + run_preds

            if targets is None:
                targets = torch.cat(run_targets, dim=0).to(device)

        avg_preds = sum_probs / float(num_tta)

        avg_preds = avg_preds.clamp(min=eps)

        final_class_preds = torch.argmax(avg_preds, dim=1)
        correct = (final_class_preds == targets).sum().item()
        total = targets.size(0)
        final_acc = correct / total

        log_probs = torch.log(avg_preds)
        final_loss = loss_fn(log_probs, targets).item()

    return final_loss, final_acc
