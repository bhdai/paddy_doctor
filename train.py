import os
import torch
import argparse
import torch.nn as nn
import pandas as pd
from torch.amp import GradScaler, autocast
from src.dataset import PaddyDataloader
from src.model import PaddyResNet
from src.engine import train_one_epoch, evaluate
import wandb


def main(args):
    if not args.no_wandb:
        wandb.init(project="paddy_doctor", config=args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using mixed precision: {args.mixed_precision}")

    data = PaddyDataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.image_size,
        num_workers=args.num_workers,
    )

    train_loader, val_loader = data.get_loaders()
    num_classes = data.num_classes
    print(f"Number of classes: {num_classes}")

    model = PaddyResNet(num_classes=num_classes)
    model = model.to(device)

    if args.full_finetune:
        print("RUNNING IN FULL FINE-TUNING MODE")
        if args.checkpoint_path and os.path.exists(args.checkpoint_path):
            print(f"Loading weights from checkpoint: {args.checkpoint_path}")
            model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        else:
            print(
                "WARNING: --checkpoint_path not provided or not found. Start full fine-tune from scratch."
            )

        print("Unfreezing the entire model for fine-tuning...")
        for param in model.parameters():
            param.requires_grad = True

        parameter_groups = [
            {"params": model.backbone.fc.parameters(), "lr": args.lr_head},
            {"params": model.backbone.layer4.parameters(), "lr": args.lr_body},
            {"params": model.backbone.layer3.parameters(), "lr": args.lr_body / 2},
            {"params": model.backbone.layer2.parameters(), "lr": args.lr_body / 3},
            {"params": model.backbone.layer1.parameters(), "lr": args.lr_early},
            {"params": model.backbone.conv1.parameters(), "lr": args.lr_early},
        ]

        optimizer = torch.optim.Adam(parameter_groups, weight_decay=args.wd_full)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[group["lr"] for group in optimizer.param_groups],
            total_steps=args.epochs * len(train_loader),
            pct_start=0.3,
        )
        print("Using OneCycleLR scheduler.")
    else:
        print("RUNNING IN HEAD-TUNING MODE")

        for param in model.backbone.parameters():
            param.requires_grad = False

        optimizer = torch.optim.Adam(
            [
                {"params": model.backbone.fc.parameters(), "lr": args.lr_head},
            ],
            weight_decay=args.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=6,
        )
        print("Using ReduceLROnPlateau scheduler.")

    criterion = nn.CrossEntropyLoss()

    # only create scaler if mixed precision is enabled
    scaler = GradScaler(
        device="cuda" if torch.cuda.is_available() else "cpu",
        enabled=args.mixed_precision,
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            args.mixed_precision,
            scheduler,
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, args.mixed_precision
        )

        if not args.full_finetune:
            scheduler.step(val_loss)

        print(f"  Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")
        print(f"  Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")

        if not args.no_wandb:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "epoch": epoch + 1,
                }
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(args.output_dir, exist_ok=True)
            save_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f" -> Val loss improved. New best model saved to {save_path}")
        else:
            patience_counter += 1
            print(
                f" -> Val loss did not improved. Early stoping counter {patience_counter}/{args.patience}"
            )

        if patience_counter >= args.patience:
            print(
                f"\nStoping earlier as validation loss has not improved for {args.patience} epochs."
            )
            break

    print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Paddy Disease Classification model"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the dataset directory ('.data/paddy_data')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/saved_models",
        help="Directory to save the best model",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr_head", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lr_backbone", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--image_size", type=int, default=224, help="Input image size(height and width)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count(),
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--no_wandb", action="store_true", help="Disable Weights & Biases logging"
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed precision training",
    )
    parser.add_argument(
        "--full_finetune",
        action="store_true",
        help="Enable full fine-tuning by unfreezing the entire model",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to a model checkpoint to load for full fine-tuning",
    )
    parser.add_argument(
        "--lr_body",
        type=float,
        default=5e-6,
        help="Learning rate for the body layers during full fine-tuning",
    )
    parser.add_argument(
        "--lr_early",
        type=float,
        default=5e-6,
        help="Learning rate for the early layers during full fine-tuning",
    )
    parser.add_argument(
        "--wd_full",
        type=float,
        default=1e-3,
        help="Weight decay for the optimizer during full fine-tuning",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for the optimizer",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Patience for early stopping",
    )

    args = parser.parse_args()
    main(args)
