import os
import torch
import argparse
import torch.nn as nn
import pandas as pd
from torch.amp import GradScaler, autocast
from src.dataset import PaddyDataloader
from src.model import PaddyMultimodalNet
from src.engine import train_one_epoch, evaluate
import wandb


def main(args):
    if not args.no_wandb:
        wandb.init(project="paddy_doctor", config=args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using mixed precision: {args.mixed_precision}")

    print("Prepare metadata and data split...")
    metadata_df = pd.read_csv(args.csv_path)

    # map each variety to an integer
    varieties = metadata_df["variety"].unique()
    variety_map = {v: i for i, v in enumerate(varieties)}
    metadata_df["variety_idx"] = metadata_df["variety"].map(variety_map)
    num_varieties = len(varieties)
    # set image_id as index for fast lookups
    metadata_df.set_index("image_id", inplace=True)

    data = PaddyDataloader(
        processed_data_dir=args.data_dir,
        metadata_df=metadata_df,
        batch_size=args.batch_size,
        img_size=args.image_size,
        num_workers=args.num_workers,
    )

    train_loader, val_loader = data.get_loaders()
    num_classes = data.num_classes
    print(f"Number of classes: {num_classes}")

    model = PaddyMultimodalNet(num_classes=num_classes, num_varieties=num_varieties)
    for param in model.backbone.layer4.parameters():
        param.requires_grad = True
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        [
            {"params": model.backbone.layer4.parameters(), "lr": args.lr_backbone},
            {"params": model.variety_embedding.parameters(), "lr": args.lr_fc},
            {"params": model.age_processor.parameters(), "lr": args.lr_fc},
            {"params": model.classifier.parameters(), "lr": args.lr_fc},
        ],
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=3,
    )
    # Only create scaler if mixed precision is enabled
    scaler = GradScaler(
        device="cuda" if torch.cuda.is_available() else "cpu",
        enabled=args.mixed_precision,
    )

    best_val_loss = float("inf")
    patience_counter = 0
    patience = 5

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
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, args.mixed_precision
        )
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
                f" -> Val loss did not improved. Early stoping counter {patience_counter}/{patience}"
            )

        if patience_counter >= patience:
            print(
                f"\nStoping earlier as validation loss has not improved for {patience} epochs."
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
    parser.add_argument("--lr_fc", type=float, default=1e-3, help="Learning rate")
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
        "--csv_path", type=str, required=True, help="Path to the metadata CSV file"
    )

    args = parser.parse_args()
    main(args)
