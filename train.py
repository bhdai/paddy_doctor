import os
import torch
import argparse
import torch.nn as nn
import pandas as pd
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from src.dataset import PaddyDataset
from src.model import TimmPaddyNet
from src.engine import train_one_epoch, evaluate
from src.augmentations import get_train_transforms, get_val_transforms
from sklearn.model_selection import StratifiedKFold
import imagehash
from PIL import Image
import wandb
import timm
from timm.scheduler.cosine_lr import CosineLRScheduler


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using mixed precision: {args.mixed_precision}")

    df = pd.read_csv(args.csv_path)

    df["path"] = df.apply(
        lambda row: os.path.join(args.data_dir, row["label"], row["image_id"]), axis=1
    )

    print(f"Original dataset size: {len(df)}")
    df["hash"] = df["path"].apply(lambda path: imagehash.phash(Image.open(path)))
    df = df.drop_duplicates(subset="hash", keep="first").reset_index(drop=True)
    print(f"Dataset size after duplicate removal: {len(df)}")

    class_map = {label: i for i, label in enumerate(df["label"].unique())}
    df["target"] = df["label"].map(class_map)

    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=42)

    for fold, (_, val_idx) in enumerate(skf.split(X=df, y=df["target"])):
        df.loc[val_idx, "fold"] = fold

    all_fold_scores = []

    for fold in range(args.num_folds):
        print(f"\nFOLD {fold}")

        if not args.no_wandb:
            wandb.init(
                project="paddy_doctor",
                name=f"fold_{fold}_{args.model_name}",
                config=args,
            )

        train_df = df[df["fold"] != fold]
        val_df = df[df["fold"] == fold]

        train_augs = get_train_transforms(image_size=args.image_size)
        val_augs = get_val_transforms(image_size=args.image_size)

        train_dataset = PaddyDataset(df=train_df, transform=train_augs)
        val_dataset = PaddyDataset(df=val_df, transform=val_augs)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TimmPaddyNet(model_name=args.model_name, num_classes=len(class_map))
        model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.wc,
        )

        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=args.epochs,
            cycle_decay=0.1,
            lr_min=1e-6,
            warmup_t=3,
            warmup_lr_init=1e-5,
            t_in_epochs=True,
            cycle_limit=1,
        )

        criterion = nn.CrossEntropyLoss()

        scaler = GradScaler(
            device="cuda" if torch.cuda.is_available() else "cpu",
            enabled=args.mixed_precision,
        )

        best_val_loss = float("inf")
        patience_counter = 0
        num_updates = 0

        for epoch in range(args.epochs):
            print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
            train_loss, train_acc, num_updates = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                scaler,
                args.mixed_precision,
                scheduler,
                num_updates,
            )
            val_loss, val_acc = evaluate(
                model, val_loader, criterion, device, args.mixed_precision
            )

            scheduler.step(epoch + 1)  # adjust the lr for the next epoch

            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss  : {val_loss:.4f}   | Val Acc  : {val_acc:.4f}")

            if not args.no_wandb:
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "epoch": epoch,
                    }
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                os.makedirs(args.output_dir, exist_ok=True)
                save_path = os.path.join(args.output_dir, f"best_model_fold_{fold}.pth")
                torch.save(model.state_dict(), save_path)
                print(f" -> Val loss improved. Model saved to {save_path}")
            else:
                patience_counter += 1
                print(
                    f" -> Val loss did not improve. Early stopping counter {patience_counter}/{args.patience}"
                )

            if patience_counter >= args.patience:
                print(f"Stopping early for fold {fold}")
                break

        all_fold_scores.append(1 - best_val_loss)
        if not args.no_wandb:
            wandb.finish()

    print("\n===== Cross-Validation Complete =====")
    print(f"Scores for each fold: {all_fold_scores}")
    print(f"Average CV Score: {sum(all_fold_scores) / len(all_fold_scores):.4f}")


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
        "--csv_path",
        type=str,
        required=True,
        help="Path to the CSV file containing image paths and labels",
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
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
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
        "--mixed-precision", action="store_true", help="Enable mixed precision training"
    )
    parser.add_argument(
        "--wc",
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
    parser.add_argument(
        "--num_folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet26d",
        help="Model architecture from timm",
    )

    args = parser.parse_args()
    main(args)
