import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm

from src.model import TimmPaddyNet
from src.augmentations import get_tta_transforms


class CFG:
    IMAGE_SIZE = 384
    BATCH_SIZE = 32
    MODEL_NAME = "resnet26d"
    NUM_CLASSES = 10
    NUM_FOLDS = 5
    NUM_TTA = 5


class TestPaddyDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        path = self.df.loc[index, "path"]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image.shape[0] > image.shape[1]:
            image = np.rot90(image)

        h, w, _ = image.shape
        if h != w:
            size = max(h, w)
            pad_top = (size - h) // 2
            pad_bottom = size - h - pad_top
            pad_left = (size - w) // 2
            pad_right = size - w - pad_left
            image = np.pad(
                image,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df = pd.read_csv(args.train_csv_path)
    class_names = sorted(train_df["label"].unique())
    class_map_inv = {i: name for i, name in enumerate(class_names)}
    print(f"Loaded class map: {class_map_inv}")

    models = []
    for fold in range(CFG.NUM_FOLDS):
        model_path = os.path.join(args.model_dir, f"best_model_fold_{fold}.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found for fold {fold} at {model_path}")

        print(f"Loading model for fold {fold}...")
        model = TimmPaddyNet(model_name=CFG.MODEL_NAME, num_classes=CFG.NUM_CLASSES)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)
    print(f"Successfully loaded {len(models)} models for the ensemble.")

    sub_df = pd.read_csv(args.submission_csv_path)
    sub_df["path"] = sub_df["image_id"].apply(
        lambda x: os.path.join(args.test_image_dir, x)
    )

    tta_transforms = get_tta_transforms(CFG.IMAGE_SIZE)
    test_dataset = TestPaddyDataset(df=sub_df, transform=tta_transforms)
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        num_workers=args.num_workers,
    )

    final_predictions = []
    with torch.no_grad():
        for _ in tqdm(range(CFG.NUM_TTA), desc="TTA Runs"):
            tta_run_preds = []
            for images in tqdm(test_loader, desc="Batch Inference", leave=False):
                images = images.to(device)
                ensemble_preds_for_batch = []
                for model in models:
                    with torch.amp.autocast(
                        device_type=device.type,
                        enabled=args.use_amp,
                    ):
                        outputs = model(images)
                    ensemble_preds_for_batch.append(torch.softmax(outputs, dim=1).cpu())
                avg_ensemble_preds = torch.stack(ensemble_preds_for_batch).mean(dim=0)
                tta_run_preds.append(avg_ensemble_preds)
            final_predictions.append(torch.cat(tta_run_preds))

    final_avg_preds = torch.stack(final_predictions).mean(dim=0)
    predicted_indices = torch.argmax(final_avg_preds, dim=1).numpy()
    predicted_labels = [class_map_inv[i] for i in predicted_indices]

    sub_df["label"] = predicted_labels
    sub_df[["image_id", "label"]].to_csv("submission.csv", index=False)

    print("\nSubmission file 'submission.csv' created successfully!")
    print(sub_df.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate submission file")
    parser.add_argument(
        "--train_csv_path",
        type=str,
        required=True,
        help="Path to the training CSV file",
    )
    parser.add_argument(
        "--submission_csv_path",
        type=str,
        required=True,
        help="Path to the sample submission CSV file",
    )
    parser.add_argument(
        "--test_image_dir",
        type=str,
        required=True,
        help="Directory containing test images",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing the trained model folds",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of DataLoader workers"
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Use Automatic Mixed Precision for inference",
    )
    args = parser.parse_args()

    main(args)
