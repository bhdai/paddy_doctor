import os
import pandas as pd
from typing import Tuple
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class MetadataWrapperDataset(Dataset):
    def __init__(self, image_folder_dataset: Dataset, metadata_df: pd.DataFrame):
        self.image_folder_dataset = image_folder_dataset
        self.metadata_df = metadata_df

    def __len__(self):
        return len(self.image_folder_dataset)

    def __getitem__(self, idx):
        image, label = self.image_folder_dataset[idx]
        path, _ = self.image_folder_dataset.samples[idx]

        image_id = os.path.basename(path)

        metadata = self.metadata_df.loc[image_id]
        variety_idx = metadata["variety_idx"]
        age = metadata["age"]

        return image, variety_idx, age, label


class PaddyDataloader:
    def __init__(
        self,
        processed_data_dir: str,
        metadata_df: pd.DataFrame,
        batch_size: int = 32,
        img_size: int = 224,
        num_workers: int = 4,
    ):
        self.data_dir = processed_data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers

        self._setup_transforms()

        base_train_dataset = datasets.ImageFolder(
            root=os.path.join(self.data_dir, "train"), transform=self.train_transform
        )
        base_val_dataset = datasets.ImageFolder(
            root=os.path.join(self.data_dir, "val"), transform=self.val_transform
        )

        self.train_dataset = MetadataWrapperDataset(
            image_folder_dataset=base_train_dataset,
            metadata_df=metadata_df,
        )

        self.val_dataset = MetadataWrapperDataset(
            image_folder_dataset=base_val_dataset,
            metadata_df=metadata_df,
        )

        self.class_names = base_train_dataset.classes
        self.num_classes = len(self.class_names)

    def _setup_transforms(self):
        """Defines the training and validation transformations."""
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        # calculate a slightly larger size for the initial resize
        intermediate_size = int(self.img_size * 1.1)

        self.train_transform = transforms.Compose(
            [
                transforms.Resize((intermediate_size, intermediate_size)),
                transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            ]
        )

        # Validation transform is also updated
        self.val_transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            ]
        )

    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return train_loader, val_loader
