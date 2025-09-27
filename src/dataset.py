import os
from typing import Tuple
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class PaddyDataloader:
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        img_size: int = 224,
        num_workers: int = 4,
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers

        self._setup_transforms()

        self.train_dataset = datasets.ImageFolder(
            root=os.path.join(self.data_dir, "train"), transform=self.train_transform
        )
        self.val_dataset = datasets.ImageFolder(
            root=os.path.join(self.data_dir, "val"), transform=self.val_transform
        )

        self.class_names = self.train_dataset.classes
        self.num_classes = len(self.class_names)

    def _setup_transforms(self):
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        self.train_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),  # resize to a larger size
                transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            ]
        )

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
