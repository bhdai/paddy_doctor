import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class PaddyDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.loc[idx, "path"]
        label = self.df.loc[idx, "target"]

        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Image not found: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img.shape[0] > img.shape[1]:
            img = np.rot90(img)

        h, w, c = img.shape
        if h != w:
            size = max(h, w)
            pad_top = (size - h) // 2
            pad_bottom = size - h - pad_top
            pad_left = (size - w) // 2
            pad_right = size - w - pad_left

            img = np.pad(
                img,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]
        else:
            img = img.astype(np.float32)

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img.transpose(2, 0, 1)).float()

        return img, torch.tensor(label, dtype=torch.long)
