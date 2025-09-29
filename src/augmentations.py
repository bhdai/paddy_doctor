import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: int):
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.0625, 0.0625),
                rotate=(-45, 45),
                p=0.5,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(p=1),
                    A.RandomGamma(p=1),
                ],
                p=0.5,
            ),
            A.Resize(height=image_size, width=image_size),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


def get_val_transforms(image_size: int):
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(),
            ToTensorV2(),
        ]
    )
