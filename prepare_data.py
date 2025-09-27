import os
import random
import shutil
import argparse


def main(source_dir, dest_dir, train_ratio):
    random.seed(42)

    train_dir = os.path.join(dest_dir, "train")
    val_dir = os.path.join(dest_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    class_names = [
        d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))
    ]

    for class_name in class_names:
        class_source_dir = os.path.join(source_dir, class_name)
        images_files = [f for f in os.listdir(class_source_dir) if f.endswith(".jpg")]
        random.shuffle(images_files)

        split_idx = int(train_ratio * len(images_files))
        train_files = images_files[:split_idx]
        val_files = images_files[split_idx:]

        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        for f in train_files:
            src_path = os.path.join(class_source_dir, f)
            dst_path = os.path.join(train_class_dir, f)
            shutil.copy(src_path, dst_path)

        for f in val_files:
            src_path = os.path.join(class_source_dir, f)
            dst_path = os.path.join(val_class_dir, f)
            shutil.copy(src_path, dst_path)

    print(
        f"Data successfully split into {train_ratio * 100:.0f}% train and {(train_ratio) * 100:.0f}% validation sets"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split image into train and test sets")
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Source directory containing class subfolders",
    )
    parser.add_argument(
        "--dest_dir",
        type=str,
        required=True,
        help="Destination directory for train/val split",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of training data (between 0 and 1)",
    )
    args = parser.parse_args()

    main(args.source_dir, args.dest_dir, args.train_ratio)
