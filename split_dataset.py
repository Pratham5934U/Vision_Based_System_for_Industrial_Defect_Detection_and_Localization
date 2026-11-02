"""
split_dataset.py
----------------
Splits the downloaded NEU Surface Defect dataset into train/test sets.

Usage:
    python split_dataset.py
    python split_dataset.py --src data/NEU-DET/images --ratio 0.8

Expected source structure:
    <src>/
    ├── crazing/        (300 .bmp images)
    ├── inclusion/
    ├── patches/
    ├── pitted_surface/
    ├── rolled-in_scale/
    └── scratches/

Output:
    data/train/<class>/*.bmp
    data/test/<class>/*.bmp
"""

import os
import shutil
import random
import argparse
from pathlib import Path


def split_dataset(src: str, train_ratio: float = 0.8, seed: int = 42):
    src_path = Path(src)
    if not src_path.exists():
        raise FileNotFoundError(f"Source directory not found: {src_path}")

    random.seed(seed)

    class_folders = [d for d in src_path.iterdir() if d.is_dir()]
    if not class_folders:
        raise ValueError(f"No class subdirectories found in {src_path}")

    print(f"Found {len(class_folders)} classes: {[d.name for d in class_folders]}")
    print(f"Train ratio: {train_ratio:.0%} | Test ratio: {1-train_ratio:.0%}\n")

    total_train, total_test = 0, 0

    for class_folder in sorted(class_folders):
        images = (
            list(class_folder.glob("*.jpg"))
            + list(class_folder.glob("*.jpeg"))
            + list(class_folder.glob("*.bmp"))
            + list(class_folder.glob("*.png"))
        )

        if not images:
            print(f"  ⚠  No images in {class_folder.name}, skipping.")
            continue

        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        train_imgs = images[:split_idx]
        test_imgs  = images[split_idx:]

        for split, imgs in [("train", train_imgs), ("test", test_imgs)]:
            dest_dir = Path("data") / split / class_folder.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                shutil.copy(img, dest_dir / img.name)

        print(f"  {class_folder.name:20s}  →  train: {len(train_imgs):4d}  |  test: {len(test_imgs):4d}")
        total_train += len(train_imgs)
        total_test  += len(test_imgs)

    print(f"\n✅ Done! Total  →  train: {total_train}  |  test: {total_test}")
    print("   Check the data/ folder to confirm.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split NEU dataset into train/test")
    parser.add_argument(
        "--src",
        default="data/NEU-DET/images",
        help="Path to the downloaded dataset folder (default: data/NEU-DET/images)"
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.8,
        help="Train split ratio (default: 0.8)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()
    split_dataset(src=args.src, train_ratio=args.ratio, seed=args.seed)
