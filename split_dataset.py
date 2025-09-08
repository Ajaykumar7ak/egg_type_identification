import os
import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def copy_files(files, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for src in files:
        dst = os.path.join(dest_dir, os.path.basename(src))
        shutil.copy2(src, dst)

def main():
    parser = argparse.ArgumentParser(description="Split dataset in data/raw into train/val/test.")
    parser.add_argument("--raw_dir", default="data/raw", help="Path to raw dataset with class folders")
    parser.add_argument("--out_dir", default="data", help="Output root (will create train/val/test)")
    parser.add_argument("--train", type=float, default=0.7, help="Train ratio")
    parser.add_argument("--val", type=float, default=0.2, help="Validation ratio")
    parser.add_argument("--test", type=float, default=0.1, help="Test ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    assert abs(args.train + args.val + args.test - 1.0) < 1e-6, "ratios must sum to 1"
    random.seed(args.seed)

    classes = [d for d in os.listdir(args.raw_dir) if os.path.isdir(os.path.join(args.raw_dir, d))]
    if not classes:
        raise SystemExit(f"No class folders found in {args.raw_dir}. Expected e.g., chicken/, duck/, turkey/, emu/, ostrich/")

    for cls in classes:
        cls_dir = os.path.join(args.raw_dir, cls)
        files = [str(Path(cls_dir)/f) for f in os.listdir(cls_dir) if Path(f).suffix.lower() in IMG_EXTS]
        files.sort()
        random.shuffle(files)
        if not files:
            print(f"[WARN] No images for class '{cls}'")
            continue

        n = len(files)
        n_train = int(n * args.train)
        n_val = int(n * args.val)
        n_test = n - n_train - n_val

        train_files = files[:n_train]
        val_files = files[n_train:n_train+n_val]
        test_files = files[n_train+n_val:]

        copy_files(train_files, os.path.join(args.out_dir, "train", cls))
        copy_files(val_files, os.path.join(args.out_dir, "val", cls))
        copy_files(test_files, os.path.join(args.out_dir, "test", cls))

        print(f"{cls}: total={n}, train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    print("Done. Train/Val/Test sets created under", args.out_dir)

if __name__ == "__main__":
    main()
