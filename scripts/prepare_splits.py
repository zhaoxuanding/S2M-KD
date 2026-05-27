import argparse
import random
import shutil
from pathlib import Path


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Create train/val/test splits from class folders.")
    parser.add_argument("--source", required=True, help="Input folder containing one subfolder per class.")
    parser.add_argument("--output", required=True, help="Output dataset root.")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def image_files(class_dir: Path):
    return [p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]


def main():
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists. Use --overwrite to replace it.")
    if output.exists():
        shutil.rmtree(output)
    random.seed(args.seed)

    for class_dir in sorted(p for p in source.iterdir() if p.is_dir()):
        files = image_files(class_dir)
        random.shuffle(files)
        n_train = int(len(files) * args.train_ratio)
        n_val = int(len(files) * args.val_ratio)
        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }
        for split, split_files in splits.items():
            out_dir = output / split / class_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for src in split_files:
                shutil.copy2(src, out_dir / src.name)
        print(class_dir.name, {k: len(v) for k, v in splits.items()})


if __name__ == "__main__":
    main()

