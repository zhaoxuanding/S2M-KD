from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CANONICAL_CLASSES = ["cocci", "healthy", "ncd", "salmo"]

CLASS_ALIASES: Dict[str, str] = {
    "cocci": "cocci",
    "coccidiosis": "cocci",
    "Coccidiosis": "cocci",
    "healthy": "healthy",
    "Healthy": "healthy",
    "ncd": "ncd",
    "newcastle": "ncd",
    "Newcastle": "ncd",
    "New Castle Disease": "ncd",
    "salmo": "salmo",
    "salmonella": "salmo",
    "Salmonella": "salmo",
}


def image_transforms(image_size: int = 224, train: bool = True):
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])


def remap_targets_to_canonical(ds: datasets.ImageFolder,
                               aliases: Dict[str, str] | None = None,
                               canonical_classes: Iterable[str] = CANONICAL_CLASSES) -> None:
    aliases = aliases or CLASS_ALIASES
    canonical = list(canonical_classes)
    canonical_to_idx = {name: i for i, name in enumerate(canonical)}

    missing = [name for name in ds.classes if name not in aliases]
    if missing:
        raise ValueError(
            f"Class folders not mapped to canonical labels: {missing}. "
            f"Expected aliases include: {sorted(aliases)}"
        )

    old_to_new = {}
    for class_name, old_idx in ds.class_to_idx.items():
        old_to_new[old_idx] = canonical_to_idx[aliases[class_name]]

    ds.samples = [(path, old_to_new[target]) for path, target in ds.samples]
    ds.targets = [target for _, target in ds.samples]
    ds.classes = canonical
    ds.class_to_idx = canonical_to_idx


def build_dataset(root: str | Path,
                  split: str,
                  image_size: int = 224,
                  train: bool = True,
                  remap_classes: bool = True) -> datasets.ImageFolder:
    split_dir = Path(root) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")
    ds = datasets.ImageFolder(split_dir, image_transforms(image_size, train=train))
    if remap_classes:
        remap_targets_to_canonical(ds)
    return ds


def build_loaders(root: str | Path,
                  image_size: int = 224,
                  batch_size: int = 128,
                  num_workers: int = 8,
                  remap_classes: bool = True) -> Tuple[DataLoader, DataLoader]:
    train_set = build_dataset(root, "train", image_size, train=True,
                              remap_classes=remap_classes)
    val_set = build_dataset(root, "val", image_size, train=False,
                            remap_classes=remap_classes)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def split_summary(root: str | Path) -> List[dict]:
    root = Path(root)
    rows = []
    for split in ["train", "val", "test"]:
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            count = sum(1 for p in class_dir.rglob("*") if p.is_file())
            rows.append({"split": split, "class": class_dir.name, "count": count})
    return rows

