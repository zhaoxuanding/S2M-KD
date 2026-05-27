import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader

from s2m_kd.data import build_dataset
from s2m_kd.models import build_feature_backbone
from s2m_kd.utils import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-dataset feature extractor + SVM evaluation.")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--backbone", default="resnet18",
                        choices=["resnet18", "resnet152", "densenet201",
                                 "mobilenetv3-large", "mobilenetv3-small", "vit_b_16"])
    parser.add_argument("--output", default="outputs/cross_dataset_svm.json")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    features, labels = [], []
    for images, y in loader:
        images = images.to(device, non_blocking=True)
        out = model(images).detach().cpu().numpy()
        features.append(out)
        labels.append(y.numpy())
    return np.concatenate(features), np.concatenate(labels)


def make_loader(root, split, image_size, batch_size, num_workers):
    ds = build_dataset(root, split, image_size=image_size, train=False, remap_classes=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True), ds.classes


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    backbone, feat_dim, image_size = build_feature_backbone(args.backbone, pretrained=True)
    backbone.to(device)

    source_train, classes = make_loader(args.source_root, "train", image_size,
                                        args.batch_size, args.num_workers)
    target_split = "test" if (Path(args.target_root) / "test").is_dir() else "val"
    target_eval, _ = make_loader(args.target_root, target_split, image_size,
                                 args.batch_size, args.num_workers)

    x_train, y_train = extract_features(backbone, source_train, device)
    x_eval, y_eval = extract_features(backbone, target_eval, device)
    clf = make_pipeline(StandardScaler(), LinearSVC(C=1.0, max_iter=5000, dual="auto", random_state=args.seed))
    clf.fit(x_train, y_train)
    result = {
        "backbone": args.backbone,
        "feature_dim": feat_dim,
        "classes": classes,
        "source_train_samples": int(x_train.shape[0]),
        "target_split": target_split,
        "target_samples": int(x_eval.shape[0]),
        "target_accuracy": round(float(clf.score(x_eval, y_eval) * 100.0), 4),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

