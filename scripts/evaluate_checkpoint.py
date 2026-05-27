import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from s2m_kd.data import build_dataset
from s2m_kd.engine import evaluate_accuracy
from s2m_kd.models import build_mobilenetv3_student, build_swin_teacher, load_checkpoint
from s2m_kd.utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on val or test split.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", choices=["mobilenetv3-large", "swin-tiny"], default="mobilenetv3-large")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output", default="outputs/eval.json")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    model = build_mobilenetv3_student(pretrained=False) if args.model == "mobilenetv3-large" else build_swin_teacher(pretrained=False)
    model = load_checkpoint(model, args.checkpoint, map_location=device)
    model.to(device)
    ds = build_dataset(args.data_root, args.split, image_size=args.image_size, train=False, remap_classes=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    acc = evaluate_accuracy(model, loader, device)
    result = {"model": args.model, "split": args.split, "samples": len(ds), "accuracy": round(acc, 4)}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

