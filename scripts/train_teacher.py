import argparse

from s2m_kd.data import build_loaders
from s2m_kd.engine import save_model, train_teacher
from s2m_kd.models import build_swin_teacher
from s2m_kd.utils import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Swin teacher.")
    parser.add_argument("--data-root", required=True, help="Dataset root with train/val folders.")
    parser.add_argument("--output", default="outputs/teacher_swin_tiny.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    train_loader, val_loader = build_loaders(
        args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = build_swin_teacher(pretrained=not args.no_pretrained)
    metrics = train_teacher(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    save_model(model, args.output)
    print(metrics)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()

