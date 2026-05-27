import argparse

from s2m_kd.data import build_loaders
from s2m_kd.engine import save_model, train_student_distill
from s2m_kd.models import build_mobilenetv3_student, build_swin_teacher, load_checkpoint
from s2m_kd.utils import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Distill a Swin teacher into MobileNetV3.")
    parser.add_argument("--data-root", required=True, help="Dataset root with train/val folders.")
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--output", default="outputs/student_mobilenetv3_distilled.pth")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
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
    teacher = build_swin_teacher(pretrained=False)
    teacher = load_checkpoint(teacher, args.teacher_checkpoint, map_location=device)
    student = build_mobilenetv3_student(pretrained=True)
    metrics = train_student_distill(
        student,
        teacher,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        alpha=args.alpha,
    )
    save_model(student, args.output)
    print(metrics)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()

