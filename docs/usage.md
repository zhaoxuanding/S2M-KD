# Usage Notes

## Train Teacher

```bash
python scripts/train_teacher.py --data-root data/Data1 --output outputs/teacher_swin_data1.pth
```

## Distill Student

```bash
python scripts/train_student_distill.py \
  --data-root data/Data1 \
  --teacher-checkpoint outputs/teacher_swin_data1.pth \
  --output outputs/student_mobilenetv3_data1.pth
```

## Evaluate

```bash
python scripts/evaluate_checkpoint.py \
  --data-root data/Data1 \
  --checkpoint outputs/student_mobilenetv3_data1.pth \
  --model mobilenetv3-large \
  --split test
```

## Prepare Splits

If a dataset is organized as one folder per class, create train/val/test folders:

```bash
python scripts/prepare_splits.py \
  --source raw_data/Data1 \
  --output data/Data1 \
  --train-ratio 0.70 \
  --val-ratio 0.15
```

