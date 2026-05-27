from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from tqdm import tqdm

from .losses import distillation_loss


def evaluate_accuracy(model: nn.Module,
                      loader,
                      device: torch.device,
                      use_amp: bool = True) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                logits = model(images)
            correct += logits.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / max(total, 1)


def train_teacher(model: nn.Module,
                  train_loader,
                  val_loader,
                  device: torch.device,
                  epochs: int = 30,
                  lr: float = 5e-4,
                  weight_decay: float = 0.05,
                  use_amp: bool = True) -> Dict[str, float]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    best_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"teacher epoch {epoch}/{epochs}")
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()
        val_acc = evaluate_accuracy(model, val_loader, device, use_amp=use_amp)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(f"epoch={epoch} loss={running_loss / max(len(train_loader), 1):.4f} val_acc={val_acc:.2f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_acc": round(best_acc, 4)}


def train_student_distill(student: nn.Module,
                          teacher: nn.Module,
                          train_loader,
                          val_loader,
                          device: torch.device,
                          epochs: int = 50,
                          lr: float = 1e-3,
                          weight_decay: float = 1e-4,
                          temperature: float = 4.0,
                          alpha: float = 0.5,
                          use_amp: bool = True) -> Dict[str, float]:
    student.to(device)
    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    best_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        student.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"student epoch {epoch}/{epochs}")
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_logits = teacher(images)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                student_logits = student(images)
                loss = distillation_loss(
                    student_logits,
                    teacher_logits,
                    labels,
                    temperature=temperature,
                    alpha=alpha,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()
        val_acc = evaluate_accuracy(student, val_loader, device, use_amp=use_amp)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in student.state_dict().items()}
        print(f"epoch={epoch} loss={running_loss / max(len(train_loader), 1):.4f} val_acc={val_acc:.2f}")

    if best_state is not None:
        student.load_state_dict(best_state)
    return {"best_val_acc": round(best_acc, 4)}


def save_model(model: nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)

