import torch.nn as nn
import torch.nn.functional as F


def distillation_loss(student_logits,
                      teacher_logits,
                      labels,
                      temperature: float = 4.0,
                      alpha: float = 0.5):
    hard = F.cross_entropy(student_logits, labels)
    soft = nn.KLDivLoss(reduction="batchmean")(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
    ) * (temperature * temperature)
    return (1.0 - alpha) * hard + alpha * soft

