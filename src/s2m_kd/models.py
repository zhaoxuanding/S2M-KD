from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def build_swin_teacher(num_classes: int = 4,
                       pretrained: bool = True,
                       model_name: str = "swin_tiny_patch4_window7_224") -> nn.Module:
    import timm

    return timm.create_model(model_name, pretrained=pretrained,
                             num_classes=num_classes)


def build_mobilenetv3_student(num_classes: int = 4,
                              size: str = "large",
                              pretrained: bool = True) -> nn.Module:
    if size == "large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
    elif size == "small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
    else:
        raise ValueError("size must be 'large' or 'small'")
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model


def load_checkpoint(model: nn.Module,
                    checkpoint_path: str,
                    map_location: str | torch.device = "cpu",
                    strict: bool = False) -> nn.Module:
    state = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(state, dict):
        for key in ["state_dict", "model", "net"]:
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    model.load_state_dict(state, strict=strict)
    return model


def build_feature_backbone(name: str, pretrained: bool = True) -> tuple[nn.Module, int, int]:
    name = name.lower()
    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, dim, 224
    if name == "resnet152":
        weights = models.ResNet152_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet152(weights=weights)
        dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, dim, 224
    if name == "densenet201":
        weights = models.DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet201(weights=weights)
        dim = model.classifier.in_features
        model.classifier = nn.Identity()
        return model, dim, 224
    if name == "mobilenetv3-large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        dim = model.classifier[0].in_features
        model.classifier = nn.Identity()
        return model, dim, 224
    if name == "mobilenetv3-small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        dim = model.classifier[0].in_features
        model.classifier = nn.Identity()
        return model, dim, 224
    if name == "vit_b_16":
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vit_b_16(weights=weights)
        dim = model.heads.head.in_features
        model.heads.head = nn.Identity()
        return model, dim, 224
    raise ValueError(f"Unsupported backbone: {name}")

