"""Arquitecturas: CNN base desde cero, EfficientNet-B0 y ViT-Tiny (preentrenados, timm)."""
import timm
import torch
import torch.nn as nn

from . import config as C


class SimpleCNN(nn.Module):
    """Línea base desde cero: 4 bloques [Conv3×3-BN-ReLU]×2 + MaxPool, GAP, Dropout, FC."""

    def __init__(self, num_classes: int = C.NUM_CLASSES, widths=(32, 64, 128, 256), p_drop: float = 0.3):
        super().__init__()
        layers, c_in = [], 3
        for w in widths:
            layers += [nn.Conv2d(c_in, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                       nn.Conv2d(w, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            c_in = w
        self.features = nn.Sequential(*layers)          # 224 → 14×14×256
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(p_drop), nn.Linear(c_in, num_classes))

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


TIMM_NAMES = {
    "effnet_b0": "efficientnet_b0.ra_in1k",
    "vit_tiny": "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
}


def build_model(arch: str, pretrained: bool = True) -> nn.Module:
    if arch == "cnn":
        return SimpleCNN()
    if arch == "effnet_b0":
        return timm.create_model(TIMM_NAMES[arch], pretrained=pretrained, num_classes=C.NUM_CLASSES, drop_rate=0.3)
    if arch == "vit_tiny":
        return timm.create_model(TIMM_NAMES[arch], pretrained=pretrained, num_classes=C.NUM_CLASSES, drop_rate=0.1)
    raise ValueError(arch)


def gradcam_layer(model: nn.Module, arch: str) -> nn.Module:
    """Capa cuyas activaciones/gradientes se usan en Grad-CAM."""
    if arch == "cnn":
        return model.features[-2]          # ReLU del último bloque (14×14)
    if arch == "effnet_b0":
        return model.bn2                   # conv_head + BN + SiLU (7×7×1280)
    if arch == "vit_tiny":
        return model.blocks[-1].norm1      # tokens del último bloque → 14×14
    raise ValueError(arch)


def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "entrenables": train}
