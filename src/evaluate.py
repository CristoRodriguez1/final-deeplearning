"""Predicción, métricas, latencia de inferencia y Grad-CAM."""
from __future__ import annotations

import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader

from . import config as C
from .data import TensorImageDataset, get_transforms
from .models import build_model, gradcam_layer


def load_trained(run: str, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary = json.load(open(C.RUNS / run / "summary.json"))
    arch = summary["config"]["arch"]
    model = build_model(arch, pretrained=False)
    model.load_state_dict(torch.load(C.RUNS / run / "best.pt", map_location=device))
    return model.to(device).eval(), arch, summary


@torch.no_grad()
def predict_proba(model, x_uint8: torch.Tensor, batch: int = 64) -> np.ndarray:
    device = next(model.parameters()).device
    ds = TensorImageDataset(x_uint8, np.zeros(len(x_uint8)), get_transforms("eval"))
    out = []
    for xb, _ in DataLoader(ds, batch, shuffle=False):
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out.append(F.softmax(model(xb.to(device)).float(), 1).cpu())
    return torch.cat(out).numpy()


def multiclass_metrics(y: np.ndarray, proba: np.ndarray, labels=range(C.NUM_CLASSES)) -> dict:
    p = proba.argmax(1)
    labels = list(labels)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, p)),
        "f1_macro": float(f1_score(y, p, labels=labels, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y, p, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, p, labels=labels, average="macro", zero_division=0)),
        "per_class": classification_report(y, p, labels=labels, target_names=[C.CLASSES[i] for i in labels],
                                           output_dict=True, zero_division=0),
        "confusion": confusion_matrix(y, p, labels=list(range(C.NUM_CLASSES))).tolist(),
    }


def healthy_vs_diseased(y: np.ndarray, proba: np.ndarray) -> dict:
    """Tarea binaria sana (0) vs. enferma (≠0): score = 1 − p(sana)."""
    yb = (y != 0).astype(int)
    score = 1 - proba[:, 0]
    pb = (proba.argmax(1) != 0).astype(int)
    return {
        "accuracy": float(accuracy_score(yb, pb)),
        "f1_enferma": float(f1_score(yb, pb, zero_division=0)),
        "recall_enferma": float(recall_score(yb, pb, zero_division=0)),
        "recall_sana": float(recall_score(1 - yb, 1 - pb, zero_division=0)),
        "auroc": float(roc_auc_score(yb, score)) if len(set(yb)) > 1 else float("nan"),
    }


def latency_ms(model, device=None, n: int = 50, batch: int = 1) -> float:
    device = device or next(model.parameters()).device
    x = torch.randn(batch, 3, C.IMG_SIZE, C.IMG_SIZE, device=device)
    with torch.no_grad():
        for _ in range(10):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t) / (n * batch) * 1000


class GradCAM:
    """Grad-CAM (Selvaraju et al., 2017). Para ViT, los tokens (sin CLS) se reordenan en 14×14."""

    def __init__(self, model, arch: str):
        self.model, self.arch = model, arch
        layer = gradcam_layer(model, arch)
        self.acts = self.grads = None
        def fwd_hook(m, i, o):            # hook sobre el tensor: compatible con activaciones inplace
            self.acts = o
            if o.requires_grad:
                o.register_hook(lambda g: setattr(self, "grads", g))
        layer.register_forward_hook(fwd_hook)

    def _to_map(self, t):
        if self.arch == "vit_tiny":                 # B×(1+196)×D → B×D×14×14
            t = t[:, 1:, :]
            s = int(t.shape[1] ** 0.5)
            t = t.transpose(1, 2).reshape(t.shape[0], -1, s, s)
        return t

    def __call__(self, x: torch.Tensor, target: int | None = None) -> tuple[np.ndarray, int, float]:
        self.model.eval()
        x = x.unsqueeze(0).to(next(self.model.parameters()).device).requires_grad_(True)
        logits = self.model(x)
        prob = F.softmax(logits, 1)[0]
        target = int(prob.argmax()) if target is None else target
        self.model.zero_grad()
        logits[0, target].backward()
        a, g = self._to_map(self.acts), self._to_map(self.grads)
        w = g.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * a).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().cpu().numpy(), target, prob[target].item()
