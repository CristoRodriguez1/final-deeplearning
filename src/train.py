"""Entrenamiento con AdamW + cosine (warmup), CE ponderada con label smoothing y early stopping por F1-macro.

Uso:  python -m src.train --run effnet_b0        (corridas definidas en RUNS_CFG)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from . import config as C
from .data import TensorImageDataset, get_caches, get_transforms, load_tables
from .models import build_model, count_params

# Configuraciones de las corridas del proyecto
RUNS_CFG = {
    "cnn_base":         dict(arch="cnn",       lr=1e-3, wd=1e-4, epochs=40, aug="train",     patience=10),
    "effnet_b0":        dict(arch="effnet_b0", lr=3e-4, wd=1e-2, epochs=25, aug="train",     patience=8),
    "vit_tiny":         dict(arch="vit_tiny",  lr=1e-4, wd=5e-2, epochs=25, aug="train",     patience=8),
    "effnet_b0_sinaug": dict(arch="effnet_b0", lr=3e-4, wd=1e-2, epochs=25, aug="train_min", patience=8),
}
DEFAULTS = dict(batch=32, warmup=2, label_smoothing=0.1, seed=C.SEED, max_train=None)


def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def hardware_info() -> dict:
    info = {"python": platform.python_version(), "torch": torch.__version__, "cpu": platform.processor() or platform.machine()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, gpu_mem_gb=round(p.total_memory / 1e9, 1))
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    return info


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    tot_loss, ys, ps = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train), torch.autocast(device.type, dtype=torch.float16, enabled=scaler is not None):
            logits = model(x)
            loss = criterion(logits, y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()
            scheduler.step()
        tot_loss += loss.item() * len(y)
        ys.append(y.cpu()); ps.append(logits.argmax(1).cpu())
    y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    return {"loss": tot_loss / len(y), "acc": float((y == p).mean()),
            "f1": float(f1_score(y, p, average="macro", zero_division=0))}


def train(run: str, overrides: dict | None = None, caches: dict | None = None, verbose: bool = True) -> dict:
    cfg = {**DEFAULTS, **RUNS_CFG.get(run, {}), **(overrides or {})}
    if os.environ.get("PROYECTO_SMOKE"):          # prueba rápida del pipeline (no usar para resultados)
        cfg.update(epochs=1, max_train=64)
    out = C.RUNS / run
    if (out / "summary.json").exists() and not cfg.get("force"):
        if verbose:
            print(f"[{run}] ya entrenado → se omite (borra {out} para reentrenar)")
        return json.load(open(out / "summary.json"))
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bra, _ = load_tables()
    caches = caches or get_caches()
    X = caches["bracol"]
    tr, va = bra.index[bra.split == "train"].to_numpy(), bra.index[bra.split == "val"].to_numpy()
    if cfg["max_train"]:
        tr, va = tr[: cfg["max_train"]], va[: cfg["max_train"]]
    ds_tr = TensorImageDataset(X[tr], bra.label.values[tr], get_transforms(cfg["aug"]))
    ds_va = TensorImageDataset(X[va], bra.label.values[va], get_transforms("eval"))
    nw = 2 if device.type == "cuda" else 0
    dl_tr = DataLoader(ds_tr, cfg["batch"], shuffle=True, num_workers=nw, pin_memory=device.type == "cuda",
                       drop_last=True, persistent_workers=nw > 0)
    dl_va = DataLoader(ds_va, 64, shuffle=False, num_workers=nw, persistent_workers=nw > 0)

    model = build_model(cfg["arch"]).to(device)
    params = count_params(model)
    meta = json.load(open(C.PROC / "meta.json"))
    w = torch.tensor(meta["class_weights"], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    steps, warm = cfg["epochs"] * len(dl_tr), cfg["warmup"] * len(dl_tr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    hist, best, bad, t0 = [], -1.0, 0, time.time()
    for ep in range(1, cfg["epochs"] + 1):
        te = time.time()
        lr = optimizer.param_groups[0]["lr"]
        m_tr = run_epoch(model, dl_tr, criterion, device, optimizer, sched, scaler)
        m_va = run_epoch(model, dl_va, criterion, device)
        row = {"epoch": ep, "lr": lr, **{f"train_{k}": v for k, v in m_tr.items()},
               **{f"val_{k}": v for k, v in m_va.items()}, "epoch_time_s": time.time() - te}
        hist.append(row)
        improved = m_va["f1"] > best
        if improved:
            best, bad, best_ep = m_va["f1"], 0, ep
            torch.save(model.state_dict(), out / "best.pt")
        else:
            bad += 1
        if verbose:
            print(f"[{run}] ep {ep:02d} | loss {m_tr['loss']:.3f}/{m_va['loss']:.3f} | "
                  f"F1 {m_tr['f1']:.3f}/{m_va['f1']:.3f} | {row['epoch_time_s']:.1f}s" + (" *" if improved else ""))
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        if bad >= cfg["patience"]:
            if verbose:
                print(f"[{run}] early stopping en la época {ep} (sin mejora en {cfg['patience']} épocas)")
            break

    summary = {"run": run, "config": cfg, "params": params, "best_epoch": best_ep, "best_val_f1": best,
               "epochs_run": len(hist), "train_time_s": time.time() - t0, "hardware": hardware_info(),
               "n_train": len(tr), "n_val": len(va)}
    json.dump(summary, open(out / "summary.json", "w"), indent=2, default=str)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--max_train", type=int)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    ov = {k: v for k, v in {"epochs": a.epochs, "max_train": a.max_train, "force": a.force}.items() if v}
    print(json.dumps(train(a.run, ov), indent=2, default=str))
