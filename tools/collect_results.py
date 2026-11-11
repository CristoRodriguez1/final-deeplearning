"""Reúne todas las cifras del proyecto en entregables/resultados.json (fuente única para diapositivas e informe).

Uso: python tools/collect_results.py [--out RAIZ_CON_runs_Y_figures]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(ROOT))
a = ap.parse_args()
OUT = Path(a.out)
RUNS = OUT / "runs"

bra = pd.read_csv(ROOT / "data/processed/bracol_leaf.csv")
roc = pd.read_csv(ROOT / "data/processed/rocole.csv")
CL = ["sana", "minador", "roya", "phoma", "cercospora"]
R = {}

R["datos"] = {
    "bracol_total": 1747, "bracol_indeterminadas": 62, "bracol_usadas": int(len(bra)),
    "por_clase": bra.clase.value_counts().reindex(CL).astype(int).to_dict(),
    "split": bra.split.value_counts().reindex(["train", "val", "test"]).astype(int).to_dict(),
    "split_por_clase": pd.crosstab(bra.clase, bra.split).reindex(CL)[["train", "val", "test"]].astype(int).to_dict("index"),
    "multiestres_pct": round(100 * (bra[["miner", "rust", "phoma", "cercospora"]].sum(1) > 1).mean(), 1),
    "rocole_total": int(len(roc)), "rocole_in": int((roc.label >= 0).sum()),
    "rocole_sana": int((roc.clase == "sana").sum()), "rocole_roya": int((roc.clase == "roya").sum()),
    "rocole_ood": int((roc.rocole_class == "red_spider_mite").sum()),
    "rocole_por_clase": roc.rocole_class.value_counts().astype(int).to_dict(),
    "meta": json.load(open(ROOT / "data/processed/meta.json", encoding="utf8")),
    # Cuantificación del cambio de dominio: salida del notebook 01 (muestra de 400 imágenes por dominio)
    "dominio": {"BRACOL": {"brillo": 0.619, "saturacion": 0.276, "frac_blanco": 0.295},
                "RoCoLe-raw": {"brillo": 0.407, "saturacion": 0.445, "frac_blanco": 0.014},
                "RoCoLe-mask": {"brillo": 0.710, "saturacion": 0.297, "frac_blanco": 0.467}},
}

runs = ["cnn_base", "effnet_b0", "vit_tiny", "effnet_b0_sinaug"]
R["runs"] = {}
for r in runs:
    s = json.load(open(RUNS / r / "summary.json"))
    h = pd.read_csv(RUNS / r / "history.csv")
    b = h.loc[h.epoch == s["best_epoch"]].iloc[0]
    R["runs"][r] = {"params_M": round(s["params"]["total"] / 1e6, 2), "best_epoch": int(s["best_epoch"]),
                    "epochs_run": int(s["epochs_run"]), "best_val_f1": round(s["best_val_f1"], 4),
                    "train_f1_at_best": round(float(b.train_f1), 4), "val_loss_at_best": round(float(b.val_loss), 4),
                    "train_loss_at_best": round(float(b.train_loss), 4),
                    "min_val_loss": round(float(h.val_loss.min()), 4), "epoch_min_val_loss": int(h.loc[h.val_loss.idxmin(), "epoch"]),
                    "final_train_f1": round(float(h.train_f1.iloc[-1]), 4), "final_val_f1": round(float(h.val_f1.iloc[-1]), 4),
                    "train_min": round(s["train_time_s"] / 60, 1), "s_epoca": round(float(h.epoch_time_s.mean()), 1),
                    "config": {k: s["config"][k] for k in ("lr", "wd", "epochs", "aug", "patience", "batch", "label_smoothing")},
                    "history": h[["epoch", "train_loss", "val_loss", "train_f1", "val_f1"]].round(4).to_dict("list")}
R["hardware"] = json.load(open(RUNS / "cnn_base" / "summary.json"))["hardware"]
hw = RUNS / "hardware.json"
if hw.exists():
    R["hardware"].update(json.load(open(hw)))

m = json.load(open(RUNS / "metrics.json"))
R["best"] = m["best_model"]
R["test"] = {r: {k: round(v, 4) for k, v in m["test"][r].items() if isinstance(v, float)} for r in runs}
R["test_por_clase"] = {r: {c: {k: round(v, 3) for k, v in m["test"][r]["per_class"][c].items()} for c in CL} for r in runs}
R["confusion_best"] = m["test"][R["best"]]["confusion"]
rt = pd.read_csv(RUNS / "resultados_test.csv")
R["latencia"] = rt[["modelo", "ms_img_gpu", "ms_img_cpu", "params_M"]].round(2).to_dict("records")
R["externo"] = pd.read_csv(RUNS / "resultados_externos.csv").round(4).to_dict("records")
R["ood"] = m["ood"]
R["por_nivel"] = m["rocole_por_nivel"]
R["multiestres"] = m["errores_multiestres"]
R["n"] = m["n"]
R["literatura"] = {"esgario2020_acc": 0.9524, "modelo": "ResNet-50 multitarea", "params_M": 25.6}

def clean(o):                     # NaN → null (JSON válido para el generador de diapositivas)
    if isinstance(o, float) and o != o:
        return None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    return o


R = clean(R)
dst = ROOT / "entregables" / ("resultados.json" if OUT == ROOT else "resultados_prueba.json")
dst.parent.mkdir(exist_ok=True)
json.dump(R, open(dst, "w", encoding="utf8"), indent=1, ensure_ascii=False, default=float)
print("ok", dst)
