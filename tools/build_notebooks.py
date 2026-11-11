"""Genera notebooks/02_entrenamiento.ipynb y notebooks/03_evaluacion_y_generalizacion.ipynb."""
from pathlib import Path
import nbformat as nbf

NB = Path(__file__).resolve().parents[1] / "notebooks"

SETUP = r'''
# ===== Configuración del entorno (Colab o local) =====
import os, sys, json, shutil, zipfile, time
from pathlib import Path
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    if not Path("/content/drive/MyDrive").exists():          # ya montado si se ejecuta desde el lanzador
        from google.colab import drive
        drive.mount("/content/drive")
    DRIVE = Path("/content/drive/MyDrive/ProyectoCafe")     # aquí está src/ y se guardan los resultados
    ROOT = Path("/content/proyecto")
    shutil.copytree(DRIVE / "src", ROOT / "src", dirs_exist_ok=True)
    (ROOT / "src" / "__init__.py").touch()
    os.environ["PROYECTO_ROOT"], os.environ["PROYECTO_OUT"] = str(ROOT), str(DRIVE)
    !pip -q install timm
else:
    ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
    os.environ.setdefault("PROYECTO_ROOT", str(ROOT)); os.environ.setdefault("PROYECTO_OUT", str(ROOT))
    DRIVE = Path(os.environ["PROYECTO_OUT"])
sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from src import config as C
C.FIG.mkdir(parents=True, exist_ok=True); C.RUNS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo:", DEVICE, "| ROOT:", ROOT, "| salidas:", C.OUT)
'''

DATA = r'''
# ===== Datos: descarga + versiones RoCoLe + caché uint8 (letterbox 256) =====
from src.data import (download_bracol, download_rocole, build_rocole_versions, build_tables,
                      get_caches, load_tables, split_fingerprint)
DRIVE_CACHE = DRIVE / "cache"
names = ["bracol", "rocole_raw", "rocole_crop", "rocole_mask"]
tables = ["bracol_leaf.csv", "rocole.csv", "meta.json"]
if IN_COLAB and DRIVE_CACHE.exists():                      # reutiliza caché y tablas de una sesión anterior
    C.CACHE.mkdir(parents=True, exist_ok=True); C.PROC.mkdir(parents=True, exist_ok=True)
    for f in list(DRIVE_CACHE.glob("*.pt")) + [DRIVE_CACHE / t for t in tables if (DRIVE_CACHE / t).exists()]:
        dst = (C.CACHE if f.suffix == ".pt" else C.PROC) / f.name
        if not dst.exists():
            shutil.copy(f, dst)
if not all((C.CACHE / f"{n}.pt").exists() for n in names) or not (C.PROC / "bracol_leaf.csv").exists():
    t = time.time()
    download_bracol(); download_rocole(); build_tables(); build_rocole_versions()
    print(f"Descarga y preprocesamiento: {time.time() - t:.0f} s")
fp = split_fingerprint()
print("Huella de la partición:", fp)
assert fp == "bd076d1914a2", "La partición no coincide con la del notebook 01"
caches = get_caches()
GUARDAR_CACHE = IN_COLAB and os.environ.get("CACHE_EN_DRIVE", "1") == "1"   # ~1.2 GB en Drive (opcional)
if GUARDAR_CACHE and not all((DRIVE_CACHE / f"{n}.pt").exists() for n in names):
    DRIVE_CACHE.mkdir(parents=True, exist_ok=True)
    for n in names:
        shutil.copy(C.CACHE / f"{n}.pt", DRIVE_CACHE / f"{n}.pt")
    for t in tables:
        shutil.copy(C.PROC / t, DRIVE_CACHE / t)
bra, roc = load_tables()
print({k: tuple(v.shape) for k, v in caches.items()})
print(bra.split.value_counts().to_dict())
'''


def nb(cells, path):
    n = nbf.v4.new_notebook()
    n["cells"] = [nbf.v4.new_markdown_cell(s.strip()) if k == "md" else nbf.v4.new_code_cell(s.strip()) for k, s in cells]
    n["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                     "language_info": {"name": "python"}, "accelerator": "GPU",
                     "colab": {"provenance": [], "gpuType": "T4"}}
    nbf.write(n, path)
    print("ok", path)


# ============================================================================ 02
nb([
("md", r'''
# 02 · Arquitecturas y entrenamiento
**Proyecto final — Redes Neuronales y Aprendizaje Profundo (EAFIT)** · Clasificación de enfermedades foliares del café

Se entrenan y comparan tres familias de modelos sobre BRACOL (5 clases: sana, minador, roya, phoma, cercospora):

| Corrida | Arquitectura | Referencia | Rol |
|---|---|---|---|
| `cnn_base` | CNN de 4 bloques entrenada **desde cero** | — | Línea base |
| `effnet_b0` | **EfficientNet-B0** preentrenada (ImageNet-1k), fine-tuning completo | Tan & Le, ICML 2019 | Modelo principal |
| `vit_tiny` | **ViT-Tiny/16** preentrenado (ImageNet-21k → 1k), fine-tuning completo | Dosovitskiy et al., ICLR 2021 | Comparación CNN vs. Transformer |
| `effnet_b0_sinaug` | EfficientNet-B0 solo con volteos | — | **Ablación** del aumento de datos |

**Modificaciones respecto a los modelos originales:** se reemplaza la capa de clasificación (1000 → 5 salidas) y se añade dropout
antes de ella (0.3 en EfficientNet, 0.1 en ViT). Todo el resto de la red se ajusta (*fine-tuning* completo).

Código fuente en `src/` (`data.py`, `models.py`, `train.py`, `evaluate.py`). Ejecutar en Colab con **GPU**
(Entorno de ejecución → Cambiar tipo → T4 GPU).
'''),
("code", SETUP),
("code", r'''
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || echo "sin GPU NVIDIA"
from src.train import hardware_info
import platform, psutil
hw = hardware_info(); hw["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1); hw["cpu_cores"] = os.cpu_count()
print(json.dumps(hw, indent=2))
json.dump(hw, open(C.RUNS / "hardware.json", "w"), indent=2)
'''),
("md", r'''
## 1. Datos y preprocesamiento
- **Letterbox 256×256**: se conserva la relación de aspecto (las lesiones no se deforman) y se rellena con el color medio de ImageNet,
  que tras la normalización vale ≈ 0, por lo que el relleno no aporta información.
- Imágenes cacheadas como tensores `uint8` para que la GPU no espere al disco.
- **Aumento (solo entrenamiento):** `RandomResizedCrop(224, escala 0.7–1)`, volteos H/V, rotaciones de 90° (la orientación de la hoja
  es arbitraria: en BRACOL es horizontal y en campo suele ser vertical), `ColorJitter` (brillo, contraste y saturación ±30 %).
- **Validación y prueba:** solo `Resize(224)` + normalización ImageNet (μ, σ de ImageNet porque los modelos fueron preentrenados ahí).
'''),
("code", DATA),
("code", r'''
from src.data import TensorImageDataset, get_transforms, denormalize
ds = TensorImageDataset(caches["bracol"][:8], bra.label.values[:8], get_transforms("train"))
fig, axes = plt.subplots(2, 6, figsize=(15, 5.4))
for r, idx in enumerate([0, 700]):
    axes[r, 0].imshow(caches["bracol"][idx].permute(1, 2, 0)); axes[r, 0].set_title(f"original ({C.CLASSES[bra.label[idx]]})", fontsize=9)
    ds_i = TensorImageDataset(caches["bracol"][idx:idx+1], [0], get_transforms("train"))
    for k in range(1, 6):
        axes[r, k].imshow(denormalize(ds_i[0][0])); axes[r, k].set_title(f"aumento {k}", fontsize=9)
for a in axes.ravel(): a.axis("off")
fig.suptitle("Letterbox 256 px y ejemplos de aumento de datos (solo entrenamiento)")
plt.tight_layout(); plt.savefig(C.FIG / "02_aumento_datos.png"); plt.show()
'''),
("md", r'''
## 2. Arquitecturas
- **CNN base:** 4 bloques `[Conv3×3 → BN → ReLU] ×2 → MaxPool2×2` (32, 64, 128, 256 filtros) → *Global Average Pooling* → Dropout(0.3) → FC(5).
  Entrada 3×224×224 → mapa 256×14×14 → vector 256 → 5 logits.
- **EfficientNet-B0:** stem Conv3×3 (32) + 16 bloques **MBConv** (convolución *inverted residual*: expansión 1×1 → convolución
  *depthwise* 3×3/5×5 → *Squeeze-and-Excitation* → proyección 1×1, con conexión residual) agrupados en 7 etapas; Conv1×1 (1280) → GAP → FC.
  Su escalamiento compuesto (profundidad, ancho y resolución) logra alta precisión con pocos parámetros y FLOPs (≈0.39 GFLOPs).
- **ViT-Tiny/16:** la imagen se corta en 14×14 = 196 *patches* de 16×16 que se proyectan a 192 dimensiones; se añade un token `[CLS]` y
  *embeddings* posicionales; 12 bloques Transformer (*multi-head self-attention* de 3 cabezas + MLP 768); el token `[CLS]` final → FC.
'''),
("code", r'''
from src.models import build_model, count_params
rows = []
for arch in ["cnn", "effnet_b0", "vit_tiny"]:
    m = build_model(arch, pretrained=False)
    with torch.no_grad():
        out = m(torch.zeros(1, 3, C.IMG_SIZE, C.IMG_SIZE))
    rows.append({"arquitectura": arch, **count_params(m), "entrada": "3×224×224", "salida": tuple(out.shape)})
arch_df = pd.DataFrame(rows); display(arch_df)
arch_df.to_csv(C.RUNS / "arquitecturas.csv", index=False)
print(build_model("cnn"))
'''),
("md", r'''
## 3. Configuración experimental
| Hiperparámetro | Valor | Justificación |
|---|---|---|
| Optimizador | AdamW | Adam con *weight decay* desacoplado: estándar para fine-tuning de CNNs y Transformers |
| Learning rate | 1e-3 (CNN) · 3e-4 (EffNet) · 1e-4 (ViT) | Desde cero tolera LR alto; los preentrenados necesitan LR bajo para no destruir lo aprendido; los ViT son más sensibles |
| Programación del LR | *warmup* lineal 2 épocas + coseno | El *warmup* estabiliza el inicio (cabeza aleatoria); el coseno afina al final |
| Batch size | 32 | Compromiso entre memoria de GPU, ruido del gradiente y número de pasos (1179 imágenes → 36 pasos/época) |
| Épocas máx. / paciencia | 40/10 (CNN) · 25/8 (preentrenados) | *Early stopping* con F1-macro de validación |
| Pérdida | Entropía cruzada **ponderada por clase** + *label smoothing* 0.1 | Desbalance 3.6× (cercospora minoritaria); el *smoothing* reduce la sobreconfianza |
| Weight decay | 1e-4 / 1e-2 / 5e-2 | Regularización; mayor en ViT (más propenso a sobreajustar con pocos datos) |
| Precisión | Mixta FP16 (AMP) en GPU | Acelera el entrenamiento ~2× sin pérdida de exactitud |
| Selección del modelo | Época con **mayor F1-macro en validación** | El conjunto de prueba no se usa para ninguna decisión |
'''),
("code", r'''
from src.train import RUNS_CFG, DEFAULTS, train
display(pd.DataFrame({k: {**DEFAULTS, **v} for k, v in RUNS_CFG.items()}).T)
'''),
("md", "## 4. Entrenamiento\nCada corrida guarda `history.csv`, `best.pt` y `summary.json` en Drive (`ProyectoCafe/runs/`). Si una corrida ya existe se omite, de modo que una desconexión de Colab no obliga a repetir todo."),
("code", r'''
summaries = {}
for run in RUNS_CFG:
    summaries[run] = train(run, caches=caches)
    print(f"→ {run}: mejor F1 val = {summaries[run]['best_val_f1']:.4f} (época {summaries[run]['best_epoch']}), "
          f"{summaries[run]['train_time_s'] / 60:.1f} min\n")
'''),
("code", r'''
# Registro de consola del entrenamiento (si las corridas se lanzaron con `python -m src.train`)
log = C.RUNS / "train_log.txt"
print(log.read_text(encoding="utf8", errors="replace") if log.exists() else "Sin registro externo: ver salida de la celda anterior.")
'''),
("md", "## 5. Curvas de aprendizaje"),
("code", r'''
NOMBRES = {"cnn_base": "CNN base (desde cero)", "effnet_b0": "EfficientNet-B0", "vit_tiny": "ViT-Tiny/16",
           "effnet_b0_sinaug": "EfficientNet-B0 sin aumento"}
fig, axes = plt.subplots(len(RUNS_CFG), 3, figsize=(16, 3.3 * len(RUNS_CFG)))
for r, run in enumerate(RUNS_CFG):
    h = pd.read_csv(C.RUNS / run / "history.csv"); be = summaries[run]["best_epoch"]
    for ax, met, ttl in zip(axes[r], ["loss", "acc", "f1"], ["Pérdida (CE ponderada)", "Accuracy", "F1-macro"]):
        ax.plot(h.epoch, h[f"train_{met}"], "-o", ms=3, label="entrenamiento")
        ax.plot(h.epoch, h[f"val_{met}"], "-o", ms=3, label="validación")
        ax.axvline(be, color="gray", ls="--", lw=1, label=f"mejor época ({be})")
        ax.set_title(f"{NOMBRES[run]} — {ttl}", fontsize=10); ax.set_xlabel("Época")
    axes[r, 0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(C.FIG / "02_curvas_todas.png"); plt.show()

# figura compacta para la presentación: F1 y pérdida de validación de los 3 modelos principales
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for run, c in zip(["cnn_base", "effnet_b0", "vit_tiny"], ["#8E5572", "#4C9F70", "#5B6C8F"]):
    h = pd.read_csv(C.RUNS / run / "history.csv")
    axes[0].plot(h.epoch, h.train_loss, "--", color=c, alpha=.6); axes[0].plot(h.epoch, h.val_loss, "-", color=c, label=NOMBRES[run])
    axes[1].plot(h.epoch, h.train_f1, "--", color=c, alpha=.6); axes[1].plot(h.epoch, h.val_f1, "-", color=c, label=NOMBRES[run])
axes[0].set_title("Pérdida (— validación, -- entrenamiento)"); axes[1].set_title("F1-macro (— validación, -- entrenamiento)")
for a in axes: a.set_xlabel("Época"); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig(C.FIG / "02_curvas_comparacion.png"); plt.show()
'''),
("code", r'''
res = []
for run, s in summaries.items():
    h = pd.read_csv(C.RUNS / run / "history.csv"); b = h.loc[h.epoch == s["best_epoch"]].iloc[0]
    res.append({"corrida": run, "parámetros (M)": round(s["params"]["total"] / 1e6, 2), "épocas": s["epochs_run"],
                "mejor época": s["best_epoch"], "F1 train": round(b.train_f1, 4), "F1 val": round(b.val_f1, 4),
                "brecha F1 (train−val)": round(b.train_f1 - b.val_f1, 4), "loss train": round(b.train_loss, 4),
                "loss val": round(b.val_loss, 4), "tiempo (min)": round(s["train_time_s"] / 60, 2),
                "s/época": round(h.epoch_time_s.mean(), 1)})
train_df = pd.DataFrame(res); display(train_df)
train_df.to_csv(C.RUNS / "resumen_entrenamiento.csv", index=False)
'''),
("md", r'''
### Análisis de las curvas
*(Se completa con los resultados obtenidos: ver celda siguiente en la versión entregada.)*
'''),
], NB / "02_entrenamiento.ipynb")


# ============================================================================ 03
nb([
("md", r'''
# 03 · Evaluación, análisis de errores y generalización externa
1. Prueba interna en **BRACOL test** (253 hojas nunca vistas) para los 4 modelos.
2. Análisis de errores del mejor modelo (elegido por F1 de **validación**).
3. **Generalización externa en RoCoLe** (otra variedad, otro país, otra cámara, hoja en campo), en tres versiones: `raw`, `crop`, `mask`.
4. Comportamiento ante una clase nunca vista (**arañita roja**, *out-of-distribution*).
5. **Grad-CAM**: ¿dónde mira el modelo?
'''),
("code", SETUP),
("code", DATA),
("code", r'''
from src.evaluate import load_trained, predict_proba, multiclass_metrics, healthy_vs_diseased, latency_ms, GradCAM
from src.train import RUNS_CFG
NOMBRES = {"cnn_base": "CNN base", "effnet_b0": "EfficientNet-B0", "vit_tiny": "ViT-Tiny", "effnet_b0_sinaug": "EffNet-B0 sin aum."}
test_idx = bra.index[bra.split == "test"].to_numpy(); y_test = bra.label.values[test_idx]
ind = roc.index[roc.label >= 0].to_numpy(); y_roc = roc.label.values[ind]          # sana (0) / roya (2)
ood = roc.index[roc.rocole_class == "red_spider_mite"].to_numpy()

models, P, summ = {}, {}, {}
for run in RUNS_CFG:
    m, arch, s = load_trained(run, DEVICE); models[run] = (m, arch); summ[run] = s
    P[(run, "test")] = predict_proba(m, caches["bracol"][test_idx])
    for v in C.ROCOLE_VERSIONS:
        P[(run, v)] = predict_proba(m, caches[f"rocole_{v}"])
np.savez_compressed(C.RUNS / "probabilidades.npz", **{f"{r}__{k}": p for (r, k), p in P.items()})
BEST = max(["cnn_base", "effnet_b0", "vit_tiny"], key=lambda r: summ[r]["best_val_f1"])
print("Mejor modelo según F1 de VALIDACIÓN:", BEST, {r: round(summ[r]["best_val_f1"], 4) for r in summ})
'''),
("md", "## 1. Resultados en BRACOL test"),
("code", r'''
rows, M = [], {}
for run in RUNS_CFG:
    M[run] = multiclass_metrics(y_test, P[(run, "test")])
    m, _ = models[run]
    lat_gpu = latency_ms(m) if DEVICE.type == "cuda" else float("nan")
    lat_cpu = latency_ms(m.cpu(), torch.device("cpu"), n=20); m.to(DEVICE)
    rows.append({"modelo": NOMBRES[run], "accuracy": M[run]["accuracy"], "precision_macro": M[run]["precision_macro"],
                 "recall_macro": M[run]["recall_macro"], "f1_macro": M[run]["f1_macro"],
                 "params_M": summ[run]["params"]["total"] / 1e6, "entrenamiento_min": summ[run]["train_time_s"] / 60,
                 "ms_img_gpu": lat_gpu, "ms_img_cpu": lat_cpu})
test_df = pd.DataFrame(rows).round(4); display(test_df)
test_df.to_csv(C.RUNS / "resultados_test.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 3.6))
x = np.arange(len(test_df)); w = 0.38
ax.bar(x - w/2, test_df.accuracy, w, label="Accuracy", color="#5B6C8F"); ax.bar(x + w/2, test_df.f1_macro, w, label="F1-macro", color="#4C9F70")
for i, (a, f) in enumerate(zip(test_df.accuracy, test_df.f1_macro)):
    ax.text(i - w/2, a + .01, f"{a:.3f}", ha="center", fontsize=8); ax.text(i + w/2, f + .01, f"{f:.3f}", ha="center", fontsize=8)
ax.set_xticks(x, test_df.modelo); ax.set_ylim(0, 1.08); ax.set_title("BRACOL test (n = %d)" % len(y_test)); ax.legend(loc="lower right")
plt.tight_layout(); plt.savefig(C.FIG / "03_test_comparacion.png"); plt.show()
'''),
("code", r'''
cm = np.array(M[BEST]["confusion"]); cmn = cm / cm.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cmn, cmap="Greens", vmin=0, vmax=1)
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{cm[i, j]}\n({cmn[i, j]:.0%})", ha="center", va="center", fontsize=8, color="white" if cmn[i, j] > .6 else "black")
ax.set_xticks(range(5), C.CLASSES, rotation=30); ax.set_yticks(range(5), C.CLASSES)
ax.set_xlabel("Predicción"); ax.set_ylabel("Clase real"); ax.set_title(f"Matriz de confusión — {NOMBRES[BEST]} (BRACOL test)")
plt.colorbar(im, fraction=.046); plt.tight_layout(); plt.savefig(C.FIG / "03_matriz_confusion_test.png"); plt.show()
pc = pd.DataFrame(M[BEST]["per_class"]).T.round(3); display(pc)
pc.to_csv(C.RUNS / "reporte_por_clase_test.csv")
'''),
("code", r'''
# ¿Los errores se concentran en hojas con varios estreses simultáneos?
tb = bra.loc[test_idx].copy(); tb["pred"] = P[(BEST, "test")].argmax(1); tb["ok"] = tb.pred == tb.label
tb["n_estreses"] = tb[["miner", "rust", "phoma", "cercospora"]].sum(1)
tb["tipo"] = np.where(tb.n_estreses > 1, "varios estreses", np.where(tb.n_estreses == 1, "un estrés", "sana"))
multi = tb.groupby("tipo").ok.agg(["mean", "count"]).rename(columns={"mean": "accuracy", "count": "n"}); display(multi.round(3))
# ¿y con la severidad?
display(tb.groupby("severity").ok.agg(["mean", "count"]).rename(columns={"mean": "accuracy", "count": "n"}).round(3))

err = tb[~tb.ok]
k = min(6, len(err))
if k:
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.2))
    for ax in axes.ravel(): ax.axis("off")
    for ax, (_, r) in zip(axes.ravel(), err.sample(k, random_state=0).iterrows()):
        x = caches["bracol"][r.name][:, 64:192, :]            # recorta el relleno del letterbox (hoja 2:1)
        ax.imshow(x.permute(1, 2, 0))
        otros = [n for n, c in zip(["minador", "roya", "phoma", "cercospora"], ["miner", "rust", "phoma", "cercospora"]) if r[c]]
        ax.set_title(f"real: {C.CLASSES[r.label]} → pred: {C.CLASSES[r.pred]}\npresentes: {', '.join(otros) or '—'}", fontsize=12)
    fig.suptitle(f"Ejemplos de errores de {NOMBRES[BEST]} en BRACOL test ({len(err)} de {len(tb)})", fontsize=13)
    plt.tight_layout(); plt.savefig(C.FIG / "03_errores_test.png"); plt.show()
'''),
("md", r'''
## 2. Generalización externa — RoCoLe
RoCoLe solo contiene **sana** y **roya** dentro del espacio de etiquetas de BRACOL. Se reportan:
- **Accuracy estricta (5 clases):** acierto exacto; el modelo puede equivocarse prediciendo cualquiera de las 5 clases.
- **Tarea binaria sana vs. enferma:** puntaje = 1 − p(sana). Mide si el modelo detecta *que hay enfermedad*, aunque confunda cuál.
  Se calcula también en BRACOL test para comparar en igualdad de condiciones.
'''),
("code", r'''
rows = []
for run in RUNS_CFG:
    hb = healthy_vs_diseased(y_test, P[(run, "test")])
    rows.append({"modelo": NOMBRES[run], "dominio": "BRACOL test", "acc_5clases": M[run]["accuracy"], **hb})
    for v in C.ROCOLE_VERSIONS:
        p = P[(run, v)][ind]
        rows.append({"modelo": NOMBRES[run], "dominio": f"RoCoLe {v}", "acc_5clases": float((p.argmax(1) == y_roc).mean()),
                     **healthy_vs_diseased(y_roc, p)})
ext_df = pd.DataFrame(rows).round(4); display(ext_df)
ext_df.to_csv(C.RUNS / "resultados_externos.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
doms = ["BRACOL test", "RoCoLe raw", "RoCoLe crop", "RoCoLe mask"]; cols = ["#4C9F70", "#C8553D", "#C9A227", "#5B6C8F"]
for ax, met, ttl in zip(axes, ["acc_5clases", "auroc"], ["Accuracy (5 clases)", "AUROC sana vs. enferma"]):
    mods = [NOMBRES[r] for r in RUNS_CFG]; x = np.arange(len(mods)); w = 0.2
    for k, (d, c) in enumerate(zip(doms, cols)):
        vals = [ext_df[(ext_df.modelo == m) & (ext_df.dominio == d)][met].item() for m in mods]
        ax.bar(x + (k - 1.5) * w, vals, w, label=d, color=c)
    ax.set_xticks(x, mods, fontsize=9); ax.set_ylim(0, 1.05); ax.set_title(ttl)
    ax.axhline(0.5 if met == "auroc" else 0, color="k", lw=.6, ls=":")
axes[0].legend(fontsize=8, ncol=2)
plt.tight_layout(); plt.savefig(C.FIG / "03_generalizacion.png"); plt.show()
'''),
("code", r'''
fig, axes = plt.subplots(1, 3, figsize=(16, 3.2))
for ax, v in zip(axes, C.ROCOLE_VERSIONS):
    pr = P[(BEST, v)][ind].argmax(1)
    cmx = np.array([[np.sum((y_roc == t) & (pr == j)) for j in range(5)] for t in (0, 2)])
    cmxn = cmx / cmx.sum(1, keepdims=True)
    ax.imshow(cmxn, cmap="Oranges", vmin=0, vmax=1)
    for i in range(2):
        for j in range(5):
            ax.text(j, i, f"{cmx[i, j]}\n{cmxn[i, j]:.0%}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(5), C.CLASSES, rotation=30, fontsize=8); ax.set_yticks([0, 1], ["sana", "roya"])
    ax.set_title(f"{NOMBRES[BEST]} — RoCoLe {v}", fontsize=10); ax.set_xlabel("Predicción")
plt.tight_layout(); plt.savefig(C.FIG / "03_confusion_rocole.png"); plt.show()
'''),
("code", r'''
# Desglose por nivel de roya y por lado de la hoja (mejor modelo)
r_ind = roc.loc[ind].copy()
for v in C.ROCOLE_VERSIONS:
    pr = P[(BEST, v)][ind]
    r_ind[f"enferma_{v}"] = pr.argmax(1) != 0
    r_ind[f"acierto_{v}"] = pr.argmax(1) == y_roc
lvl = r_ind.groupby("rocole_class")[[f"enferma_{v}" for v in C.ROCOLE_VERSIONS]].mean().round(3)
lvl.columns = [f"% predicho enfermo ({v})" for v in C.ROCOLE_VERSIONS]; display(lvl)
lado = r_ind.groupby(["lado", "clase"])[[f"acierto_{v}" for v in C.ROCOLE_VERSIONS]].mean().round(3); display(lado)
lvl.to_csv(C.RUNS / "rocole_por_nivel.csv"); lado.to_csv(C.RUNS / "rocole_por_lado.csv")

fig, ax = plt.subplots(figsize=(8, 3.6))
order = ["healthy", "rust_level_1", "rust_level_2", "rust_level_3", "rust_level_4"]
for v, c in zip(C.ROCOLE_VERSIONS, ["#C8553D", "#C9A227", "#5B6C8F"]):
    ax.plot(order, lvl.loc[order, f"% predicho enfermo ({v})"], "-o", color=c, label=v)
ax.set_ylabel("Fracción predicha como enferma"); ax.set_ylim(0, 1.05); ax.legend()
ax.set_title(f"{NOMBRES[BEST]}: detección de enfermedad según severidad de roya (RoCoLe)")
plt.tight_layout(); plt.savefig(C.FIG / "03_rocole_severidad.png"); plt.show()
'''),
("md", "## 3. Clase nunca vista: arañita roja (*out-of-distribution*)"),
("code", r'''
from sklearn.metrics import roc_auc_score
rows = []
for v in C.ROCOLE_VERSIONS:
    pr = P[(BEST, v)][ood]
    d = pd.Series(pr.argmax(1)).map(dict(enumerate(C.CLASSES))).value_counts(normalize=True).reindex(C.CLASSES).fillna(0)
    msp_in = P[(BEST, v)][ind].max(1); msp_ood = pr.max(1)
    auc = roc_auc_score(np.r_[np.zeros(len(msp_in)), np.ones(len(msp_ood))], -np.r_[msp_in, msp_ood])
    rows.append({"versión": v, **d.round(3).to_dict(), "confianza media OOD": msp_ood.mean(), "confianza media sana/roya": msp_in.mean(),
                 "AUROC detección OOD (MSP)": auc})
ood_df = pd.DataFrame(rows).round(3); display(ood_df); ood_df.to_csv(C.RUNS / "ood_aranita.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 3.6))
ax.hist(P[(BEST, "test")].max(1), bins=25, alpha=.6, density=True, label="BRACOL test", color="#4C9F70")
ax.hist(P[(BEST, "raw")][ind].max(1), bins=25, alpha=.6, density=True, label="RoCoLe sana/roya (raw)", color="#C8553D")
ax.hist(P[(BEST, "raw")][ood].max(1), bins=25, alpha=.6, density=True, label="RoCoLe arañita roja (raw)", color="#5B6C8F")
ax.set_xlabel("Confianza máxima softmax"); ax.set_ylabel("Densidad"); ax.legend(fontsize=8)
ax.set_title(f"{NOMBRES[BEST]}: confianza dentro y fuera del dominio")
plt.tight_layout(); plt.savefig(C.FIG / "03_confianza_ood.png"); plt.show()
'''),
("md", "## 4. Grad-CAM — ¿en qué se fija el modelo?"),
("code", r'''
from src.data import get_transforms
tf = get_transforms("eval")

def overlay(ax, x_uint8, cam, title, crop=None, fs=12):
    img = torch.nn.functional.interpolate(x_uint8[None].float(), size=C.IMG_SIZE, mode="bilinear")[0].permute(1, 2, 0).byte().numpy()
    if crop == "h":                      # hoja horizontal (BRACOL): quita las bandas de relleno
        img, cam = img[56:168], cam[56:168]
    elif crop == "v":                    # hoja vertical (RoCoLe): quita las bandas laterales
        img, cam = img[:, 56:168], cam[:, 56:168]
    ax.imshow(img); ax.imshow(cam, cmap="jet", alpha=.42); ax.set_title(title, fontsize=fs); ax.axis("off")

# (a) BRACOL test: un ejemplo bien clasificado de 3 clases enfermas, comparando los 3 modelos
gcs = {r: GradCAM(*models[r]) for r in ["cnn_base", "effnet_b0", "vit_tiny"]}
cls_show = [2, 1, 3]                     # roya, minador, phoma
picks = [tb[(tb.label == c) & tb.ok].index[0] if ((tb.label == c) & tb.ok).any() else tb[tb.label == c].index[0] for c in cls_show]
fig, axes = plt.subplots(4, 3, figsize=(12, 8.6))
for j, idx in enumerate(picks):
    x = caches["bracol"][idx]
    axes[0, j].imshow(x[:, 64:192, :].permute(1, 2, 0)); axes[0, j].set_title(f"real: {C.CLASSES[bra.label[idx]]}", fontsize=13); axes[0, j].axis("off")
    for i, r in enumerate(gcs, start=1):
        cam, t, p = gcs[r](tf(x)); overlay(axes[i, j], x, cam, f"{NOMBRES[r]} → {C.CLASSES[t]} ({p:.2f})", crop="h")
plt.tight_layout(); plt.savefig(C.FIG / "03_gradcam_bracol.png"); plt.show()
'''),
("code", r'''
# (b) RoCoLe: misma hoja en raw / crop / mask con el mejor modelo (roya y sana)
gc = GradCAM(*models[BEST])
sel = list(roc[(roc.rocole_class.isin(["rust_level_2", "rust_level_3"]))].sample(2, random_state=3).index) + \
      list(roc[roc.rocole_class == "healthy"].sample(1, random_state=3).index)
fig, axes = plt.subplots(len(sel), 3, figsize=(8.4, 11))
for i, idx in enumerate(sel):
    for j, v in enumerate(C.ROCOLE_VERSIONS):
        x = caches[f"rocole_{v}"][idx]; cam, t, p = gc(tf(x))
        overlay(axes[i, j], x, cam, f"{v}: {roc.clase[idx]} → {C.CLASSES[t]} ({p:.2f})", crop="v")
plt.tight_layout(); plt.savefig(C.FIG / "03_gradcam_rocole.png"); plt.show()

# figura adicional: muchas hojas de RoCoLe raw vs mask (para cuantificar visualmente)
rng = np.random.default_rng(0)
many = rng.choice(ind, 8, replace=False)
fig, axes = plt.subplots(2, 8, figsize=(16, 5.4))
for j, idx in enumerate(many):
    for i, v in enumerate(["raw", "mask"]):
        x = caches[f"rocole_{v}"][idx]; cam, t, p = gc(tf(x))
        overlay(axes[i, j], x, cam, f"{v}: {roc.clase[idx][:4]}→{C.CLASSES[t][:5]}", crop="v", fs=10)
plt.tight_layout(); plt.savefig(C.FIG / "03_gradcam_rocole_muestra.png"); plt.show()
'''),
("md", r'''
## 5. Fotos propias (opcional)
Si el equipo sube fotos de hojas de café a `MyDrive/ProyectoCafe/fotos_propias/` (subcarpetas con el nombre de la clase real,
p. ej. `sana/`, `roya/`, o sin subcarpeta si no se conoce), esta celda las clasifica con el mejor modelo.
'''),
("code", r'''
from PIL import Image, ImageOps
from src.data import letterbox
fp = DRIVE / "fotos_propias"
files = sorted([p for p in fp.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]) if fp.exists() else []
if not files:
    print("No hay fotos propias en", fp)
else:
    xs = torch.stack([torch.from_numpy(np.asarray(letterbox(ImageOps.exif_transpose(Image.open(f))))).permute(2, 0, 1) for f in files])
    pr = predict_proba(models[BEST][0], xs)
    n = len(files); cols = min(5, n); rws = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rws, cols, figsize=(3.2 * cols, 3.4 * rws), squeeze=False)
    for ax in axes.ravel(): ax.axis("off")
    for ax, f, x, p in zip(axes.ravel(), files, xs, pr):
        real = f.parent.name if f.parent != fp else "?"
        cam, t, conf = gc(tf(x)); overlay(ax, x, cam, f"real: {real} → {C.CLASSES[p.argmax()]} ({p.max():.2f})")
    plt.tight_layout(); plt.savefig(C.FIG / "03_fotos_propias.png"); plt.show()
'''),
("code", r'''
# ===== Guardar resumen de métricas y empaquetar resultados =====
out = {"best_model": BEST, "val_f1": {r: summ[r]["best_val_f1"] for r in summ},
       "test": {r: {k: M[r][k] for k in ("accuracy", "f1_macro", "precision_macro", "recall_macro", "per_class", "confusion")} for r in M},
       "externo": ext_df.to_dict("records"), "ood": ood_df.to_dict("records"),
       "rocole_por_nivel": lvl.reset_index().to_dict("records"), "errores_multiestres": multi.reset_index().to_dict("records"),
       "n": {"test": int(len(y_test)), "rocole_in": int(len(ind)), "rocole_ood": int(len(ood))}}
json.dump(out, open(C.RUNS / "metrics.json", "w"), indent=2, default=float)
zp = DRIVE / "resultados.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for p in list(C.RUNS.rglob("*")) + list(C.FIG.rglob("*.png")):
        if p.is_file() and p.suffix != ".pt":
            z.write(p, p.relative_to(DRIVE))
print("Guardado", zp, f"{zp.stat().st_size / 1e6:.1f} MB")
'''),
], NB / "03_evaluacion_y_generalizacion.ipynb")
