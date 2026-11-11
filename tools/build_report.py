"""Genera entregables/Informe_ProyectoFinal.pdf (≤ 3 páginas) desde entregables/resultados.json y figures/.
Uso: python tools/build_report.py [resultados.json] [salida.pdf] [carpeta_figuras;carpeta2]"""
import json
import sys
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "entregables" / "resultados.json"
OUTF = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "entregables" / "Informe_ProyectoFinal.pdf"
FIGS = [Path(p) for p in (sys.argv[3].split(";") if len(sys.argv) > 3 else [str(ROOT / "figures")])]
R = json.load(open(DATA, encoding="utf8"))

COFFEE, LEAF, MUTE, LINE = colors.HexColor("#3B2A20"), colors.HexColor("#2E7D4F"), colors.HexColor("#5E6B63"), colors.HexColor("#D5DBD3")
base = ParagraphStyle("b", fontName="Times-Roman", fontSize=9.6, leading=11.6, alignment=TA_JUSTIFY, spaceAfter=3)
h1 = ParagraphStyle("h1", fontName="Times-Bold", fontSize=15, leading=18, alignment=TA_CENTER, textColor=COFFEE, spaceAfter=2)
sub = ParagraphStyle("s", fontName="Times-Italic", fontSize=9.5, leading=11.5, alignment=TA_CENTER, textColor=MUTE, spaceAfter=6)
h2 = ParagraphStyle("h2", fontName="Times-Bold", fontSize=11, leading=13, textColor=COFFEE, spaceBefore=5, spaceAfter=2)
cap = ParagraphStyle("c", fontName="Times-Italic", fontSize=8.3, leading=9.8, textColor=MUTE, alignment=TA_JUSTIFY, spaceAfter=4)
cell = ParagraphStyle("t", fontName="Times-Roman", fontSize=8.4, leading=9.8)
cellb = ParagraphStyle("tb", parent=cell, fontName="Times-Bold", textColor=colors.white)


def P(t, st=base):
    return Paragraph(t, st)


def fig(name, width_cm):
    for d in FIGS:
        p = d / name
        if p.exists():
            w, h = PILImage.open(p).size
            return Image(str(p), width=width_cm * cm, height=width_cm * cm * h / w)
    raise FileNotFoundError(name)


def table(rows, widths, bold_first_col=False):
    data = [[Paragraph(str(c), cellb if i == 0 else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], hAlign="CENTER")
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), COFFEE), ("GRID", (0, 0), (-1, -1), 0.4, LINE),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
    return t


pct = lambda v, d=1: f"{100 * v:.{d}f}".replace(".", ",") + " %"
num = lambda v, d=3: f"{v:.{d}f}".replace(".", ",")
NOM = {"cnn_base": "CNN base (desde cero)", "effnet_b0": "EfficientNet-B0", "vit_tiny": "ViT-Tiny/16", "effnet_b0_sinaug": "EfficientNet-B0 sin aumento"}
NOMDF = {"cnn_base": "CNN base", "effnet_b0": "EfficientNet-B0", "vit_tiny": "ViT-Tiny", "effnet_b0_sinaug": "EffNet-B0 sin aum."}
ext = lambda run, dom: next(e for e in R["externo"] if e["modelo"] == NOMDF[run] and e["dominio"] == dom)
D, RU, T, BEST = R["datos"], R["runs"], R["test"], R["best"]
bt, raw, crop, mask = ext(BEST, "BRACOL test"), ext(BEST, "RoCoLe raw"), ext(BEST, "RoCoLe crop"), ext(BEST, "RoCoLe mask")
hw = R["hardware"]
equipo = f"GPU {hw['gpu']} ({hw['gpu_mem_gb']} GB)" if hw.get("gpu") else "CPU Intel Core i5-1155G7 (4 núcleos), sin GPU"
ood = next(o for o in R["ood"] if o["versión"] == "raw")
ood_cls = max(["sana", "minador", "roya", "phoma", "cercospora"], key=lambda c: ood[c])
TXT = json.load(open(ROOT / "tools" / "textos_informe.json", encoding="utf8")) if (ROOT / "tools" / "textos_informe.json").exists() else {}

S = []
S += [P("Clasificación de enfermedades foliares del café y evaluación de su generalización ante cambio de dominio", h1),
      P("Cristobal Rodriguez Rojas · Juan Esteban Villada · Cristian David Cabezas Jimenez — Redes Neuronales y Aprendizaje Profundo, Universidad EAFIT, noviembre de 2026", sub)]

S += [P("1. Problema y objetivos", h2),
      P("La roya del café (<i>Hemileia vastatrix</i>) y otras enfermedades foliares reducen la productividad del cultivo; su diagnóstico "
        "visual depende de expertos. Las redes convolucionales pueden aprender estos patrones de textura y color, pero los datasets "
        "públicos se capturan en condiciones controladas, por lo que es incierto si un modelo entrenado así funciona en campo. "
        "<b>Objetivo general:</b> desarrollar y evaluar un modelo de Deep Learning que clasifique enfermedades y plagas en hojas de café "
        "y medir cuantitativamente su generalización a datos de otra región, variedad y condición de captura. "
        "<b>Objetivos específicos:</b> (1) construir un pipeline de preprocesamiento y aumento con partición estratificada sin fuga de "
        "información; (2) entrenar y comparar una CNN desde cero, EfficientNet-B0 y ViT-Tiny con accuracy, F1-macro y costo computacional; "
        "(3) evaluar el mejor modelo en un dataset externo, cuantificar la caída de rendimiento y explicarla con Grad-CAM.")]

sp = D["split"]
S += [P("2. Datos y preprocesamiento", h2),
      P(f"<b>BRACOL</b> (Krohling et al., 2019): {D['bracol_total']} fotos de hojas de café arábica (Brasil), desprendidas y sobre fondo "
        f"blanco, etiquetadas con su estrés predominante. Se descartan {D['bracol_indeterminadas']} «indeterminadas» y quedan "
        f"{D['bracol_usadas']} en 5 clases: " + ", ".join(f"{k} {v}" for k, v in D["por_clase"].items()) +
        f" (desbalance 3,6×; el {str(D['multiestres_pct']).replace('.', ',')} % tiene varios estreses). Partición estratificada 70/15/15 con semilla fija: "
        f"{sp['train']}/{sp['val']}/{sp['test']} imágenes. <b>RoCoLe</b> (Parraga-Alava et al., 2019), usado <i>solo</i> como conjunto externo: "
        f"{D['rocole_total']} fotos de café robusta (Ecuador) tomadas en la planta, en campo: {D['rocole_sana']} sanas, {D['rocole_roya']} con roya "
        f"(niveles 1–4) y {D['rocole_ood']} con arañita roja, una plaga ausente en BRACOL. El EDA cuantificó el cambio de dominio: el brillo medio "
        f"baja de 0,62 a 0,41 y la fracción de fondo blanco de 30 % a 1 %; además las fotos de RoCoLe traen orientación EXIF (se corrige). "
        "<b>Preprocesamiento:</b> <i>letterbox</i> a 256×256 (conserva la geometría de las lesiones; relleno con la media de ImageNet), "
        "aumento solo en entrenamiento (recorte aleatorio 0,7–1, volteos, rotaciones de 90°, ColorJitter ±30 %), redimensión a 224 px y "
        "normalización con μ/σ de ImageNet. Con las máscaras de segmentación de RoCoLe se crearon tres versiones de cada foto externa: "
        "<i>raw</i>, <i>crop</i> (recorte a la hoja) y <i>mask</i> (fondo blanco), para aislar el efecto del fondo.")]

S += [P("3. Modelos y entrenamiento", h2),
      P(f"Se comparan (i) una <b>CNN base</b> desde cero (4 bloques [Conv3×3-BN-ReLU]×2 + MaxPool, GAP, FC; {RU['cnn_base']['params_M']:.2f} M parámetros); "
        f"(ii) <b>EfficientNet-B0</b> (Tan y Le, 2019; 16 bloques MBConv con convolución <i>depthwise</i> y Squeeze-Excitation; "
        f"{RU['effnet_b0']['params_M']:.2f} M) y (iii) <b>ViT-Tiny/16</b> (Dosovitskiy et al., 2021; 196 parches, 12 bloques de "
        f"<i>self-attention</i>; {RU['vit_tiny']['params_M']:.2f} M), ambos preentrenados en ImageNet y con la cabeza reemplazada por 5 salidas "
        "con dropout (única modificación; <i>fine-tuning</i> completo). Entrenamiento: AdamW, LR 1e-3 / 3e-4 / 1e-4 (CNN / EffNet / ViT), "
        "<i>warmup</i> de 2 épocas + coseno, batch 32, entropía cruzada ponderada por clase (pesos N/(K·n<sub>c</sub>) de train) con "
        "<i>label smoothing</i> 0,1 y <i>early stopping</i> por F1-macro de validación (máx. 40/25 épocas). Ablación: EfficientNet-B0 solo con volteos. "
        f"Equipo: {equipo}, {round(hw['ram_gb'])} GB de RAM, PyTorch {str(hw['torch']).split('+')[0]} y timm.")]

rows = [["Modelo", "Accuracy", "F1-macro", "Mejor ép.", "Parám.", "Entrenam.", "ms/img CPU"]]
lat = {l["modelo"]: l for l in R["latencia"]}
for r in ["cnn_base", "effnet_b0", "vit_tiny", "effnet_b0_sinaug"]:
    rows.append([NOM[r], pct(T[r]["accuracy"]), pct(T[r]["f1_macro"]), f"{RU[r]['best_epoch']}/{RU[r]['epochs_run']}",
                 f"{RU[r]['params_M']:.2f} M", f"{RU[r]['train_min']:.0f} min", f"{lat[NOMDF[r]]['ms_img_cpu']:.0f}"])
S += [P("4. Resultados", h2),
      KeepTogether([table(rows, [4.4, 1.7, 1.7, 1.7, 1.9, 2.3, 2.1]),
                    P(f"Tabla 1. BRACOL test (n = {R['n']['test']}). El mejor modelo según F1 de validación es {NOM[BEST]}. "
                      "Esgario et al. (2020) reportan 95,24 % de accuracy en este dataset con ResNet-50 multitarea (partición distinta).", cap)]),
      P(TXT.get("resultados_test", f"{NOM[BEST]} obtiene {pct(T[BEST]['accuracy'])} de accuracy y {pct(T[BEST]['f1_macro'])} de F1-macro en test.")),
      KeepTogether([fig("02_curvas_comparacion.png", 15.5),
                    P("Figura 1. Curvas de pérdida y F1-macro (línea continua: validación; punteada: entrenamiento).", cap)]),
      P(TXT.get("curvas", "")), ]

rows2 = [["Dominio (" + NOMDF[BEST] + ")", "Acc. 5 clases", "Acc. sana/enferma", "Recall enferma", "Recall sana", "AUROC"]]
for e in (bt, raw, crop, mask):
    rows2.append([e["dominio"], pct(e["acc_5clases"]), pct(e["accuracy"]), pct(e["recall_enferma"]), pct(e["recall_sana"]), num(e["auroc"], 3)])
S += [KeepTogether([table(rows2, [4.4, 2.3, 2.8, 2.4, 2.2, 1.7]),
                    P(f"Tabla 2. Generalización: test interno vs. RoCoLe (n = {R['n']['rocole_in']}, solo sana/roya). "
                      "Sana/enferma: puntaje 1 − p(sana).", cap)]),
      P(TXT.get("generalizacion", "")),
      KeepTogether([fig("03_gradcam_rocole.png", 7.0),
                    P("Figura 2. Grad-CAM del mejor modelo en la misma hoja de RoCoLe: raw, crop y mask.", cap)]),
      P(TXT.get("extra", f"Arañita roja (clase nunca vista): el {pct(ood[ood_cls], 0)} se predice como {ood_cls}; "
                         f"AUROC de detección por confianza máxima = {num(ood['AUROC detección OOD (MSP)'], 2)}.")), ]

S += [P("5. Conclusiones", h2), P(TXT.get("conclusiones", "")),
      P("<b>Referencias.</b> Krohling, Esgario & Ventura (2019) BRACOL, Mendeley Data 10.17632/yy2k5y8mxg.1 · Parraga-Alava et al. (2019) "
        "RoCoLe, Data in Brief 25 · Esgario et al. (2020) Comput. Electron. Agric. 169 · Tan & Le (2019) EfficientNet, ICML · "
        "Dosovitskiy et al. (2021) ViT, ICLR · Selvaraju et al. (2017) Grad-CAM, ICCV · Hendrycks & Gimpel (2017) ICLR. "
        "Código y notebooks: carpeta <i>src/</i> y <i>notebooks/</i> del proyecto.", cap)]

doc = SimpleDocTemplate(str(OUTF), pagesize=letter, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.3 * cm, bottomMargin=1.3 * cm,
                        title="Informe proyecto final — Enfermedades foliares del café", author="Equipo del proyecto final")
doc.build(S, onLaterPages=lambda c, d: (c.setFont("Times-Roman", 8), c.drawRightString(letter[0] - 1.6 * cm, 0.8 * cm, f"{d.page}")),
          onFirstPage=lambda c, d: (c.setFont("Times-Roman", 8), c.drawRightString(letter[0] - 1.6 * cm, 0.8 * cm, "1")))
from pypdf import PdfReader
print("ok", OUTF, "páginas:", len(PdfReader(str(OUTF)).pages))
