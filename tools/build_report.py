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
    return Paragraph(t.replace(" %", "00a0%"), st)


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
S += [P("Clasificación de enfermedades en hojas de café y su generalización a fotos de campo", h1),
      P("Cristobal Rodriguez Rojas, Juan Esteban Villada y Cristian David Cabezas Jimenez<br/>"
        "Redes Neuronales y Aprendizaje Profundo, Universidad EAFIT, noviembre de 2026", sub)]

S += [P("1. Problema y objetivos", h2),
      P("La roya del café (<i>Hemileia vastatrix</i>) es la enfermedad foliar que más afecta al cultivo y, junto con otras como la "
        "cercospora y el minador, reduce la producción. Hoy su diagnóstico depende de que un experto revise las hojas. Las redes "
        "convolucionales pueden aprender a reconocer estos síntomas por su textura y color, pero casi todos los datasets públicos se "
        "toman en laboratorio, con la hoja sobre un fondo blanco, y no es claro que un modelo entrenado así sirva en campo. "
        "<b>Objetivo general:</b> desarrollar un modelo de Deep Learning que clasifique enfermedades y plagas en hojas de café y medir "
        "qué tan bien generaliza a fotos de otra región, otra variedad y otras condiciones de captura. <b>Objetivos específicos:</b> "
        "primero, preparar los datos con una partición estratificada y sin fuga de información; segundo, entrenar y comparar una CNN "
        "desde cero, EfficientNet-B0 y ViT-Tiny usando accuracy, F1-macro y costo computacional; y tercero, evaluar el mejor modelo en "
        "un dataset externo, medir cuánto cae el rendimiento y buscar la causa con Grad-CAM.")]

sp = D["split"]
S += [P("2. Datos y preprocesamiento", h2),
      P(f"Para entrenar usamos <b>BRACOL</b> (Krohling et al., 2019), con {D['bracol_total']} fotos de hojas de café arábica de Brasil, "
        "desprendidas de la planta y fotografiadas sobre fondo blanco. Cada hoja está etiquetada con su estrés predominante. "
        f"Descartamos las {D['bracol_indeterminadas']} marcadas como indeterminadas y quedaron {D['bracol_usadas']} hojas en 5 clases: "
        + ", ".join(f"{k} {v}" for k, v in D["por_clase"].items()) +
        f". Las clases están desbalanceadas (roya tiene 3,6 veces más imágenes que cercospora) y el "
        f"{str(D['multiestres_pct']).replace('.', ',')} % de las hojas tiene más de una enfermedad. Las dividimos en entrenamiento, "
        f"validación y prueba (70/15/15, estratificado y con semilla fija), con {sp['train']}, {sp['val']} y {sp['test']} imágenes. "
        f"Para la evaluación externa usamos <b>RoCoLe</b> (Parraga-Alava et al., 2019), que no se usó en ningún momento del entrenamiento: "
        f"{D['rocole_total']} fotos de café robusta de Ecuador tomadas en campo, con la hoja todavía en la planta. Hay {D['rocole_sana']} "
        f"sanas, {D['rocole_roya']} con roya en niveles de 1 a 4 y {D['rocole_ood']} con arañita roja, una plaga que no aparece en BRACOL. "
        "En el análisis exploratorio ya se notaba la diferencia entre los dos datasets: el brillo medio pasa de 0,62 a 0,41 y el fondo "
        "blanco pasa de ocupar el 30 % de la imagen a solo el 1 %. En el preprocesamiento llevamos cada imagen a 256×256 sin deformarla "
        "(<i>letterbox</i>, rellenando con el color medio de ImageNet), recortamos a 224 píxeles y normalizamos con la media y la "
        "desviación de ImageNet. Solo en entrenamiento aplicamos aumento de datos: recortes aleatorios, volteos, rotaciones de 90° y "
        "cambios de brillo, contraste y saturación. Además, con las máscaras de segmentación de RoCoLe armamos tres versiones de cada "
        "foto externa: la original (<i>raw</i>), un recorte a la hoja (<i>crop</i>) y la hoja sobre fondo blanco (<i>mask</i>). Así "
        "podemos ver cuánto influye el fondo.")]

pm = lambda r: f"{RU[r]['params_M']:.2f}".replace(".", ",")
S += [P("3. Modelos y entrenamiento", h2),
      P("Comparamos tres modelos. El primero es una <b>CNN</b> que diseñamos y entrenamos desde cero, con 4 bloques convolucionales "
        "(cada uno con dos convoluciones 3×3 con batch normalization y ReLU, y un max pooling), global average pooling y una capa final; "
        f"tiene {pm('cnn_base')} M de parámetros. El segundo es <b>EfficientNet-B0</b> (Tan y Le, 2019), formada por 16 bloques MBConv que "
        f"usan convoluciones <i>depthwise</i> y <i>squeeze-and-excitation</i>, con {pm('effnet_b0')} M de parámetros. El tercero es "
        "<b>ViT-Tiny/16</b> (Dosovitskiy et al., 2021), un transformer que divide la imagen en 196 parches y los procesa con 12 bloques "
        f"de <i>self-attention</i>, con {pm('vit_tiny')} M de parámetros. Los dos últimos parten de pesos preentrenados en ImageNet; lo único "
        "que les cambiamos fue la capa de salida, que ahora tiene 5 clases y dropout, y ajustamos toda la red. Entrenamos con AdamW, con "
        "learning rate de 1e-3 para la CNN, 3e-4 para EfficientNet y 1e-4 para el ViT, 2 épocas de <i>warmup</i> y luego decaimiento "
        "coseno, y batch de 32. Como pérdida usamos entropía cruzada con pesos inversos a la frecuencia de cada clase y <i>label "
        "smoothing</i> de 0,1. Nos quedamos con la época de mejor F1-macro en validación, con <i>early stopping</i> y un máximo de 40 "
        "épocas para la CNN y 25 para los demás. También entrenamos una EfficientNet-B0 solo con volteos, para medir el efecto del "
        "aumento de datos. Todo se entrenó en un portátil con CPU Intel Core i5-1155G7, sin GPU, 20 GB de RAM y PyTorch 2.15 con timm.")]

rows = [["Modelo", "Accuracy", "F1-macro", "Mejor ép.", "Parám.", "Entrenam.", "ms/img CPU"]]
lat = {l["modelo"]: l for l in R["latencia"]}
for r in ["cnn_base", "effnet_b0", "vit_tiny", "effnet_b0_sinaug"]:
    rows.append([NOM[r], pct(T[r]["accuracy"]), pct(T[r]["f1_macro"]), f"{RU[r]['best_epoch']}/{RU[r]['epochs_run']}",
                 f"{pm(r)} M", f"{RU[r]['train_min']:.0f} min", f"{lat[NOMDF[r]]['ms_img_cpu']:.0f}"])
S += [P("4. Resultados", h2),
      KeepTogether([table(rows, [4.4, 1.7, 1.7, 1.7, 1.9, 2.3, 2.1]),
                    P(f"Tabla 1. Resultados en la partición de prueba de BRACOL ({R['n']['test']} imágenes). Esgario et al. (2020) reportan "
                      "95,24 % de accuracy en este dataset con otra partición.", cap)]),
      P(TXT.get("resultados_test", f"{NOM[BEST]} obtiene {pct(T[BEST]['accuracy'])} de accuracy y {pct(T[BEST]['f1_macro'])} de F1-macro en test.")),
      KeepTogether([fig("02_curvas_comparacion.png", 15.5),
                    P("Figura 1. Curvas de pérdida y F1-macro. La línea continua es validación y la punteada, entrenamiento.", cap)]),
      P(TXT.get("curvas", "")), ]

rows2 = [["Dominio (" + NOMDF[BEST] + ")", "Acc. 5 clases", "Acc. sana/enferma", "Recall enferma", "Recall sana", "AUROC"]]
for e in (bt, raw, crop, mask):
    rows2.append([e["dominio"], pct(e["acc_5clases"]), pct(e["accuracy"]), pct(e["recall_enferma"]), pct(e["recall_sana"]), num(e["auroc"], 3)])
S += [KeepTogether([table(rows2, [4.4, 2.3, 2.8, 2.4, 2.2, 1.7]),
                    P(f"Tabla 2. Resultados del ViT-Tiny en la prueba interna y en RoCoLe ({R['n']['rocole_in']} fotos sanas o con roya). "
                      "Para sana contra enferma usamos 1 − p(sana) como puntaje.", cap)]),
      P(TXT.get("generalizacion", "")),
      KeepTogether([fig("03_gradcam_rocole.png", 6.0),
                    P("Figura 2. Grad-CAM del ViT-Tiny sobre tres hojas de RoCoLe (una por fila), cada una en sus versiones raw, crop y mask.", cap)]),
      P(TXT.get("extra", f"Arañita roja (clase nunca vista): el {pct(ood[ood_cls], 0)} se predice como {ood_cls}; "
                         f"AUROC de detección por confianza máxima = {num(ood['AUROC detección OOD (MSP)'], 2)}.")), ]

S += [P("5. Conclusiones", h2), P(TXT.get("conclusiones", "")),
      P("<b>Referencias.</b> Krohling, Esgario y Ventura (2019), BRACOL, Mendeley Data, doi:10.17632/yy2k5y8mxg.1. "
        "Parraga-Alava et al. (2019), RoCoLe, Data in Brief 25. Esgario et al. (2020), Computers and Electronics in Agriculture 169. "
        "Tan y Le (2019), EfficientNet, ICML. Dosovitskiy et al. (2021), ViT, ICLR. Selvaraju et al. (2017), Grad-CAM, ICCV. "
        "Hendrycks y Gimpel (2017), ICLR.", cap)]

doc = SimpleDocTemplate(str(OUTF), pagesize=letter, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.3 * cm, bottomMargin=1.3 * cm,
                        title="Proyecto final - Enfermedades en hojas de café",
                        author="Cristobal Rodriguez Rojas, Juan Esteban Villada, Cristian David Cabezas Jimenez")
doc.build(S, onLaterPages=lambda c, d: (c.setFont("Times-Roman", 8), c.drawRightString(letter[0] - 1.6 * cm, 0.8 * cm, f"{d.page}")),
          onFirstPage=lambda c, d: (c.setFont("Times-Roman", 8), c.drawRightString(letter[0] - 1.6 * cm, 0.8 * cm, "1")))
from pypdf import PdfReader, PdfWriter
w = PdfWriter(clone_from=str(OUTF))
w.add_metadata({"/Title": "Proyecto final - Enfermedades en hojas de café",
                "/Author": "Cristobal Rodriguez Rojas, Juan Esteban Villada, Cristian David Cabezas Jimenez",
                "/Producer": "", "/Creator": "", "/Subject": "", "/Keywords": ""})
w.write(str(OUTF))
print("ok", OUTF, "páginas:", len(PdfReader(str(OUTF)).pages))
