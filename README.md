# Clasificación de enfermedades foliares del café y generalización ante cambio de dominio

Proyecto final — Redes Neuronales y Aprendizaje Profundo (EAFIT, 2026-2)

Entrenamos clasificadores de hojas de café (sana, minador, roya, phoma, cercospora) sobre **BRACOL** (arábica, Brasil, hoja
desprendida sobre fondo blanco) y evaluamos su generalización en **RoCoLe** (robusta, Ecuador, hoja en la planta en campo).

## Resultados principales
| Modelo | Parámetros | F1 val | Accuracy test | F1-macro test | Acc. RoCoLe raw → mask | AUROC sana/enferma raw → mask |
|---|---|---|---|---|---|---|
| CNN base (desde cero) | 1,17 M | 0,877 | 84,2 % | 0,808 | 21,6 % → 77,1 % | 0,57 → 0,92 |
| EfficientNet-B0 | 4,01 M | 0,915 | 88,1 % | 0,864 | 55,1 % → 69,6 % | 0,59 → 0,90 |
| **ViT-Tiny/16** (seleccionado) | 5,53 M | **0,935** | 87,4 % | 0,843 | 31,3 % → 69,0 % | 0,75 → 0,93 |
| EfficientNet-B0 sin aumento | 4,01 M | 0,913 | 88,1 % | 0,851 | 43,9 % → 38,3 % | 0,43 → 0,59 |

El rendimiento cae fuertemente en fotos de campo y se recupera en buena parte al eliminar el fondo; el aumento de datos casi no
cambia el test interno, pero es decisivo para generalizar. Entrenamiento en CPU (Intel i5-1155G7, 20 GB RAM; Colab no asignó GPU).

## Entregables
- `entregables/Presentacion_ProyectoFinal.pptx`: presentación (18 diapositivas).
- `entregables/Informe_ProyectoFinal.pdf`: informe de 3 páginas.
- `notebooks/`: los tres notebooks ejecutados, con salidas.

## Estructura
```
notebooks/
  01_datos_y_eda.ipynb                 descarga, etiquetas, partición 70/15/15, EDA y cambio de dominio
  02_entrenamiento.ipynb               arquitecturas, hiperparámetros, entrenamiento y curvas (Colab GPU)
  03_evaluacion_y_generalizacion.ipynb test BRACOL, errores, RoCoLe raw/crop/mask, OOD, Grad-CAM (Colab GPU)
src/
  config.py    rutas y constantes          data.py      descarga, letterbox, caché y aumentos
  models.py    CNN base, EfficientNet-B0, ViT-Tiny     train.py  bucle de entrenamiento
  evaluate.py  métricas, latencia, Grad-CAM
data/processed/  tablas de etiquetas y partición (bracol_leaf.csv, rocole.csv, meta.json)
runs/            historiales, resúmenes y métricas de cada corrida
figures/         todas las figuras usadas en la presentación y el informe
entregables/     presentación (.pptx) e informe (.pdf)
```

## Cómo reproducir
**Local (como se hizo en este proyecto, CPU):**
```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute notebooks/01_datos_y_eda.ipynb      # descarga datos, partición y EDA
python -m src.train --run vit_tiny          # también: cnn_base, effnet_b0, effnet_b0_sinaug
jupyter nbconvert --to notebook --execute notebooks/02_entrenamiento.ipynb   # omite corridas ya entrenadas; curvas
jupyter nbconvert --to notebook --execute notebooks/03_evaluacion_y_generalizacion.ipynb
python tools/collect_results.py && node tools/build_deck.js && python tools/build_report.py   # entregables
```

**Colab (GPU, si hay cuota):** copiar `src/` y `tools/` a `MyDrive/ProyectoCafe/`, abrir `notebooks/00_colab_runner.ipynb`
con entorno T4 y ejecutar todo: descarga los datos desde sus fuentes oficiales, entrena y evalúa, y deja los notebooks ejecutados
en `ProyectoCafe/notebooks_ejecutados/`.

## Datos
- BRACOL — Krohling, Esgario & Ventura (2019), Mendeley Data, doi:10.17632/yy2k5y8mxg.1 (CC BY 4.0).
  El `.zip` de Mendeley está truncado; se usa la copia completa enlazada en https://github.com/esgario/lara2018.
- RoCoLe — Parraga-Alava, Cusme, Loor & Santander (2019), Mendeley Data, doi:10.17632/c5yvn32dzg.2 (CC BY 4.0).
