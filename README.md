# Clasificación de enfermedades en hojas de café

Proyecto final de Redes Neuronales y Aprendizaje Profundo, EAFIT 2026-2.

Integrantes: Cristobal Rodriguez Rojas, Juan Esteban Villada y Cristian David Cabezas Jimenez.

Entrenamos clasificadores de hojas de café con 5 clases (sana, minador, roya, phoma y cercospora) usando el dataset BRACOL,
que tiene fotos de hojas de Brasil sobre fondo blanco. Luego probamos qué tan bien generalizan en RoCoLe, un dataset de Ecuador
con fotos tomadas en campo.

## Resultados

| Modelo | Parámetros | F1 val | Accuracy test | F1-macro test | Acc. RoCoLe (raw / mask) | AUROC sana vs. enferma (raw / mask) |
|---|---|---|---|---|---|---|
| CNN base (desde cero) | 1,17 M | 0,877 | 84,2 % | 0,808 | 21,6 % / 77,1 % | 0,57 / 0,92 |
| EfficientNet-B0 | 4,01 M | 0,915 | 88,1 % | 0,864 | 55,1 % / 69,6 % | 0,59 / 0,90 |
| ViT-Tiny/16 (seleccionado) | 5,53 M | 0,935 | 87,4 % | 0,843 | 31,3 % / 69,0 % | 0,75 / 0,93 |
| EfficientNet-B0 sin aumento | 4,01 M | 0,913 | 88,1 % | 0,851 | 43,9 % / 38,3 % | 0,43 / 0,59 |

En las fotos de campo el rendimiento cae mucho, y mejora bastante cuando se quita el fondo de la imagen (versión mask).
Todos los modelos se entrenaron en CPU (Intel i5-1155G7, 20 GB de RAM).

## Contenido

```
notebooks/
  01_datos_y_eda.ipynb                  descarga de datos, etiquetas, partición 70/15/15 y análisis exploratorio
  02_entrenamiento.ipynb                arquitecturas, hiperparámetros, entrenamiento y curvas
  03_evaluacion_y_generalizacion.ipynb  test en BRACOL, errores, RoCoLe (raw/crop/mask), arañita roja y Grad-CAM
src/
  config.py     rutas y constantes
  data.py       descarga, letterbox, caché y aumento de datos
  models.py     CNN base, EfficientNet-B0 y ViT-Tiny
  train.py      entrenamiento
  evaluate.py   métricas, latencia y Grad-CAM
data/processed/ etiquetas y partición (bracol_leaf.csv, rocole.csv, meta.json)
runs/           historial, resumen y métricas de cada modelo
figures/        figuras de los notebooks
```

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute notebooks/01_datos_y_eda.ipynb
python -m src.train --run vit_tiny
jupyter nbconvert --to notebook --execute notebooks/02_entrenamiento.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_evaluacion_y_generalizacion.ipynb
```

El notebook 01 descarga los datasets. `src.train` también acepta `cnn_base`, `effnet_b0` y `effnet_b0_sinaug`, y el
notebook 02 entrena las corridas que falten.

## Datos

- BRACOL: Krohling, Esgario y Ventura (2019), Mendeley Data, doi:10.17632/yy2k5y8mxg.1 (CC BY 4.0). El zip de Mendeley
  está incompleto, así que usamos la copia enlazada en https://github.com/esgario/lara2018.
- RoCoLe: Parraga-Alava, Cusme, Loor y Santander (2019), Mendeley Data, doi:10.17632/c5yvn32dzg.2 (CC BY 4.0).
