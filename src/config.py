"""Rutas y constantes compartidas del proyecto.

Las rutas se pueden redefinir con variables de entorno (útil en Colab):
  PROYECTO_ROOT  -> raíz con data/ (por defecto, la carpeta padre de src/)
  PROYECTO_OUT   -> dónde se guardan runs/ y figures/ (en Colab: Google Drive)
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("PROYECTO_ROOT", Path(__file__).resolve().parents[1]))
OUT = Path(os.environ.get("PROYECTO_OUT", ROOT))

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "cache"
RUNS = OUT / "runs"
FIG = OUT / "figures"

CLASSES = ["sana", "minador", "roya", "phoma", "cercospora"]
NUM_CLASSES = len(CLASSES)
SEED = 42

CACHE_SIZE = 256      # lado del lienzo cuadrado cacheado (letterbox)
IMG_SIZE = 224        # resolución de entrada a las redes
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PAD_COLOR = tuple(round(255 * m) for m in IMAGENET_MEAN)  # tras normalizar ≈ 0: el relleno no aporta señal

ROCOLE_VERSIONS = ("raw", "crop", "mask")
