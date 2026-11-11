"""Descarga de datos, preprocesamiento (letterbox), caché en tensores y transformaciones."""
from __future__ import annotations

import csv
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision.transforms import v2

from . import config as C

BRACOL_GDRIVE_ID = "15YHebAGrx1Vhv8-naave-R5o3Uo70jsm"   # enlace oficial (README de esgario/lara2018)
ROCOLE_CSV_URL = ("https://data.mendeley.com/public-files/datasets/c5yvn32dzg/files/"
                  "e39cecdd-8dca-445a-b734-10229427f2fa/file_downloaded")
EXIF_T = {3: Image.Transpose.ROTATE_180, 6: Image.Transpose.ROTATE_270, 8: Image.Transpose.ROTATE_90}


# ----------------------------------------------------------------------------- descarga
def _download(url: str, dst: Path, chunk: int = 1 << 20) -> Path:
    dst = Path(dst)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for b in r.iter_content(chunk):
                f.write(b)
    tmp.replace(dst)
    return dst


def download_bracol() -> Path:
    z = _download(f"https://drive.usercontent.google.com/download?id={BRACOL_GDRIVE_ID}&export=download&confirm=t",
                  C.RAW / "bracol" / "bracol_gdrive.zip")
    root = C.RAW / "bracol" / "coffee-datasets"
    if not (root / "leaf" / "dataset.csv").exists():
        with zipfile.ZipFile(z) as zf:
            zf.extractall(C.RAW / "bracol")
    return root


def rocole_rows() -> list[dict]:
    path = _download(ROCOLE_CSV_URL, C.RAW / "rocole" / "RoCoLE-csv.csv")
    csv.field_size_limit(10**9)
    return list(csv.DictReader(open(path, encoding="utf8")))


def download_rocole(workers: int = 16) -> None:
    rows = rocole_rows()
    def fetch(row):
        _download(row["Labeled Data"], C.RAW / "rocole" / "images" / row["External ID"].strip())
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(fetch, rows))


def build_rocole_versions(workers: int = 8, maxside: int = 1024) -> None:
    """raw / crop (bbox de la hoja +5 %) / mask (fondo blanco). Corrige la orientación EXIF."""
    polys = {r["External ID"].strip(): json.loads(r["Label"])["Leaf"][0]["geometry"] for r in rocole_rows()}
    for v in C.ROCOLE_VERSIONS:
        (C.PROC / "rocole" / v).mkdir(parents=True, exist_ok=True)

    def make(name):
        outs = {v: C.PROC / "rocole" / v / name for v in C.ROCOLE_VERSIONS}
        if all(p.exists() for p in outs.values()):
            return
        im = Image.open(C.RAW / "rocole" / "images" / name)
        orient = im.getexif().get(274)
        im = im.convert("RGB")
        poly = [(p["x"], p["y"]) for p in polys[name]]
        mask = Image.new("L", im.size, 0)
        ImageDraw.Draw(mask).polygon(poly, fill=255)
        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
        mx, my = 0.05 * (max(xs) - min(xs)), 0.05 * (max(ys) - min(ys))
        box = (max(0, int(min(xs) - mx)), max(0, int(min(ys) - my)),
               min(im.width, int(max(xs) + mx)), min(im.height, int(max(ys) + my)))
        white = Image.new("RGB", im.size, (255, 255, 255))
        versions = {"raw": im, "crop": im.crop(box), "mask": Image.composite(im, white, mask).crop(box)}
        for v, img in versions.items():
            if orient in EXIF_T:
                img = img.transpose(EXIF_T[orient])
            img.thumbnail((maxside, maxside))
            img.save(outs[v], quality=92)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(make, polys.keys()))


def build_tables(force: bool = False) -> None:
    """Reproduce las tablas del notebook 01: bracol_leaf.csv (con partición 70/15/15), rocole.csv y meta.json."""
    import re
    from sklearn.model_selection import train_test_split
    if (C.PROC / "bracol_leaf.csv").exists() and (C.PROC / "rocole.csv").exists() and not force:
        return
    C.PROC.mkdir(parents=True, exist_ok=True)
    root = C.RAW / "bracol" / "coffee-datasets"
    bra = pd.read_csv(root / "leaf" / "dataset.csv")
    bra["path"] = bra["id"].map(lambda i: f"data/raw/bracol/coffee-datasets/leaf/images/{i}.jpg")
    bra = bra[bra.predominant_stress < 5].reset_index(drop=True)
    bra["label"] = bra["predominant_stress"].astype(int)
    bra["clase"] = bra["label"].map(dict(enumerate(C.CLASSES)))
    _, tmp = train_test_split(bra, test_size=0.30, stratify=bra["label"], random_state=C.SEED)
    val_df, test_df = train_test_split(tmp, test_size=0.50, stratify=tmp["label"], random_state=C.SEED)
    bra["split"] = "train"
    bra.loc[val_df.index, "split"] = "val"
    bra.loc[test_df.index, "split"] = "test"
    freq = bra[bra.split == "train"]["label"].value_counts().sort_index()
    w = (freq.sum() / (len(freq) * freq)).round(3)
    cols = ["id", "path", "label", "clase", "miner", "rust", "phoma", "cercospora", "severity", "split"]
    bra[cols].to_csv(C.PROC / "bracol_leaf.csv", index=False)
    json.dump({"classes": C.CLASSES, "classes_en": ["healthy", "miner", "rust", "phoma", "cercospora"],
               "class_weights": w.tolist(), "seed": C.SEED}, open(C.PROC / "meta.json", "w"), indent=2, ensure_ascii=False)

    rmap = {"healthy": "sana", "rust_level_1": "roya", "rust_level_2": "roya", "rust_level_3": "roya",
            "rust_level_4": "roya", "red_spider_mite": "arañita_roja"}
    name_re = re.compile(r"C(\d+)P(\d+)([HE])(\d+)", re.I)
    recs = []
    for row in rocole_rows():
        lab = json.loads(row["Label"])["classification"]
        name = row["External ID"].strip()
        m = name_re.match(name)
        recs.append({"file": name, "path": f"data/raw/rocole/images/{name}", "rocole_class": lab, "clase": rmap[lab],
                     "rust_level": int(lab[-1]) if lab.startswith("rust") else 0,
                     "lado": {"H": "haz", "E": "envés"}.get(m.group(3), "?") if m else "?",
                     "planta": f"C{m.group(1)}P{m.group(2)}" if m else "?"})
    roc = pd.DataFrame(recs)
    roc["label"] = roc["clase"].map({c: i for i, c in enumerate(C.CLASSES)}).fillna(-1).astype(int)
    roc[["file", "path", "rocole_class", "clase", "label", "rust_level", "lado", "planta"]].to_csv(C.PROC / "rocole.csv", index=False)


def split_fingerprint() -> str:
    """Huella de la partición (para verificar que es idéntica entre máquinas)."""
    import hashlib
    bra = pd.read_csv(C.PROC / "bracol_leaf.csv")
    s = ",".join(f"{i}:{sp}" for i, sp in zip(bra["id"], bra["split"]))
    return hashlib.md5(s.encode()).hexdigest()[:12]


# ----------------------------------------------------------------------------- preprocesamiento
def letterbox(img: Image.Image, size: int = C.CACHE_SIZE, color=C.PAD_COLOR) -> Image.Image:
    """Redimensiona conservando la relación de aspecto y rellena hasta size×size."""
    img = img.convert("RGB")
    s = size / max(img.size)
    w, h = max(1, round(img.width * s)), max(1, round(img.height * s))
    img = img.resize((w, h), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), color)
    canvas.paste(img, ((size - w) // 2, (size - h) // 2))
    return canvas


def build_cache(paths: list[Path], out: Path, workers: int = 8) -> torch.Tensor:
    """Tensor uint8 N×3×S×S con las imágenes en letterbox (se guarda en disco para reusar)."""
    out = Path(out)
    if out.exists():
        return torch.load(out)
    def load(p):
        return np.asarray(letterbox(Image.open(p)), dtype=np.uint8)
    with ThreadPoolExecutor(workers) as ex:
        arr = np.stack(list(ex.map(load, paths)))
    x = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(x, out)
    return x


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    bra = pd.read_csv(C.PROC / "bracol_leaf.csv")
    roc = pd.read_csv(C.PROC / "rocole.csv")
    return bra, roc


def get_caches() -> dict[str, torch.Tensor]:
    """Devuelve {'bracol': X, 'rocole_raw': X, 'rocole_crop': X, 'rocole_mask': X} (uint8)."""
    bra, roc = load_tables()
    caches = {"bracol": build_cache([C.ROOT / p for p in bra["path"]], C.CACHE / "bracol.pt")}
    for v in C.ROCOLE_VERSIONS:
        caches[f"rocole_{v}"] = build_cache([C.PROC / "rocole" / v / f for f in roc["file"]],
                                            C.CACHE / f"rocole_{v}.pt")
    return caches


# ----------------------------------------------------------------------------- transformaciones
def _rot90():
    return v2.RandomChoice([v2.RandomRotation((a, a)) for a in (0, 90, 180, 270)])


def get_transforms(kind: str):
    """'train' = aumento completo; 'train_min' = solo volteos (ablación); 'eval' = sin aumento."""
    norm = [v2.ToDtype(torch.float32, scale=True), v2.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD)]
    if kind == "train":
        aug = [v2.RandomResizedCrop(C.IMG_SIZE, scale=(0.7, 1.0), ratio=(0.8, 1.25), antialias=True),
               v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(), _rot90(),
               v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03)]
    elif kind == "train_min":
        aug = [v2.Resize(C.IMG_SIZE, antialias=True), v2.RandomHorizontalFlip(), v2.RandomVerticalFlip()]
    elif kind == "eval":
        aug = [v2.Resize(C.IMG_SIZE, antialias=True)]
    else:
        raise ValueError(kind)
    return v2.Compose(aug + norm)


class TensorImageDataset(Dataset):
    def __init__(self, x: torch.Tensor, y, transform):
        self.x, self.y, self.t = x, torch.as_tensor(np.asarray(y), dtype=torch.long), transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.t(self.x[i]), self.y[i]


def denormalize(x: torch.Tensor) -> np.ndarray:
    """Tensor normalizado C×H×W → imagen H×W×3 en [0, 1] para graficar."""
    m = torch.tensor(C.IMAGENET_MEAN).view(3, 1, 1)
    s = torch.tensor(C.IMAGENET_STD).view(3, 1, 1)
    return (x.cpu() * s + m).clamp(0, 1).permute(1, 2, 0).numpy()
