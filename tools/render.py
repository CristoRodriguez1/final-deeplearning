"""Renderiza un .pptx o .pdf a PNG (LibreOffice + PyMuPDF) y arma cuadrículas 2×2 para revisión visual.
Uso: python tools/render.py archivo.pptx carpeta_salida [páginas, p. ej. 1,4,7]"""
import subprocess
import sys
from pathlib import Path

import pymupdf
from PIL import Image

src, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
pages = [int(p) for p in sys.argv[3].split(",")] if len(sys.argv) > 3 else None
out.mkdir(parents=True, exist_ok=True)
pdf = src
if src.suffix.lower() != ".pdf":
    subprocess.run([r"C:\Program Files\LibreOffice\program\soffice.exe", "--headless", "--convert-to", "pdf",
                    "--outdir", str(out), str(src)], check=True, capture_output=True)
    pdf = out / (src.stem + ".pdf")
doc = pymupdf.open(pdf)
print("páginas:", doc.page_count)
for f in out.glob("p-*.png"):
    f.unlink()
files = []
for i, p in enumerate(doc, start=1):
    if pages and i not in pages:
        continue
    f = out / f"p-{i:02d}.png"
    p.get_pixmap(dpi=110).save(f)
    files.append(f)
for f in out.glob("grid-*.png"):
    f.unlink()
for g in range(0, len(files), 4):
    ims = [Image.open(f) for f in files[g:g + 4]]
    w, h = ims[0].size
    c = Image.new("RGB", (2 * w + 10, 2 * h + 10), "black")
    for k, im in enumerate(ims):
        c.paste(im, ((k % 2) * (w + 10), (k // 2) * (h + 10)))
    c.save(out / f"grid-{g // 4 + 1}.png")
print("ok", out)
