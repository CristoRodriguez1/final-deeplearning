"""Genera notebooks/00_colab_runner.ipynb: monta Drive, copia el código y ejecuta 02 y 03 en la GPU de Colab."""
from pathlib import Path
import nbformat as nbf

cells = [
    nbf.v4.new_markdown_cell("# 00 · Lanzador en Colab\nEjecuta los notebooks 02 y 03 en la GPU y guarda las versiones ejecutadas en `MyDrive/ProyectoCafe/notebooks_ejecutados/`."),
    nbf.v4.new_code_cell(r'''
from google.colab import drive
drive.mount("/content/drive")
import os, shutil, hashlib, subprocess
from pathlib import Path
DRIVE = Path("/content/drive/MyDrive/ProyectoCafe"); ROOT = Path("/content/proyecto")
for d in ("src", "tools"):
    shutil.copytree(DRIVE / d, ROOT / d, dirs_exist_ok=True)
(ROOT / "src" / "__init__.py").touch()
for p in sorted(list((ROOT / "src").glob("*.py")) + list((ROOT / "tools").glob("*.py"))):
    print(p.name, p.stat().st_size, hashlib.md5(p.read_bytes()).hexdigest()[:10])
!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip -q install timm
'''.strip()),
    nbf.v4.new_code_cell(r'''
os.environ["PROYECTO_ROOT"] = str(ROOT); os.environ["PROYECTO_OUT"] = str(DRIVE)
os.environ.pop("PROYECTO_SMOKE", None); os.environ["CACHE_EN_DRIVE"] = "0"   # 02 y 03 corren en la misma VM
(ROOT / "notebooks").mkdir(exist_ok=True); (DRIVE / "notebooks_ejecutados").mkdir(exist_ok=True)
!cd /content/proyecto && python tools/build_notebooks.py
'''.strip()),
    nbf.v4.new_code_cell(r'''
for nb in ["02_entrenamiento", "03_evaluacion_y_generalizacion"]:
    print("Ejecutando", nb, flush=True)
    r = subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", "--ExecutePreprocessor.timeout=-1",
                        "--output-dir", str(DRIVE / "notebooks_ejecutados"), str(ROOT / "notebooks" / f"{nb}.ipynb")],
                       capture_output=True, text=True, cwd=str(ROOT / "notebooks"))
    print(r.stdout[-2000:], r.stderr[-3000:])
    (DRIVE / "notebooks_ejecutados" / f"{nb}.status").write_text("OK" if r.returncode == 0 else "ERROR\n" + r.stderr[-5000:])
    if r.returncode != 0:
        break
print("FIN")
'''.strip()),
]
n = nbf.v4.new_notebook(); n["cells"] = cells
n["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                 "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}}
out = Path(__file__).resolve().parents[1] / "notebooks" / "00_colab_runner.ipynb"
nbf.write(n, out); print("ok", out)
