import json
from pathlib import Path
import shutil
nb_path = Path("notebooks/12_ablation_experiments.ipynb")
backup = nb_path.with_suffix(".ipynb.bak")
shutil.copy2(nb_path, backup)
with nb_path.open("r", encoding="utf-8") as f:
    nb = json.load(f)
old = (
    "return text+'\\nUse only visible image evidence. "
    "Do not infer exact roadway counts, percentages, "
    "scene identifiers, or assigned categories unless visually supported.'"
)
new = (
    "return ('Task: Assess the roadway flood burden and potential transportation disruption "
    "using only visible evidence from the Sentinel-2 optical and Sentinel-1 radar images.\\n"
    "Describe visible flood extent, roadway impacts, and uncertainty. "
    "Do not infer exact roadway counts, percentages, scene identifiers, "
    "or assigned categories unless visually supported.')"
)
patched = False
for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell["source"])
    if old in src:
        src = src.replace(old, new)
        cell["source"] = src.splitlines(keepends=True)
        patched = True
        break
if not patched:
    raise RuntimeError("Return statement not found.")
with nb_path.open("w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)
print("Notebook patched successfully.")
