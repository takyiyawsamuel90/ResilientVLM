from __future__ import annotations
import json
import shutil
from datetime import datetime
from pathlib import Path
NOTEBOOK_PATH = Path(
    "notebooks/10_train_roadflood_vlm.ipynb"
)
if not NOTEBOOK_PATH.exists():
    raise FileNotFoundError(NOTEBOOK_PATH)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = NOTEBOOK_PATH.with_name(
    f"10_train_roadflood_vlm_before_sturm_{timestamp}.ipynb"
)
shutil.copy2(
    NOTEBOOK_PATH,
    backup_path,
)
notebook = json.loads(
    NOTEBOOK_PATH.read_text(encoding="utf-8")
)
replacements = 0
for cell_index, cell in enumerate(notebook.get("cells", [])):
    if cell.get("cell_type") != "code":
        continue
    source = "".join(cell.get("source", []))
    original = source
    # Replace the split directory assignment without relying
    # on the exact original path.
    lines = source.splitlines(keepends=True)
    updated_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("SPLIT_DIR ="):
            indentation = line[: len(line) - len(line.lstrip())]
            line = (
                indentation
                + 'SPLIT_DIR = PROJECT_ROOT / "data" / "processed" '
                + '/ "sturm_vlm_quick" / "training_splits"\n'
            )
            replacements += 1
        elif (
            stripped.startswith("OUTPUT_ROOT =")
            and "roadflood_vlm_training" in stripped
        ):
            indentation = line[: len(line) - len(line.lstrip())]
            line = (
                indentation
                + 'OUTPUT_ROOT = PROJECT_ROOT / "outputs" '
                + '/ "sturm_auxiliary_training"\n'
            )
            replacements += 1
        elif (
            stripped.startswith("TRAINING_ROOT =")
            and "roadflood_vlm_training" in stripped
        ):
            # Leave source adapter discovery pointing to the
            # existing RoadFlood training directory.
            pass
        updated_lines.append(line)
    source = "".join(updated_lines)
    if source != original:
        cell["source"] = source.splitlines(keepends=True)
        print(
            f"Patched path configuration in cell {cell_index}"
        )
if replacements < 1:
    raise RuntimeError(
        "SPLIT_DIR assignment was not found. "
        "Notebook was not changed."
    )
NOTEBOOK_PATH.write_text(
    json.dumps(
        notebook,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)
print(f"Patched notebook: {NOTEBOOK_PATH}")
print(f"Backup notebook : {backup_path}")
print(f"Path replacements: {replacements}")
