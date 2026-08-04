import json
import shutil
from datetime import datetime
from pathlib import Path
nb_path = Path("notebooks/10_train_roadflood_vlm.ipynb")
backup = nb_path.with_name(
    f"{nb_path.stem}_backup_prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ipynb"
)
shutil.copy2(nb_path, backup)
with nb_path.open("r", encoding="utf-8") as f:
    nb = json.load(f)
patched = False
NEW_PROMPT = '''SYSTEM_PROMPT = (
    "You are RoadFlood-VLM, a multimodal satellite flood-classification assistant. "
    "Use Sentinel-2 optical imagery together with Sentinel-1 radar imagery to "
    "classify visible flood burden. Return ONLY valid JSON."
)
'''
for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell["source"])
    if "SYSTEM_PROMPT" in src:
        lines = src.splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("SYSTEM_PROMPT"):
                start = i
                break
        else:
            continue
        end = start + 1
        while end < len(lines):
            if lines[end].strip().endswith(")"):
                end += 1
                break
            end += 1
        new_lines = (
            lines[:start]
            + NEW_PROMPT.splitlines()
            + lines[end:]
        )
        cell["source"] = [x + "\n" for x in new_lines]
        patched = True
        break
if not patched:
    raise RuntimeError("SYSTEM_PROMPT not found.")
with nb_path.open("w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)
print("Notebook patched successfully.")
print("Backup:", backup)
