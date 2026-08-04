import json
from pathlib import Path
nb_path = Path("notebooks/10_train_roadflood_vlm.ipynb")
nb = json.loads(nb_path.read_text())
patched=False
for cell in nb["cells"]:
    if cell.get("cell_type")!="code":
        continue
    src="".join(cell["source"])
    if "from peft import LoraConfig" in src:
        src=src.replace(
            "from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training",
            "from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel"
        )
        cell["source"]=src.splitlines(True)
    if "model = get_peft_model(model, lora_config)" in src:
        replacement=r'''
EXISTING_ADAPTER_DIR = Path(
    os.environ.get(
        "ROADFLOOD_EXISTING_ADAPTER",
        str(
            PROJECT_ROOT
            / "outputs"
            / "roadflood_vlm_training"
            / "full_20260729T032411Z"
            / "final_adapter"
        ),
    )
)
if EXISTING_ADAPTER_DIR.exists():
    print(f"Loading existing adapter: {EXISTING_ADAPTER_DIR}")
    model=PeftModel.from_pretrained(
        model,
        EXISTING_ADAPTER_DIR,
        is_trainable=True,
    )
else:
    print("Existing adapter not found. Creating new LoRA.")
    model=get_peft_model(
        model,
        lora_config,
    )
'''
        src=src.replace(
            "model = get_peft_model(model, lora_config)",
            replacement,
        )
        cell["source"]=src.splitlines(True)
        patched=True
if not patched:
    raise RuntimeError("Target line not found.")
nb_path.write_text(json.dumps(nb,indent=2))
print("Notebook patched.")
