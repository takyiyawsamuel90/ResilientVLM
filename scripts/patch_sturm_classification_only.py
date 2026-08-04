from __future__ import annotations
import json
import shutil
from datetime import datetime
from pathlib import Path
import pandas as pd
SPLIT_DIR = Path(
    "data/processed/sturm_vlm_quick/training_splits"
)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_dir = SPLIT_DIR.parent / f"training_splits_before_class_only_{timestamp}"
shutil.copytree(
    SPLIT_DIR,
    backup_dir,
)
SYSTEM_PROMPT = (
    "You are RoadFlood-VLM, a multimodal satellite flood-classification "
    "assistant. Use the Sentinel-2 optical image and Sentinel-1 radar image "
    "to classify visible flood burden. Return only valid JSON."
)
INSTRUCTION_TEXT = (
    "Analyze the Sentinel-2 optical image and Sentinel-1 radar image. "
    "Classify the visible scene flood burden as Low, Moderate, or High. "
    "Return only JSON using exactly this schema: "
    '{"scene_flood_burden":"Low|Moderate|High"}'
)
for split in ["train", "validation", "test"]:
    csv_path = SPLIT_DIR / f"{split}.csv"
    df = pd.read_csv(
        csv_path,
        keep_default_na=False,
    )
    df = df[
        df["scene_flood_burden"].isin(
            ["Low", "Moderate", "High"]
        )
    ].copy()
    df["system_prompt"] = SYSTEM_PROMPT
    df["instruction_text"] = INSTRUCTION_TEXT
    df["response_text"] = df[
        "scene_flood_burden"
    ].map(
        lambda label: json.dumps(
            {
                "scene_flood_burden": label,
            },
            separators=(",", ":"),
        )
    )
    df.to_csv(
        csv_path,
        index=False,
    )
    jsonl_path = SPLIT_DIR / f"{split}.jsonl"
    with jsonl_path.open(
        "w",
        encoding="utf-8",
    ) as output:
        for record in df.to_dict(
            orient="records"
        ):
            output.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(
        f"{split}: {len(df)} records"
    )
    print(
        df["scene_flood_burden"]
        .value_counts()
        .to_dict()
    )
print(f"Backup: {backup_dir}")
print("STURM splits patched to classification-only targets.")
