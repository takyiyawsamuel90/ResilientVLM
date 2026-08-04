from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
SOURCE_DIR = Path(
    "data/processed/sturm_vlm_quick/splits"
).resolve()
OUTPUT_DIR = Path(
    "data/processed/sturm_vlm_quick/training_splits"
).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CLASSES = ["Low", "Moderate", "High"]
TRAIN_RECORDS_PER_CLASS = 500
RANDOM_SEED = 42
def load_split(name: str) -> pd.DataFrame:
    path = SOURCE_DIR / f"{name}.csv"
    # Prevent the literal category "None" from becoming NaN.
    df = pd.read_csv(
        path,
        keep_default_na=False,
    )
    required = {
        "instruction_id",
        "scene_id",
        "floodmap_id",
        "scene_flood_burden",
        "s1_png_path",
        "s2_png_path",
        "instruction_text",
        "response_text",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"{path} is missing: {sorted(missing)}"
        )
    return df
train_df = load_split("train")
validation_df = load_split("validation")
test_df = load_split("test")
# The no-flood class contains only 10 total examples.
# Exclude it from this rapid three-class auxiliary experiment.
train_df = train_df[
    train_df["scene_flood_burden"].isin(CLASSES)
].copy()
validation_df = validation_df[
    validation_df["scene_flood_burden"].isin(CLASSES)
].copy()
test_df = test_df[
    test_df["scene_flood_burden"].isin(CLASSES)
].copy()
# ------------------------------------------------------------------
# Balance training through deterministic class-specific sampling.
# Moderate and High may be oversampled; Low may be downsampled.
# ------------------------------------------------------------------
balanced_parts = []
for class_index, class_name in enumerate(CLASSES):
    class_df = train_df[
        train_df["scene_flood_burden"] == class_name
    ].copy()
    if class_df.empty:
        raise RuntimeError(
            f"No training records found for {class_name}"
        )
    sampled = class_df.sample(
        n=TRAIN_RECORDS_PER_CLASS,
        replace=len(class_df) < TRAIN_RECORDS_PER_CLASS,
        random_state=RANDOM_SEED + class_index,
    ).copy()
    sampled["source_instruction_id"] = sampled[
        "instruction_id"
    ]
    sampled["instruction_id"] = [
        f"{value}_balanced_{index:04d}"
        for index, value in enumerate(
            sampled["instruction_id"].astype(str)
        )
    ]
    balanced_parts.append(sampled)
balanced_train_df = (
    pd.concat(
        balanced_parts,
        ignore_index=True,
    )
    .sample(
        frac=1.0,
        random_state=RANDOM_SEED,
    )
    .reset_index(drop=True)
)
# ------------------------------------------------------------------
# Confirm event-group separation.
# ------------------------------------------------------------------
train_groups = set(
    balanced_train_df["floodmap_id"].astype(str)
)
validation_groups = set(
    validation_df["floodmap_id"].astype(str)
)
test_groups = set(
    test_df["floodmap_id"].astype(str)
)
if train_groups & validation_groups:
    raise RuntimeError(
        "Train/validation floodmap leakage detected."
    )
if train_groups & test_groups:
    raise RuntimeError(
        "Train/test floodmap leakage detected."
    )
if validation_groups & test_groups:
    raise RuntimeError(
        "Validation/test floodmap leakage detected."
    )
def save_split(
    name: str,
    dataframe: pd.DataFrame,
) -> None:
    dataframe = dataframe.reset_index(drop=True)
    dataframe.to_csv(
        OUTPUT_DIR / f"{name}.csv",
        index=False,
    )
    with (
        OUTPUT_DIR / f"{name}.jsonl"
    ).open("w", encoding="utf-8") as output:
        for record in dataframe.to_dict(
            orient="records"
        ):
            output.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )
save_split("train", balanced_train_df)
save_split("validation", validation_df)
save_split("test", test_df)
summary = {
    "classification_classes": CLASSES,
    "training_strategy": (
        "Balanced deterministic sampling with replacement "
        "where required"
    ),
    "records": {
        "train": len(balanced_train_df),
        "validation": len(validation_df),
        "test": len(test_df),
    },
    "class_counts": {
        "train": (
            balanced_train_df[
                "scene_flood_burden"
            ].value_counts().to_dict()
        ),
        "validation": (
            validation_df[
                "scene_flood_burden"
            ].value_counts().to_dict()
        ),
        "test": (
            test_df[
                "scene_flood_burden"
            ].value_counts().to_dict()
        ),
    },
    "floodmap_groups": {
        "train": len(train_groups),
        "validation": len(validation_groups),
        "test": len(test_groups),
    },
    "group_leakage": False,
}
(
    OUTPUT_DIR / "training_split_summary.json"
).write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)
print(json.dumps(summary, indent=2))
