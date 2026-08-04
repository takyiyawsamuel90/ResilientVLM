from __future__ import annotations
import math
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
PROJECT_ROOT = Path.cwd().resolve()
PREDICTIONS_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "sturm_classification_evaluation"
    / "evaluation_20260802T000516Z"
    / "predictions.csv"
)
TEST_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sturm_vlm_quick"
    / "training_splits"
    / "test.csv"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "sturm_classification_evaluation"
    / "evaluation_20260802T000516Z"
    / "vision_figures"
)
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
def resolve_path(value: str) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
def first_existing_column(
    frame: pd.DataFrame,
    candidates: list[str],
) -> str:
    for column in candidates:
        if column in frame.columns:
            return column
    raise KeyError(
        f"None of these columns were found: {candidates}"
    )
predictions = pd.read_csv(
    PREDICTIONS_CSV,
    keep_default_na=False,
)
test = pd.read_csv(
    TEST_CSV,
    keep_default_na=False,
)
merge_columns = [
    column
    for column in [
        "instruction_id",
        "scene_id",
        "floodmap_id",
    ]
    if column in predictions.columns
    and column in test.columns
]
if not merge_columns:
    predictions["record_index"] = predictions.index
    test["record_index"] = test.index
    merge_columns = ["record_index"]
data = predictions.merge(
    test,
    on=merge_columns,
    how="left",
    suffixes=("", "_test"),
)
s2_column = first_existing_column(
    data,
    [
        "s2_png_path",
        "training_s2_path",
        "s2_path",
        "s2",
    ],
)
s1_column = first_existing_column(
    data,
    [
        "s1_png_path",
        "training_s1_path",
        "s1_path",
        "s1",
    ],
)
mask_column = None
for candidate in [
    "flood_mask_path",
    "mask_path",
    "label_path",
    "floodmap_path",
    "mask_png_path",
]:
    if candidate in data.columns:
        mask_column = candidate
        break
data["correct"] = (
    data["reference_label"].astype(str)
    == data["predicted_label"].astype(str)
)
print("Records:", len(data))
print("Correct:", int(data["correct"].sum()))
print("Incorrect:", int((~data["correct"]).sum()))
print("S2 column:", s2_column)
print("S1 column:", s1_column)
print("Mask column:", mask_column)
def load_image(path_value: str) -> Image.Image:
    path = resolve_path(path_value)
    if not path.exists():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")
def plot_record_row(
    axis_row,
    row: pd.Series,
    include_mask: bool,
) -> None:
    images = [
        (
            "Sentinel-2 Optical",
            load_image(row[s2_column]),
        ),
        (
            "Sentinel-1 SAR",
            load_image(row[s1_column]),
        ),
    ]
    if include_mask and mask_column:
        mask_path = resolve_path(row[mask_column])
        if mask_path.exists():
            images.append(
                (
                    "Reference Flood Mask",
                    Image.open(mask_path).convert("RGB"),
                )
            )
    for axis, (title, image) in zip(
        axis_row,
        images,
    ):
        axis.imshow(image)
        axis.set_title(
            title,
            fontsize=10,
        )
        axis.axis("off")
    status = (
        "Correct"
        if row["correct"]
        else "Incorrect"
    )
    title = (
        f"Reference: {row['reference_label']} | "
        f"Prediction: {row['predicted_label']} | "
        f"{status}"
    )
    axis_row[0].set_ylabel(
        title,
        rotation=90,
        fontsize=9,
        labelpad=18,
    )
def save_examples(
    frame: pd.DataFrame,
    filename: str,
    figure_title: str,
    records_per_figure: int = 4,
) -> None:
    selected = frame.head(
        records_per_figure
    ).copy()
    if selected.empty:
        print(
            f"No records available for {filename}"
        )
        return
    number_columns = (
        3
        if mask_column
        else 2
    )
    figure, axes = plt.subplots(
        len(selected),
        number_columns,
        figsize=(
            4.6 * number_columns,
            3.7 * len(selected),
        ),
        squeeze=False,
    )
    for row_index, (_, row) in enumerate(
        selected.iterrows()
    ):
        plot_record_row(
            axes[row_index],
            row,
            include_mask=bool(mask_column),
        )
    figure.suptitle(
        figure_title,
        fontsize=15,
        y=1.01,
    )
    figure.tight_layout()
    output_path = OUTPUT_DIR / filename
    figure.savefig(
        output_path,
        dpi=400,
        bbox_inches="tight",
    )
    plt.close(figure)
    print("Saved:", output_path)
correct_examples = (
    data[data["correct"]]
    .sort_values(
        ["reference_label", "scene_id"]
        if "scene_id" in data.columns
        else ["reference_label"]
    )
)
incorrect_examples = (
    data[~data["correct"]]
    .sort_values(
        ["reference_label", "scene_id"]
        if "scene_id" in data.columns
        else ["reference_label"]
    )
)
moderate_errors = data[
    (
        data["reference_label"]
        == "Moderate"
    )
    & (
        ~data["correct"]
    )
]
save_examples(
    correct_examples,
    "figure_vision_correct_examples.png",
    "Representative Correct Multimodal Flood-Burden Predictions",
    records_per_figure=4,
)
save_examples(
    incorrect_examples,
    "figure_vision_incorrect_examples.png",
    "Representative Incorrect Multimodal Flood-Burden Predictions",
    records_per_figure=4,
)
save_examples(
    moderate_errors,
    "figure_moderate_class_errors.png",
    "Moderate Flood-Burden Misclassification Examples",
    records_per_figure=4,
)
# One example from each reference class
class_examples = []
for label in [
    "Low",
    "Moderate",
    "High",
]:
    class_frame = data[
        data["reference_label"]
        == label
    ]
    correct_class = class_frame[
        class_frame["correct"]
    ]
    if not correct_class.empty:
        class_examples.append(
            correct_class.iloc[0]
        )
    elif not class_frame.empty:
        class_examples.append(
            class_frame.iloc[0]
        )
if class_examples:
    class_examples_df = pd.DataFrame(
        class_examples
    )
    save_examples(
        class_examples_df,
        "figure_low_moderate_high_examples.png",
        "Representative Low, Moderate, and High Flood-Burden Scenes",
        records_per_figure=3,
    )
summary = (
    data.groupby(
        [
            "reference_label",
            "predicted_label",
        ],
        dropna=False,
    )
    .size()
    .reset_index(
        name="records"
    )
)
summary.to_csv(
    OUTPUT_DIR
    / "vision_prediction_summary.csv",
    index=False,
)
data.to_csv(
    OUTPUT_DIR
    / "vision_predictions_with_paths.csv",
    index=False,
)
print(
    "\nGenerated qualitative vision figures in:",
    OUTPUT_DIR,
)
