from __future__ import annotations
import gc
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)
PROJECT_ROOT = Path.cwd().resolve()
TEST_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sturm_vlm_quick"
    / "training_splits"
    / "test.csv"
)
ADAPTER_DIR = Path(
    os.environ.get(
        "STURM_ADAPTER_DIR",
        str(
            PROJECT_ROOT
            / "outputs"
            / "sturm_auxiliary_training"
            / "full_20260801T213141Z"
            / "final_adapter"
        ),
    )
).resolve()
MODEL_ID = os.environ.get(
    "ROADFLOOD_MODEL_ID",
    "Qwen/Qwen2.5-VL-3B-Instruct",
)
STAMP = datetime.now(
    timezone.utc
).strftime("%Y%m%dT%H%M%SZ")
OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "sturm_classification_evaluation"
    / f"evaluation_{STAMP}"
)
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
LABELS = [
    "Low",
    "Moderate",
    "High",
]
SYSTEM_PROMPT = (
    "You are RoadFlood-VLM, a multimodal satellite "
    "flood-classification assistant. Use Sentinel-2 optical "
    "imagery together with Sentinel-1 radar imagery to classify "
    "visible flood burden. Return only valid JSON."
)
USER_PROMPT = (
    "Analyze the Sentinel-2 optical image and Sentinel-1 radar image. "
    "Classify the visible scene flood burden as Low, Moderate, or High. "
    'Return only JSON using exactly this schema: '
    '{"scene_flood_burden":"Low|Moderate|High"}'
)
MAX_NEW_TOKENS = 40
def normalize_label(value: Any) -> str | None:
    text = str(value).strip().lower()
    mapping = {
        "low": "Low",
        "moderate": "Moderate",
        "medium": "Moderate",
        "high": "High",
    }
    return mapping.get(text)
def parse_prediction(text: str) -> dict[str, Any]:
    result = {
        "json_valid": False,
        "predicted_label": None,
    }
    cleaned = str(text).strip()
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            result["json_valid"] = True
            result["predicted_label"] = normalize_label(
                parsed.get("scene_flood_burden")
            )
            return result
    except json.JSONDecodeError:
        pass
    match = re.search(
        r"\b(Low|Moderate|Medium|High)\b",
        cleaned,
        flags=re.IGNORECASE,
    )
    if match:
        result["predicted_label"] = normalize_label(
            match.group(1)
        )
    return result
test_df = pd.read_csv(
    TEST_CSV,
    keep_default_na=False,
)
test_df = test_df[
    test_df["scene_flood_burden"].isin(
        LABELS
    )
].reset_index(drop=True)
print("Test records:", len(test_df))
print("Adapter:", ADAPTER_DIR)
use_4bit = torch.cuda.is_available()
dtype = (
    torch.bfloat16
    if (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )
    else torch.float16
    if torch.cuda.is_available()
    else torch.float32
)
quantization_config = None
if use_4bit:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
processor = AutoProcessor.from_pretrained(
    ADAPTER_DIR,
    trust_remote_code=True,
)
if processor.tokenizer.pad_token_id is None:
    processor.tokenizer.pad_token = (
        processor.tokenizer.eos_token
    )
model_kwargs = {
    "trust_remote_code": True,
    "torch_dtype": dtype,
    "low_cpu_mem_usage": True,
}
if torch.cuda.is_available():
    model_kwargs["device_map"] = "auto"
if quantization_config is not None:
    model_kwargs[
        "quantization_config"
    ] = quantization_config
base_model = (
    Qwen2_5_VLForConditionalGeneration
    .from_pretrained(
        MODEL_ID,
        **model_kwargs,
    )
)
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_DIR,
)
model.eval()
model.config.use_cache = True
records = []
for index, row in test_df.iterrows():
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(
                        row["s2_png_path"]
                    ),
                },
                {
                    "type": "image",
                    "image": str(
                        row["s1_png_path"]
                    ),
                },
                {
                    "type": "text",
                    "text": USER_PROMPT,
                },
            ],
        },
    ]
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = (
        process_vision_info(messages)
    )
    processor_kwargs = {
        "text": [prompt_text],
        "images": [image_inputs],
        "padding": True,
        "return_tensors": "pt",
    }
    if video_inputs:
        processor_kwargs["videos"] = [
            video_inputs
        ]
    batch = processor(
        **processor_kwargs
    )
    device = next(
        model.parameters()
    ).device
    batch = {
        key: (
            value.to(device)
            if isinstance(
                value,
                torch.Tensor,
            )
            else value
        )
        for key, value in batch.items()
    }
    prompt_length = batch[
        "input_ids"
    ].shape[1]
    with torch.inference_mode():
        generated = model.generate(
            **batch,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
        )
    generated_only = generated[
        :,
        prompt_length:,
    ]
    prediction_text = (
        processor.batch_decode(
            generated_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        .strip()
    )
    parsed = parse_prediction(
        prediction_text
    )
    reference_label = normalize_label(
        row["scene_flood_burden"]
    )
    records.append(
        {
            "record_index": int(index),
            "instruction_id": row[
                "instruction_id"
            ],
            "scene_id": row["scene_id"],
            "floodmap_id": row[
                "floodmap_id"
            ],
            "reference_label": (
                reference_label
            ),
            "predicted_label": (
                parsed["predicted_label"]
            ),
            "json_valid": (
                parsed["json_valid"]
            ),
            "prediction_text": (
                prediction_text
            ),
        }
    )
    pd.DataFrame(records).to_csv(
        OUTPUT_DIR / "predictions.csv",
        index=False,
    )
    print(
        f"[{index + 1}/{len(test_df)}] "
        f"reference={reference_label} "
        f"prediction={parsed['predicted_label']}"
    )
    del batch
    del generated
    del generated_only
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
prediction_df = pd.DataFrame(records)
reference_labels = (
    prediction_df["reference_label"]
    .astype(str)
)
predicted_labels = (
    prediction_df["predicted_label"]
    .fillna("Missing")
    .astype(str)
)
accuracy = accuracy_score(
    reference_labels,
    predicted_labels,
)
balanced_accuracy = balanced_accuracy_score(
    reference_labels,
    predicted_labels,
)
macro_precision, macro_recall, macro_f1, _ = (
    precision_recall_fscore_support(
        reference_labels,
        predicted_labels,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )
)
weighted_precision, weighted_recall, weighted_f1, _ = (
    precision_recall_fscore_support(
        reference_labels,
        predicted_labels,
        labels=LABELS,
        average="weighted",
        zero_division=0,
    )
)
report = classification_report(
    reference_labels,
    predicted_labels,
    labels=LABELS,
    output_dict=True,
    zero_division=0,
)
metrics = {
    "adapter_dir": str(
        ADAPTER_DIR
    ),
    "test_records": int(
        len(prediction_df)
    ),
    "accuracy": float(
        accuracy
    ),
    "balanced_accuracy": float(
        balanced_accuracy
    ),
    "macro_precision": float(
        macro_precision
    ),
    "macro_recall": float(
        macro_recall
    ),
    "macro_f1": float(
        macro_f1
    ),
    "weighted_precision": float(
        weighted_precision
    ),
    "weighted_recall": float(
        weighted_recall
    ),
    "weighted_f1": float(
        weighted_f1
    ),
    "json_validity": float(
        prediction_df[
            "json_valid"
        ].mean()
    ),
    "missing_label_rate": float(
        (
            predicted_labels
            == "Missing"
        ).mean()
    ),
    "classification_report": report,
}
(
    OUTPUT_DIR / "evaluation_metrics.json"
).write_text(
    json.dumps(
        metrics,
        indent=2,
    ),
    encoding="utf-8",
)
matrix = confusion_matrix(
    reference_labels,
    predicted_labels,
    labels=LABELS,
)
figure, axis = plt.subplots(
    figsize=(6.5, 5.5)
)
image = axis.imshow(
    matrix,
    aspect="auto",
)
axis.set_title(
    "STURM Flood-Burden Classification"
)
axis.set_xlabel(
    "Predicted class"
)
axis.set_ylabel(
    "Reference class"
)
axis.set_xticks(
    np.arange(len(LABELS))
)
axis.set_xticklabels(
    LABELS
)
axis.set_yticks(
    np.arange(len(LABELS))
)
axis.set_yticklabels(
    LABELS
)
for row_index in range(
    matrix.shape[0]
):
    for column_index in range(
        matrix.shape[1]
    ):
        axis.text(
            column_index,
            row_index,
            int(
                matrix[
                    row_index,
                    column_index,
                ]
            ),
            ha="center",
            va="center",
        )
figure.colorbar(
    image,
    ax=axis,
    label="Record count",
)
figure.tight_layout()
figure.savefig(
    OUTPUT_DIR
    / "confusion_matrix.png",
    dpi=400,
    bbox_inches="tight",
)
plt.close(figure)
print(
    json.dumps(
        metrics,
        indent=2,
    )
)
print("Output:", OUTPUT_DIR)
