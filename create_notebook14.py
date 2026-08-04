import json
from pathlib import Path
from textwrap import dedent
notebook_path = Path(
    "notebooks/14_label_hidden_vision_evaluation.ipynb"
)
cells = []
def add_markdown(text):
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": dedent(text).strip().splitlines(
                keepends=True
            ),
        }
    )
def add_code(text):
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": dedent(text).strip().splitlines(
                keepends=True
            ),
        }
    )
add_markdown(
    """
    # RoadFlood-VLM Label-Hidden Vision Evaluation
    This notebook evaluates whether RoadFlood-VLM uses Sentinel imagery
    to infer scene flood burden and critical-network disruption.
    The target labels and transportation-grounding values are removed from
    the prompt. The same trained adapter is evaluated under five controlled
    image conditions:
    1. Correct Sentinel-1 and Sentinel-2 images
    2. Sentinel-1 shuffled between scenes
    3. Sentinel-2 shuffled between scenes
    4. Both image modalities shuffled
    5. No images
    The experiment is performed without retraining.
    """
)
add_code(
    """
    from __future__ import annotations
    import gc
    import json
    import os
    import re
    from collections import Counter
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
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
    )
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    """
)
add_code(
    """
    def find_project_root() -> Path:
        start = Path.cwd().resolve()
        for candidate in [start, *start.parents]:
            if (
                (candidate / "data").exists()
                and (candidate / "notebooks").exists()
            ):
                return candidate
        raise FileNotFoundError(
            "Run from the ResilientVLM repository "
            "or one of its subdirectories."
        )
    PROJECT_ROOT = find_project_root()
    DATASET_ROOT = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "vlm_dataset"
    )
    TEST_CSV = (
        DATASET_ROOT
        / "splits"
        / "test.csv"
    )
    TRAINING_ROOT = (
        PROJECT_ROOT
        / "outputs"
        / "roadflood_vlm_training"
    )
    STAMP = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
    OUTPUT_ROOT = (
        PROJECT_ROOT
        / "outputs"
        / "roadflood_vlm_label_hidden"
    )
    RUN_DIR = (
        OUTPUT_ROOT
        / f"evaluation_{STAMP}"
    )
    RUN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    MODEL_ID = os.environ.get(
        "ROADFLOOD_MODEL_ID",
        "Qwen/Qwen2.5-VL-3B-Instruct",
    )
    ADAPTER_PATH_RAW = os.environ.get(
        "ROADFLOOD_ADAPTER_PATH",
        "",
    ).strip()
    if ADAPTER_PATH_RAW:
        ADAPTER_DIR = Path(
            ADAPTER_PATH_RAW
        ).expanduser().resolve()
    else:
        full_runs = sorted(
            path
            for path in TRAINING_ROOT.glob("full_*")
            if (
                path
                / "final_adapter"
            ).exists()
        )
        if not full_runs:
            raise FileNotFoundError(
                "No full training adapter was found."
            )
        ADAPTER_DIR = (
            full_runs[-1]
            / "final_adapter"
        )
    if not TEST_CSV.exists():
        raise FileNotFoundError(TEST_CSV)
    print("Project root:", PROJECT_ROOT)
    print("Test CSV:", TEST_CSV)
    print("Adapter:", ADAPTER_DIR)
    print("Output:", RUN_DIR)
    """
)
add_code(
    """
    test_df = pd.read_csv(TEST_CSV)
    required_columns = {
        "instruction_id",
        "scene_id",
        "scene_flood_burden",
        "critical_network_disruption",
        "training_s2_path",
        "training_s1_path",
    }
    missing = (
        required_columns
        - set(test_df.columns)
    )
    if missing:
        raise KeyError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )
    def resolve_path(value: Any) -> Path:
        raw = (
            ""
            if pd.isna(value)
            else str(value).strip()
        )
        if not raw:
            return Path("")
        path = Path(raw)
        if (
            path.is_absolute()
            and path.exists()
        ):
            return path
        normalized = raw.replace(
            "\\\\",
            "/",
        )
        for marker in [
            "data/",
            "training_images/",
        ]:
            index = normalized.find(marker)
            if index >= 0:
                suffix = Path(
                    normalized[index:]
                )
                for root in [
                    PROJECT_ROOT,
                    DATASET_ROOT,
                ]:
                    candidate = (
                        root
                        / suffix
                    )
                    if candidate.exists():
                        return candidate.resolve()
        return path
    test_df["s2"] = (
        test_df["training_s2_path"]
        .map(resolve_path)
    )
    test_df["s1"] = (
        test_df["training_s1_path"]
        .map(resolve_path)
    )
    valid = (
        test_df["s2"].map(Path.exists)
        & test_df["s1"].map(Path.exists)
        & test_df[
            "scene_flood_burden"
        ].notna()
        & test_df[
            "critical_network_disruption"
        ].notna()
    )
    test_df = (
        test_df.loc[valid]
        .reset_index(drop=True)
    )
    max_records = int(
        os.environ.get(
            "ROADFLOOD_VISION_MAX_RECORDS",
            "0",
        )
    )
    if max_records > 0:
        test_df = test_df.head(
            max_records
        )
    print("Evaluation records:", len(test_df))
    print(
        "Unique scenes:",
        test_df["scene_id"].nunique(),
    )
    """
)
add_markdown(
    """
    ## Controlled label-hidden prompt
    The prompt contains no scene ID, target class, roadway count, exposure
    percentage, disruption score, rank, or grounding-reliability label.
    """
)
add_code(
    """
    LABELS = [
        "None",
        "Low",
        "Moderate",
        "High",
    ]
    SYSTEM_PROMPT = (
        "You are RoadFlood-VLM, a transportation-flood "
        "assessment assistant. Infer transportation flood "
        "conditions from the supplied satellite imagery. "
        "Do not invent numerical roadway statistics. "
        "Return only the requested JSON object."
    )
    LABEL_HIDDEN_PROMPT = '''
    Analyze the supplied Sentinel imagery.
    Predict both categorical outcomes:
    1. scene_flood_burden:
       None, Low, Moderate, or High
    2. critical_network_disruption:
       None, Low, Moderate, or High
    Use visible flood extent, apparent roadway exposure,
    network interruption, and uncertainty.
    Return only valid JSON using exactly this schema:
    {
      "scene_flood_burden": "<class>",
      "critical_network_disruption": "<class>"
    }
    '''.strip()
    print(LABEL_HIDDEN_PROMPT)
    """
)
add_code(
    """
    CONDITIONS = [
        "correct_images",
        "s1_shuffled",
        "s2_shuffled",
        "both_shuffled",
        "no_images",
    ]
    RANDOM_SEED = 42
    rng = np.random.default_rng(
        RANDOM_SEED
    )
    scene_table = (
        test_df[
            [
                "scene_id",
                "s1",
                "s2",
            ]
        ]
        .drop_duplicates("scene_id")
        .reset_index(drop=True)
    )
    scene_ids = (
        scene_table["scene_id"]
        .tolist()
    )
    def shuffled_mapping(
        modality: str,
        seed_offset: int,
    ) -> dict[str, Path]:
        local_rng = np.random.default_rng(
            RANDOM_SEED
            + seed_offset
        )
        paths = (
            scene_table[modality]
            .tolist()
        )
        shuffled = paths.copy()
        if len(shuffled) > 1:
            for _ in range(100):
                local_rng.shuffle(shuffled)
                if all(
                    original != replacement
                    for original, replacement
                    in zip(paths, shuffled)
                ):
                    break
        return dict(
            zip(
                scene_ids,
                shuffled,
            )
        )
    S1_SHUFFLED = shuffled_mapping(
        "s1",
        1,
    )
    S2_SHUFFLED = shuffled_mapping(
        "s2",
        2,
    )
    print("Conditions:", CONDITIONS)
    """
)
add_code(
    """
    use_4bit = (
        os.environ.get(
            "ROADFLOOD_USE_4BIT",
            "1",
        )
        == "1"
        and torch.cuda.is_available()
    )
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
        quantization_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
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
    MAX_NEW_TOKENS = int(
        os.environ.get(
            "ROADFLOOD_MAX_NEW_TOKENS",
            "80",
        )
    )
    print("Model loaded.")
    print("4-bit:", use_4bit)
    print("dtype:", dtype)
    """
)
add_code(
    """
    def image_paths_for_condition(
        row: pd.Series,
        condition: str,
    ) -> tuple[
        Path | None,
        Path | None,
    ]:
        scene_id = str(
            row["scene_id"]
        )
        if condition == "correct_images":
            return (
                row["s2"],
                row["s1"],
            )
        if condition == "s1_shuffled":
            return (
                row["s2"],
                S1_SHUFFLED[scene_id],
            )
        if condition == "s2_shuffled":
            return (
                S2_SHUFFLED[scene_id],
                row["s1"],
            )
        if condition == "both_shuffled":
            return (
                S2_SHUFFLED[scene_id],
                S1_SHUFFLED[scene_id],
            )
        if condition == "no_images":
            return None, None
        raise ValueError(condition)
    def build_messages(
        row: pd.Series,
        condition: str,
    ) -> list[dict[str, Any]]:
        s2_path, s1_path = (
            image_paths_for_condition(
                row,
                condition,
            )
        )
        content = []
        if s2_path is not None:
            content.append(
                {
                    "type": "image",
                    "image": str(s2_path),
                }
            )
        if s1_path is not None:
            content.append(
                {
                    "type": "image",
                    "image": str(s1_path),
                }
            )
        content.append(
            {
                "type": "text",
                "text": LABEL_HIDDEN_PROMPT,
            }
        )
        return [
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
                "content": content,
            },
        ]
    """
)
add_code(
    """
    CATEGORY_PATTERN = re.compile(
        r"(None|Low|Moderate|High)",
        flags=re.IGNORECASE,
    )
    def normalize_category(
        value: Any,
    ) -> str | None:
        if value is None:
            return None
        match = CATEGORY_PATTERN.search(
            str(value)
        )
        if not match:
            return None
        return match.group(1).title()
    def parse_prediction(
        text: str,
    ) -> dict[str, Any]:
        result = {
            "json_valid": False,
            "predicted_flood_burden": None,
            "predicted_disruption": None,
        }
        cleaned = str(text).strip()
        cleaned = re.sub(
            r"^```(?:json)?\\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\\s*```$",
            "",
            cleaned,
        )
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                result["json_valid"] = True
                result[
                    "predicted_flood_burden"
                ] = normalize_category(
                    parsed.get(
                        "scene_flood_burden"
                    )
                )
                result[
                    "predicted_disruption"
                ] = normalize_category(
                    parsed.get(
                        "critical_network_disruption"
                    )
                )
                return result
        except json.JSONDecodeError:
            pass
        flood_match = re.search(
            r"scene[_\\s-]*flood[_\\s-]*burden"
            r'[^A-Za-z]+'
            r"(None|Low|Moderate|High)",
            cleaned,
            flags=re.IGNORECASE,
        )
        disruption_match = re.search(
            r"critical[_\\s-]*network[_\\s-]*"
            r"disruption"
            r'[^A-Za-z]+'
            r"(None|Low|Moderate|High)",
            cleaned,
            flags=re.IGNORECASE,
        )
        if flood_match:
            result[
                "predicted_flood_burden"
            ] = flood_match.group(1).title()
        if disruption_match:
            result[
                "predicted_disruption"
            ] = disruption_match.group(1).title()
        return result
    """
)
add_code(
    """
    def generate_prediction(
        row: pd.Series,
        condition: str,
    ) -> str:
        messages = build_messages(
            row,
            condition,
        )
        text = (
            processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
        image_inputs, video_inputs = (
            process_vision_info(messages)
        )
        processor_kwargs = {
            "text": [text],
            "padding": True,
            "return_tensors": "pt",
        }
        if image_inputs:
            processor_kwargs[
                "images"
            ] = [image_inputs]
        if video_inputs:
            processor_kwargs[
                "videos"
            ] = [video_inputs]
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
            for key, value
            in batch.items()
        }
        prompt_length = (
            batch["input_ids"]
            .shape[1]
        )
        with torch.inference_mode():
            generated_ids = model.generate(
                **batch,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
            )
        generated_only = (
            generated_ids[
                :,
                prompt_length:,
            ]
        )
        prediction = (
            processor.batch_decode(
                generated_only,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            .strip()
        )
        del batch
        del generated_ids
        del generated_only
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return prediction
    """
)
add_code(
    """
    all_prediction_files = {}
    for condition in CONDITIONS:
        condition_dir = (
            RUN_DIR
            / condition
        )
        condition_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        output_csv = (
            condition_dir
            / "predictions.csv"
        )
        if output_csv.exists():
            existing = pd.read_csv(
                output_csv
            )
            completed = set(
                existing[
                    "record_index"
                ].astype(int)
            )
            records = existing.to_dict(
                orient="records"
            )
        else:
            completed = set()
            records = []
        print(
            "\\nRUNNING CONDITION:",
            condition,
        )
        for index, row in test_df.iterrows():
            if int(index) in completed:
                continue
            prediction = generate_prediction(
                row,
                condition,
            )
            parsed = parse_prediction(
                prediction
            )
            s2_path, s1_path = (
                image_paths_for_condition(
                    row,
                    condition,
                )
            )
            record = {
                "record_index": int(index),
                "condition": condition,
                "instruction_id": str(
                    row["instruction_id"]
                ),
                "scene_id": str(
                    row["scene_id"]
                ),
                "reference_flood_burden": (
                    normalize_category(
                        row[
                            "scene_flood_burden"
                        ]
                    )
                ),
                "reference_disruption": (
                    normalize_category(
                        row[
                            "critical_network_disruption"
                        ]
                    )
                ),
                "prediction_text": prediction,
                "prediction_json_valid": (
                    parsed["json_valid"]
                ),
                "predicted_flood_burden": (
                    parsed[
                        "predicted_flood_burden"
                    ]
                ),
                "predicted_disruption": (
                    parsed[
                        "predicted_disruption"
                    ]
                ),
                "s2_path_used": (
                    ""
                    if s2_path is None
                    else str(s2_path)
                ),
                "s1_path_used": (
                    ""
                    if s1_path is None
                    else str(s1_path)
                ),
            }
            records.append(record)
            pd.DataFrame(records).to_csv(
                output_csv,
                index=False,
            )
            print(
                f"[{condition}] "
                f"[{index + 1}/{len(test_df)}] "
                f"scene={record['scene_id']} "
                f"flood={record['predicted_flood_burden']} "
                f"disruption={record['predicted_disruption']}"
            )
        all_prediction_files[
            condition
        ] = output_csv
    print(all_prediction_files)
    """
)
add_code(
    """
    def classification_metrics(
        reference: pd.Series,
        prediction: pd.Series,
    ) -> dict[str, Any]:
        reference = (
            reference
            .fillna("Missing")
            .astype(str)
        )
        prediction = (
            prediction
            .fillna("Missing")
            .astype(str)
        )
        labels_with_missing = (
            LABELS
            + ["Missing"]
        )
        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                reference,
                prediction,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        )
        matrix = confusion_matrix(
            reference,
            prediction,
            labels=labels_with_missing,
        )
        return {
            "records": int(
                len(reference)
            ),
            "accuracy": float(
                accuracy_score(
                    reference,
                    prediction,
                )
            ),
            "macro_precision": float(
                precision
            ),
            "macro_recall": float(
                recall
            ),
            "macro_f1": float(
                f1
            ),
            "missing_prediction_rate": float(
                (
                    prediction
                    == "Missing"
                ).mean()
            ),
            "confusion_matrix": (
                matrix.tolist()
            ),
            "confusion_labels": (
                labels_with_missing
            ),
        }
    summary_records = []
    metrics_by_condition = {}
    for condition, path in (
        all_prediction_files.items()
    ):
        dataframe = pd.read_csv(path)
        flood_metrics = (
            classification_metrics(
                dataframe[
                    "reference_flood_burden"
                ],
                dataframe[
                    "predicted_flood_burden"
                ],
            )
        )
        disruption_metrics = (
            classification_metrics(
                dataframe[
                    "reference_disruption"
                ],
                dataframe[
                    "predicted_disruption"
                ],
            )
        )
        json_validity = float(
            dataframe[
                "prediction_json_valid"
            ]
            .astype(bool)
            .mean()
        )
        metrics = {
            "condition": condition,
            "record_level": {
                "flood_burden": (
                    flood_metrics
                ),
                "critical_network_disruption": (
                    disruption_metrics
                ),
                "json_validity": (
                    json_validity
                ),
            },
        }
        metrics_by_condition[
            condition
        ] = metrics
        metrics_path = (
            RUN_DIR
            / condition
            / "metrics.json"
        )
        metrics_path.write_text(
            json.dumps(
                metrics,
                indent=2,
            ),
            encoding="utf-8",
        )
        summary_records.append(
            {
                "condition": condition,
                "records": len(dataframe),
                "flood_accuracy": (
                    flood_metrics["accuracy"]
                ),
                "flood_macro_precision": (
                    flood_metrics[
                        "macro_precision"
                    ]
                ),
                "flood_macro_recall": (
                    flood_metrics[
                        "macro_recall"
                    ]
                ),
                "flood_macro_f1": (
                    flood_metrics[
                        "macro_f1"
                    ]
                ),
                "disruption_accuracy": (
                    disruption_metrics[
                        "accuracy"
                    ]
                ),
                "disruption_macro_precision": (
                    disruption_metrics[
                        "macro_precision"
                    ]
                ),
                "disruption_macro_recall": (
                    disruption_metrics[
                        "macro_recall"
                    ]
                ),
                "disruption_macro_f1": (
                    disruption_metrics[
                        "macro_f1"
                    ]
                ),
                "json_validity": (
                    json_validity
                ),
            }
        )
    summary_df = pd.DataFrame(
        summary_records
    )
    summary_csv = (
        RUN_DIR
        / "label_hidden_summary.csv"
    )
    summary_json = (
        RUN_DIR
        / "label_hidden_summary.json"
    )
    summary_df.to_csv(
        summary_csv,
        index=False,
    )
    summary_json.write_text(
        json.dumps(
            metrics_by_condition,
            indent=2,
        ),
        encoding="utf-8",
    )
    display(
        summary_df.round(4)
    )
    """
)
add_markdown(
    """
    ## Scene-level sensitivity analysis
    The test split contains repeated instructions from the same scenes.
    The following analysis collapses repeated records by scene using majority
    voting. These scene-level results should be emphasized over the inflated
    record-level counts.
    """
)
add_code(
    """
    scene_summary_records = []
    for condition, path in (
        all_prediction_files.items()
    ):
        dataframe = pd.read_csv(path)
        def majority_vote(series):
            valid = (
                series
                .dropna()
                .astype(str)
            )
            if valid.empty:
                return None
            counts = Counter(valid)
            return counts.most_common(
                1
            )[0][0]
        scene_df = (
            dataframe
            .groupby(
                "scene_id",
                as_index=False,
            )
            .agg(
                reference_flood_burden=(
                    "reference_flood_burden",
                    "first",
                ),
                reference_disruption=(
                    "reference_disruption",
                    "first",
                ),
                predicted_flood_burden=(
                    "predicted_flood_burden",
                    majority_vote,
                ),
                predicted_disruption=(
                    "predicted_disruption",
                    majority_vote,
                ),
            )
        )
        scene_df.to_csv(
            RUN_DIR
            / condition
            / "scene_level_predictions.csv",
            index=False,
        )
        flood_metrics = (
            classification_metrics(
                scene_df[
                    "reference_flood_burden"
                ],
                scene_df[
                    "predicted_flood_burden"
                ],
            )
        )
        disruption_metrics = (
            classification_metrics(
                scene_df[
                    "reference_disruption"
                ],
                scene_df[
                    "predicted_disruption"
                ],
            )
        )
        scene_summary_records.append(
            {
                "condition": condition,
                "unique_scenes": len(
                    scene_df
                ),
                "flood_accuracy": (
                    flood_metrics["accuracy"]
                ),
                "flood_macro_precision": (
                    flood_metrics[
                        "macro_precision"
                    ]
                ),
                "flood_macro_recall": (
                    flood_metrics[
                        "macro_recall"
                    ]
                ),
                "flood_macro_f1": (
                    flood_metrics[
                        "macro_f1"
                    ]
                ),
                "disruption_accuracy": (
                    disruption_metrics[
                        "accuracy"
                    ]
                ),
                "disruption_macro_precision": (
                    disruption_metrics[
                        "macro_precision"
                    ]
                ),
                "disruption_macro_recall": (
                    disruption_metrics[
                        "macro_recall"
                    ]
                ),
                "disruption_macro_f1": (
                    disruption_metrics[
                        "macro_f1"
                    ]
                ),
            }
        )
    scene_summary_df = pd.DataFrame(
        scene_summary_records
    )
    scene_summary_df.to_csv(
        RUN_DIR
        / "label_hidden_scene_summary.csv",
        index=False,
    )
    display(
        scene_summary_df.round(4)
    )
    """
)
add_code(
    """
    condition_labels = {
        "correct_images": "Correct Images",
        "s1_shuffled": "S1 Shuffled",
        "s2_shuffled": "S2 Shuffled",
        "both_shuffled": "Both Shuffled",
        "no_images": "No Images",
    }
    plot_df = summary_df.copy()
    plot_df["condition_label"] = (
        plot_df["condition"]
        .map(condition_labels)
    )
    plot_df = (
        plot_df
        .set_index("condition_label")
    )
    figure, axis = plt.subplots(
        figsize=(11, 6.5)
    )
    plot_df[
        [
            "flood_accuracy",
            "flood_macro_f1",
            "disruption_accuracy",
            "disruption_macro_f1",
        ]
    ].plot(
        kind="bar",
        ax=axis,
        width=0.82,
    )
    axis.set_title(
        "Label-Hidden Vision-Language Classification"
    )
    axis.set_xlabel(
        "Image condition"
    )
    axis.set_ylabel(
        "Score"
    )
    axis.set_ylim(
        0,
        1,
    )
    axis.tick_params(
        axis="x",
        rotation=20,
    )
    axis.grid(
        axis="y",
        alpha=0.25,
    )
    axis.legend(
        title="Metric",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    figure.tight_layout()
    figure_path = (
        RUN_DIR
        / "label_hidden_classification.png"
    )
    figure.savefig(
        figure_path,
        dpi=400,
        bbox_inches="tight",
    )
    plt.show()
    print(figure_path)
    """
)
add_code(
    """
    confusion_figure_paths = []
    for condition, path in (
        all_prediction_files.items()
    ):
        dataframe = pd.read_csv(path)
        for target_name, reference_column, prediction_column in [
            (
                "flood_burden",
                "reference_flood_burden",
                "predicted_flood_burden",
            ),
            (
                "critical_disruption",
                "reference_disruption",
                "predicted_disruption",
            ),
        ]:
            matrix = confusion_matrix(
                dataframe[
                    reference_column
                ].fillna("Missing"),
                dataframe[
                    prediction_column
                ].fillna("Missing"),
                labels=(
                    LABELS
                    + ["Missing"]
                ),
            )
            figure, axis = plt.subplots(
                figsize=(6.5, 5.5)
            )
            image = axis.imshow(
                matrix,
                aspect="auto",
            )
            axis.set_title(
                f"{condition_labels[condition]}: "
                f"{target_name.replace('_', ' ').title()}"
            )
            axis.set_xlabel(
                "Predicted class"
            )
            axis.set_ylabel(
                "Reference class"
            )
            labels = (
                LABELS
                + ["Missing"]
            )
            axis.set_xticks(
                np.arange(len(labels))
            )
            axis.set_xticklabels(
                labels,
                rotation=30,
                ha="right",
            )
            axis.set_yticks(
                np.arange(len(labels))
            )
            axis.set_yticklabels(
                labels,
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
            output_path = (
                RUN_DIR
                / (
                    f"confusion_{condition}_"
                    f"{target_name}.png"
                )
            )
            figure.savefig(
                output_path,
                dpi=400,
                bbox_inches="tight",
            )
            plt.close(figure)
            confusion_figure_paths.append(
                str(output_path)
            )
    print(
        "\\n".join(
            confusion_figure_paths
        )
    )
    """
)
add_code(
    """
    manifest = {
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "model_id": MODEL_ID,
        "adapter_dir": str(
            ADAPTER_DIR
        ),
        "test_csv": str(TEST_CSV),
        "evaluation_records": int(
            len(test_df)
        ),
        "unique_scenes": int(
            test_df[
                "scene_id"
            ].nunique()
        ),
        "conditions": CONDITIONS,
        "random_seed": RANDOM_SEED,
        "summary_csv": str(
            summary_csv
        ),
        "summary_json": str(
            summary_json
        ),
        "scene_summary_csv": str(
            RUN_DIR
            / "label_hidden_scene_summary.csv"
        ),
        "classification_figure": str(
            figure_path
        ),
        "confusion_figures": (
            confusion_figure_paths
        ),
    }
    manifest_path = (
        RUN_DIR
        / "run_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            manifest,
            indent=2,
        )
    )
    """
)
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "resilientvlm",
            "language": "python",
            "name": "resilientvlm",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
notebook_path.write_text(
    json.dumps(
        notebook,
        indent=2,
    ),
    encoding="utf-8",
)
print(f"Created: {notebook_path}")
print(f"Cells: {len(cells)}")
