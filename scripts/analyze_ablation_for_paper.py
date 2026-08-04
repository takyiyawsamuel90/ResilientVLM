from __future__ import annotations
import json
import math
import os
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None
# =============================================================================
# 1. CONFIGURATION
# =============================================================================
PROJECT_ROOT = Path.cwd().resolve()
ABLATION_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "roadflood_vlm_ablation"
)
RUN_DIR_RAW = os.environ.get(
    "ROADFLOOD_ABLATION_RUN_DIR",
    "",
).strip()
if RUN_DIR_RAW:
    RUN_DIR = Path(RUN_DIR_RAW).expanduser().resolve()
else:
    candidate_runs = sorted(
        path
        for path in ABLATION_ROOT.glob("ablation_*")
        if path.is_dir()
    )
    if not candidate_runs:
        raise FileNotFoundError(
            f"No ablation runs found under {ABLATION_ROOT}"
        )
    RUN_DIR = candidate_runs[-1]
PAPER_DIR = RUN_DIR / "paper_outputs"
TABLE_DIR = PAPER_DIR / "tables"
FIGURE_DIR = PAPER_DIR / "figures"
TEXT_DIR = PAPER_DIR / "manuscript_text"
QUALITATIVE_DIR = PAPER_DIR / "qualitative"
for directory in [
    PAPER_DIR,
    TABLE_DIR,
    FIGURE_DIR,
    TEXT_DIR,
    QUALITATIVE_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
EXPERIMENTS = [
    "full",
    "optical_only",
    "sar_only",
    "vision_only",
    "grounding_only",
]
DISPLAY_NAMES = {
    "full": "Full",
    "optical_only": "Optical Only",
    "sar_only": "SAR Only",
    "vision_only": "Vision Only",
    "grounding_only": "Grounding Only",
}
LONG_NAMES = {
    "full": "Full: Sentinel-2 + Sentinel-1 + Grounding",
    "optical_only": "Optical Only: Sentinel-2 + Grounding",
    "sar_only": "SAR Only: Sentinel-1 + Grounding",
    "vision_only": "Vision Only: Sentinel-2 + Sentinel-1",
    "grounding_only": "Grounding Only",
}
CORE_METRICS = [
    "bleu",
    "rouge1_f1",
    "rouge2_f1",
    "rougeL_f1",
    "token_f1",
    "numeric_f1",
    "json_value_accuracy",
]
RECORD_METRICS = [
    "rouge1_f1",
    "rouge2_f1",
    "rougeL_f1",
    "token_f1",
    "numeric_f1",
    "json_value_accuracy",
]
BOOTSTRAP_ITERATIONS = int(
    os.environ.get(
        "ROADFLOOD_BOOTSTRAP_ITERATIONS",
        "10000",
    )
)
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
print("=" * 90)
print("ROADFLOOD-VLM PAPER RESULTS GENERATOR")
print("=" * 90)
print(f"Run directory : {RUN_DIR}")
print(f"Paper outputs : {PAPER_DIR}")
print(f"Bootstrap runs: {BOOTSTRAP_ITERATIONS:,}")
# =============================================================================
# 2. LOAD OVERALL SUMMARY
# =============================================================================
SUMMARY_CSV = RUN_DIR / "ablation_summary.csv"
if not SUMMARY_CSV.exists():
    raise FileNotFoundError(SUMMARY_CSV)
summary_df = pd.read_csv(SUMMARY_CSV)
required_summary_columns = {
    "configuration",
    "sentinel_2",
    "sentinel_1",
    "transportation_grounding",
    "records",
    "bleu",
    "rouge1_f1",
    "rouge2_f1",
    "rougeL_f1",
    "token_f1",
    "numeric_f1",
    "json_value_accuracy",
}
missing_summary_columns = (
    required_summary_columns
    - set(summary_df.columns)
)
if missing_summary_columns:
    raise KeyError(
        "Ablation summary is missing columns: "
        + ", ".join(sorted(missing_summary_columns))
    )
summary_df = (
    summary_df
    .set_index("configuration")
    .reindex(EXPERIMENTS)
    .reset_index()
)
summary_df["configuration_label"] = (
    summary_df["configuration"]
    .map(DISPLAY_NAMES)
)
summary_df["configuration_long"] = (
    summary_df["configuration"]
    .map(LONG_NAMES)
)
# =============================================================================
# 3. LOAD RECORD-LEVEL RESULTS
# =============================================================================
def load_experiment_predictions(
    experiment: str,
) -> pd.DataFrame:
    extended_path = (
        RUN_DIR
        / experiment
        / "predictions_extended_metrics.csv"
    )
    basic_path = (
        RUN_DIR
        / experiment
        / "predictions.csv"
    )
    if extended_path.exists():
        path = extended_path
    elif basic_path.exists():
        path = basic_path
    else:
        raise FileNotFoundError(
            f"No prediction file found for {experiment}"
        )
    dataframe = pd.read_csv(path)
    required = {
        "instruction_id",
        "scene_id",
        "reference",
        "prediction",
        "token_f1",
    }
    missing = required - set(dataframe.columns)
    if missing:
        raise KeyError(
            f"{path} is missing columns: {sorted(missing)}"
        )
    if "task_family" not in dataframe.columns:
        dataframe["task_family"] = "unknown"
    dataframe["experiment"] = experiment
    dataframe["configuration_label"] = DISPLAY_NAMES[
        experiment
    ]
    return dataframe
prediction_frames = {
    experiment: load_experiment_predictions(experiment)
    for experiment in EXPERIMENTS
}
record_counts = {
    experiment: len(dataframe)
    for experiment, dataframe
    in prediction_frames.items()
}
print("\nRecord counts")
print("-" * 90)
for experiment in EXPERIMENTS:
    print(
        f"{DISPLAY_NAMES[experiment]:<18}: "
        f"{record_counts[experiment]}"
    )
# =============================================================================
# 4. PUBLICATION-READY OVERALL TABLE
# =============================================================================
paper_columns = [
    "configuration",
    "configuration_label",
    "sentinel_2",
    "sentinel_1",
    "transportation_grounding",
    "records",
    "bleu",
    "rouge1_f1",
    "rouge2_f1",
    "rougeL_f1",
    "token_f1",
    "numeric_f1",
    "json_value_accuracy",
]
paper_table_df = summary_df[paper_columns].copy()
baseline = (
    paper_table_df
    .set_index("configuration")
    .loc["full"]
)
for metric in CORE_METRICS:
    baseline_value = baseline[metric]
    paper_table_df[
        f"{metric}_absolute_change"
    ] = (
        paper_table_df[metric]
        - baseline_value
    )
    paper_table_df[
        f"{metric}_percent_change"
    ] = np.where(
        baseline_value != 0,
        100.0
        * (
            paper_table_df[metric]
            - baseline_value
        )
        / baseline_value,
        np.nan,
    )
paper_table_csv = (
    TABLE_DIR
    / "table_ablation_overall.csv"
)
paper_table_df.to_csv(
    paper_table_csv,
    index=False,
)
# Main compact manuscript table
main_table_df = paper_table_df[
    [
        "configuration_label",
        "bleu",
        "rougeL_f1",
        "token_f1",
        "numeric_f1",
        "json_value_accuracy",
    ]
].copy()
main_table_df.columns = [
    "Configuration",
    "BLEU",
    "ROUGE-L",
    "Token F1",
    "Numeric F1",
    "JSON Value Accuracy",
]
main_table_csv = (
    TABLE_DIR
    / "table_ablation_main.csv"
)
main_table_df.to_csv(
    main_table_csv,
    index=False,
)
def dataframe_to_latex(
    dataframe: pd.DataFrame,
) -> str:
    formatted = dataframe.copy()
    numeric_columns = formatted.select_dtypes(
        include=[np.number]
    ).columns
    for column in numeric_columns:
        formatted[column] = formatted[column].map(
            lambda value: (
                f"{value:.3f}"
                if pd.notna(value)
                else "--"
            )
        )
    return formatted.to_latex(
        index=False,
        escape=True,
        caption=(
            "Inference-time input ablation results for "
            "RoadFlood-VLM on the held-out test records."
        ),
        label="tab:roadflood_ablation",
        position="htbp",
    )
latex_path = (
    TABLE_DIR
    / "table_ablation_main.tex"
)
latex_path.write_text(
    dataframe_to_latex(main_table_df),
    encoding="utf-8",
)
# =============================================================================
# 5. TASK-FAMILY RESULTS
# =============================================================================
task_family_rows = []
for experiment, dataframe in prediction_frames.items():
    available_metrics = [
        metric
        for metric in RECORD_METRICS
        if metric in dataframe.columns
    ]
    aggregation = {
        "instruction_id": "count",
    }
    aggregation.update(
        {
            metric: "mean"
            for metric in available_metrics
        }
    )
    grouped = (
        dataframe
        .groupby(
            "task_family",
            dropna=False,
        )
        .agg(aggregation)
        .reset_index()
        .rename(
            columns={
                "instruction_id": "records",
            }
        )
    )
    grouped.insert(
        0,
        "configuration",
        experiment,
    )
    grouped.insert(
        1,
        "configuration_label",
        DISPLAY_NAMES[experiment],
    )
    task_family_rows.append(grouped)
task_family_df = pd.concat(
    task_family_rows,
    ignore_index=True,
)
task_family_csv = (
    TABLE_DIR
    / "table_ablation_by_task_family.csv"
)
task_family_df.to_csv(
    task_family_csv,
    index=False,
)
for metric in [
    "token_f1",
    "rougeL_f1",
    "numeric_f1",
]:
    if metric not in task_family_df.columns:
        continue
    wide = task_family_df.pivot_table(
        index="task_family",
        columns="configuration_label",
        values=metric,
    )
    wide = wide.reindex(
        columns=[
            DISPLAY_NAMES[experiment]
            for experiment in EXPERIMENTS
        ]
    )
    wide.to_csv(
        TABLE_DIR
        / f"table_task_family_{metric}_wide.csv"
    )
# =============================================================================
# 6. PAIRED BOOTSTRAP, WILCOXON, AND EFFECT SIZE
# =============================================================================
full_df = prediction_frames["full"].copy()
statistics_rows = []
for experiment in EXPERIMENTS:
    if experiment == "full":
        continue
    comparison_df = (
        prediction_frames[experiment]
        .copy()
    )
    merged = full_df.merge(
        comparison_df,
        on="instruction_id",
        suffixes=(
            "_full",
            "_ablation",
        ),
        validate="one_to_one",
    )
    for metric in RECORD_METRICS:
        full_column = f"{metric}_full"
        ablation_column = f"{metric}_ablation"
        if (
            full_column not in merged.columns
            or ablation_column not in merged.columns
        ):
            continue
        full_values = pd.to_numeric(
            merged[full_column],
            errors="coerce",
        )
        ablation_values = pd.to_numeric(
            merged[ablation_column],
            errors="coerce",
        )
        valid = (
            full_values.notna()
            & ablation_values.notna()
        )
        full_values = (
            full_values[valid]
            .to_numpy(dtype=float)
        )
        ablation_values = (
            ablation_values[valid]
            .to_numpy(dtype=float)
        )
        differences = (
            ablation_values
            - full_values
        )
        sample_count = len(differences)
        if sample_count == 0:
            continue
        sampled_indices = rng.integers(
            0,
            sample_count,
            size=(
                BOOTSTRAP_ITERATIONS,
                sample_count,
            ),
        )
        bootstrap_means = differences[
            sampled_indices
        ].mean(axis=1)
        mean_difference = float(
            differences.mean()
        )
        lower_ci = float(
            np.quantile(
                bootstrap_means,
                0.025,
            )
        )
        upper_ci = float(
            np.quantile(
                bootstrap_means,
                0.975,
            )
        )
        probability_better = float(
            (
                bootstrap_means
                > 0
            ).mean()
        )
        difference_std = float(
            differences.std(ddof=1)
        ) if sample_count > 1 else np.nan
        cohens_dz = (
            mean_difference / difference_std
            if (
                pd.notna(difference_std)
                and difference_std > 0
            )
            else np.nan
        )
        wilcoxon_statistic = np.nan
        wilcoxon_p_value = np.nan
        if (
            wilcoxon is not None
            and sample_count >= 2
            and not np.allclose(
                differences,
                0,
            )
        ):
            try:
                test_result = wilcoxon(
                    ablation_values,
                    full_values,
                    zero_method="wilcox",
                    alternative="two-sided",
                )
                wilcoxon_statistic = float(
                    test_result.statistic
                )
                wilcoxon_p_value = float(
                    test_result.pvalue
                )
            except ValueError:
                pass
        statistics_rows.append(
            {
                "configuration": experiment,
                "configuration_label": (
                    DISPLAY_NAMES[experiment]
                ),
                "metric": metric,
                "paired_records": sample_count,
                "full_mean": float(
                    full_values.mean()
                ),
                "ablation_mean": float(
                    ablation_values.mean()
                ),
                "mean_difference": (
                    mean_difference
                ),
                "ci_95_lower": lower_ci,
                "ci_95_upper": upper_ci,
                "bootstrap_probability_ablation_better": (
                    probability_better
                ),
                "cohens_dz": cohens_dz,
                "wilcoxon_statistic": (
                    wilcoxon_statistic
                ),
                "wilcoxon_p_value": (
                    wilcoxon_p_value
                ),
                "ci_excludes_zero": bool(
                    lower_ci > 0
                    or upper_ci < 0
                ),
            }
        )
statistics_df = pd.DataFrame(
    statistics_rows
)
statistics_csv = (
    TABLE_DIR
    / "table_paired_statistics.csv"
)
statistics_df.to_csv(
    statistics_csv,
    index=False,
)
# =============================================================================
# 7. QUALITATIVE COMPARISON TABLE
# =============================================================================
qualitative = full_df[
    [
        "instruction_id",
        "scene_id",
        "task_family",
        "prompt",
        "reference",
        "prediction",
        "token_f1",
    ]
].copy()
qualitative = qualitative.rename(
    columns={
        "prediction": "prediction_full",
        "token_f1": "token_f1_full",
    }
)
for experiment in EXPERIMENTS:
    if experiment == "full":
        continue
    current = prediction_frames[experiment][
        [
            "instruction_id",
            "prediction",
            "token_f1",
        ]
    ].copy()
    current = current.rename(
        columns={
            "prediction": (
                f"prediction_{experiment}"
            ),
            "token_f1": (
                f"token_f1_{experiment}"
            ),
        }
    )
    qualitative = qualitative.merge(
        current,
        on="instruction_id",
        how="left",
        validate="one_to_one",
    )
qualitative["vision_drop_from_full"] = (
    qualitative["token_f1_vision_only"]
    - qualitative["token_f1_full"]
)
qualitative["sar_change_from_full"] = (
    qualitative["token_f1_sar_only"]
    - qualitative["token_f1_full"]
)
qualitative["grounding_change_from_full"] = (
    qualitative["token_f1_grounding_only"]
    - qualitative["token_f1_full"]
)
all_qualitative_csv = (
    QUALITATIVE_DIR
    / "qualitative_all_records.csv"
)
qualitative.to_csv(
    all_qualitative_csv,
    index=False,
)
largest_vision_drops = (
    qualitative
    .nsmallest(
        min(10, len(qualitative)),
        "vision_drop_from_full",
    )
)
largest_vision_drops.to_csv(
    QUALITATIVE_DIR
    / "qualitative_largest_vision_drops.csv",
    index=False,
)
largest_sar_improvements = (
    qualitative
    .nlargest(
        min(10, len(qualitative)),
        "sar_change_from_full",
    )
)
largest_sar_improvements.to_csv(
    QUALITATIVE_DIR
    / "qualitative_largest_sar_improvements.csv",
    index=False,
)
best_full = qualitative.nlargest(
    min(10, len(qualitative)),
    "token_f1_full",
)
best_full.to_csv(
    QUALITATIVE_DIR
    / "qualitative_best_full_predictions.csv",
    index=False,
)
worst_full = qualitative.nsmallest(
    min(10, len(qualitative)),
    "token_f1_full",
)
worst_full.to_csv(
    QUALITATIVE_DIR
    / "qualitative_worst_full_predictions.csv",
    index=False,
)
# =============================================================================
# 8. VISUALIZATION 1 — MAIN GROUPED BAR CHART
# =============================================================================
plot_metrics = [
    "bleu",
    "rougeL_f1",
    "token_f1",
    "numeric_f1",
    "json_value_accuracy",
]
plot_labels = {
    "bleu": "BLEU",
    "rougeL_f1": "ROUGE-L",
    "token_f1": "Token F1",
    "numeric_f1": "Numeric F1",
    "json_value_accuracy": "JSON Value Accuracy",
}
plot_df = (
    summary_df
    .set_index("configuration_label")
    [plot_metrics]
)
plot_df.columns = [
    plot_labels[column]
    for column in plot_df.columns
]
figure, axis = plt.subplots(
    figsize=(12, 6.8)
)
plot_df.plot(
    kind="bar",
    ax=axis,
    width=0.82,
)
axis.set_title(
    "RoadFlood-VLM Inference-Time Input Ablation"
)
axis.set_xlabel(
    "Input configuration"
)
axis.set_ylabel(
    "Evaluation score"
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
main_figure_path = (
    FIGURE_DIR
    / "figure_ablation_grouped_metrics.png"
)
figure.savefig(
    main_figure_path,
    dpi=400,
    bbox_inches="tight",
)
plt.close(figure)
# =============================================================================
# 9. VISUALIZATION 2 — CHANGE FROM FULL
# =============================================================================
change_metrics = [
    "bleu",
    "rougeL_f1",
    "token_f1",
    "numeric_f1",
]
change_labels = [
    "BLEU",
    "ROUGE-L",
    "Token F1",
    "Numeric F1",
]
change_rows = []
for _, row in paper_table_df.iterrows():
    if row["configuration"] == "full":
        continue
    result = {
        "configuration": (
            row["configuration_label"]
        )
    }
    for metric, label in zip(
        change_metrics,
        change_labels,
    ):
        result[label] = row[
            f"{metric}_percent_change"
        ]
    change_rows.append(result)
change_df = (
    pd.DataFrame(change_rows)
    .set_index("configuration")
)
figure, axis = plt.subplots(
    figsize=(11, 6.5)
)
change_df.plot(
    kind="bar",
    ax=axis,
    width=0.82,
)
axis.axhline(
    0,
    linewidth=1,
)
axis.set_title(
    "Performance Change Relative to Full Inputs"
)
axis.set_xlabel(
    "Ablation configuration"
)
axis.set_ylabel(
    "Change from full configuration (%)"
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
change_figure_path = (
    FIGURE_DIR
    / "figure_percent_change_from_full.png"
)
figure.savefig(
    change_figure_path,
    dpi=400,
    bbox_inches="tight",
)
plt.close(figure)
# =============================================================================
# 10. VISUALIZATION 3 — METRIC HEATMAP
# =============================================================================
heatmap_metrics = [
    "bleu",
    "rouge1_f1",
    "rouge2_f1",
    "rougeL_f1",
    "token_f1",
    "numeric_f1",
    "json_value_accuracy",
]
heatmap_labels = [
    "BLEU",
    "ROUGE-1",
    "ROUGE-2",
    "ROUGE-L",
    "Token F1",
    "Numeric F1",
    "JSON Value",
]
heatmap_data = (
    summary_df
    .set_index("configuration_label")
    [heatmap_metrics]
)
figure, axis = plt.subplots(
    figsize=(10.5, 5.5)
)
image = axis.imshow(
    heatmap_data.to_numpy(),
    aspect="auto",
)
axis.set_title(
    "RoadFlood-VLM Ablation Metric Heatmap"
)
axis.set_xticks(
    np.arange(len(heatmap_labels))
)
axis.set_xticklabels(
    heatmap_labels,
    rotation=35,
    ha="right",
)
axis.set_yticks(
    np.arange(len(heatmap_data.index))
)
axis.set_yticklabels(
    heatmap_data.index,
)
for row_index in range(
    heatmap_data.shape[0]
):
    for column_index in range(
        heatmap_data.shape[1]
    ):
        value = heatmap_data.iloc[
            row_index,
            column_index,
        ]
        axis.text(
            column_index,
            row_index,
            (
                f"{value:.3f}"
                if pd.notna(value)
                else "--"
            ),
            ha="center",
            va="center",
        )
figure.colorbar(
    image,
    ax=axis,
    label="Score",
)
figure.tight_layout()
heatmap_path = (
    FIGURE_DIR
    / "figure_ablation_heatmap.png"
)
figure.savefig(
    heatmap_path,
    dpi=400,
    bbox_inches="tight",
)
plt.close(figure)
# =============================================================================
# 11. VISUALIZATION 4 — TRANSPORTATION GROUNDING IMPACT
# =============================================================================
grounding_comparison = (
    summary_df[
        summary_df["configuration"].isin(
            [
                "full",
                "vision_only",
                "grounding_only",
            ]
        )
    ]
    .set_index("configuration_label")
    [
        [
            "bleu",
            "rougeL_f1",
            "token_f1",
            "numeric_f1",
        ]
    ]
)
grounding_comparison.columns = [
    "BLEU",
    "ROUGE-L",
    "Token F1",
    "Numeric F1",
]
figure, axis = plt.subplots(
    figsize=(9.5, 6)
)
grounding_comparison.plot(
    kind="bar",
    ax=axis,
    width=0.78,
)
axis.set_title(
    "Effect of Transportation Grounding"
)
axis.set_xlabel(
    "Input configuration"
)
axis.set_ylabel(
    "Evaluation score"
)
axis.set_ylim(
    0,
    1,
)
axis.tick_params(
    axis="x",
    rotation=15,
)
axis.grid(
    axis="y",
    alpha=0.25,
)
axis.legend(
    title="Metric",
)
figure.tight_layout()
grounding_figure_path = (
    FIGURE_DIR
    / "figure_transportation_grounding_impact.png"
)
figure.savefig(
    grounding_figure_path,
    dpi=400,
    bbox_inches="tight",
)
plt.close(figure)
# =============================================================================
# 12. VISUALIZATION 5 — TASK-FAMILY HEATMAPS
# =============================================================================
task_heatmap_paths = []
for metric in [
    "token_f1",
    "rougeL_f1",
    "numeric_f1",
]:
    if metric not in task_family_df.columns:
        continue
    task_matrix = task_family_df.pivot_table(
        index="task_family",
        columns="configuration_label",
        values=metric,
    )
    ordered_columns = [
        DISPLAY_NAMES[experiment]
        for experiment in EXPERIMENTS
        if DISPLAY_NAMES[experiment]
        in task_matrix.columns
    ]
    task_matrix = task_matrix.reindex(
        columns=ordered_columns
    )
    figure_height = max(
        4.5,
        0.55 * len(task_matrix.index) + 2,
    )
    figure, axis = plt.subplots(
        figsize=(9.5, figure_height)
    )
    image = axis.imshow(
        task_matrix.to_numpy(),
        aspect="auto",
    )
    axis.set_title(
        f"{metric.replace('_', ' ').title()} "
        "by Task Family"
    )
    axis.set_xticks(
        np.arange(
            len(task_matrix.columns)
        )
    )
    axis.set_xticklabels(
        task_matrix.columns,
        rotation=25,
        ha="right",
    )
    axis.set_yticks(
        np.arange(
            len(task_matrix.index)
        )
    )
    axis.set_yticklabels(
        task_matrix.index,
    )
    for row_index in range(
        task_matrix.shape[0]
    ):
        for column_index in range(
            task_matrix.shape[1]
        ):
            value = task_matrix.iloc[
                row_index,
                column_index,
            ]
            axis.text(
                column_index,
                row_index,
                (
                    f"{value:.3f}"
                    if pd.notna(value)
                    else "--"
                ),
                ha="center",
                va="center",
            )
    figure.colorbar(
        image,
        ax=axis,
        label="Score",
    )
    figure.tight_layout()
    output_path = (
        FIGURE_DIR
        / f"figure_task_family_{metric}.png"
    )
    figure.savefig(
        output_path,
        dpi=400,
        bbox_inches="tight",
    )
    plt.close(figure)
    task_heatmap_paths.append(
        str(output_path)
    )
# =============================================================================
# 13. MANUSCRIPT RESULTS TEXT
# =============================================================================
def metric_value(
    configuration: str,
    metric: str,
) -> float:
    row = summary_df.loc[
        summary_df["configuration"]
        == configuration
    ]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][metric])
full_bleu = metric_value(
    "full",
    "bleu",
)
full_rouge_l = metric_value(
    "full",
    "rougeL_f1",
)
full_token_f1 = metric_value(
    "full",
    "token_f1",
)
full_numeric_f1 = metric_value(
    "full",
    "numeric_f1",
)
vision_bleu = metric_value(
    "vision_only",
    "bleu",
)
vision_rouge_l = metric_value(
    "vision_only",
    "rougeL_f1",
)
vision_token_f1 = metric_value(
    "vision_only",
    "token_f1",
)
vision_numeric_f1 = metric_value(
    "vision_only",
    "numeric_f1",
)
sar_bleu = metric_value(
    "sar_only",
    "bleu",
)
sar_token_f1 = metric_value(
    "sar_only",
    "token_f1",
)
sar_numeric_f1 = metric_value(
    "sar_only",
    "numeric_f1",
)
grounding_bleu = metric_value(
    "grounding_only",
    "bleu",
)
grounding_token_f1 = metric_value(
    "grounding_only",
    "token_f1",
)
grounding_json = metric_value(
    "grounding_only",
    "json_value_accuracy",
)
def percent_change(
    new_value: float,
    baseline_value: float,
) -> float:
    if baseline_value == 0:
        return float("nan")
    return (
        100.0
        * (
            new_value
            - baseline_value
        )
        / baseline_value
    )
results_text = f"""
## Ablation Results
An inference-time input ablation analysis was conducted using the same
fine-tuned RoadFlood-VLM adapter and the same {int(summary_df["records"].max())}
held-out instruction records. Model weights were held fixed while Sentinel-2
imagery, Sentinel-1 imagery, or transportation-grounding context was selectively
withheld.
The complete input configuration achieved a BLEU score of {full_bleu:.3f},
ROUGE-L F1 of {full_rouge_l:.3f}, Token F1 of {full_token_f1:.3f}, and
Numeric F1 of {full_numeric_f1:.3f}. The SAR-only configuration remained
comparable to the full configuration, obtaining BLEU={sar_bleu:.3f},
Token F1={sar_token_f1:.3f}, and Numeric F1={sar_numeric_f1:.3f}.
Because the test set is small, these modest improvements should be interpreted
using the paired bootstrap confidence intervals rather than as definitive
evidence that the SAR-only configuration is superior.
Removing transportation grounding produced the largest degradation.
The vision-only configuration reduced BLEU from {full_bleu:.3f} to
{vision_bleu:.3f} ({percent_change(vision_bleu, full_bleu):.1f}%),
ROUGE-L F1 from {full_rouge_l:.3f} to {vision_rouge_l:.3f}
({percent_change(vision_rouge_l, full_rouge_l):.1f}%), and Token F1 from
{full_token_f1:.3f} to {vision_token_f1:.3f}
({percent_change(vision_token_f1, full_token_f1):.1f}%).
Numeric F1 also decreased from {full_numeric_f1:.3f} to
{vision_numeric_f1:.3f}
({percent_change(vision_numeric_f1, full_numeric_f1):.1f}%).
The grounding-only configuration achieved BLEU={grounding_bleu:.3f},
Token F1={grounding_token_f1:.3f}, and JSON value accuracy={grounding_json:.3f}.
Its performance remained close to the full input configuration, demonstrating
that structured transportation-grounding information was the dominant source
of scene-specific numerical and narrative content on this benchmark.
""".strip()
results_path = (
    TEXT_DIR
    / "results_ablation.md"
)
results_path.write_text(
    results_text,
    encoding="utf-8",
)
discussion_text = """
## Discussion of the Ablation Analysis
The results demonstrate a substantial dependence on structured transportation
grounding. When grounding was removed and the model received only Sentinel-1
and Sentinel-2 images, language similarity, categorical consistency, and
structured-output accuracy declined sharply. This finding indicates that the
model does not reliably reconstruct exact roadway counts, exposure shares,
network-disruption attributes, or formatted summaries directly from the image
pair.
The optical-only and SAR-only configurations remained close to the full-input
configuration when transportation grounding was retained. This suggests that
the model can tolerate the absence of one image modality when structured
scene-level evidence remains available. The relatively strong SAR-only scores
are consistent with the value of radar imagery for identifying flood-related
surface conditions, although the paired statistical analysis should determine
whether the observed differences from the full configuration are distinguishable
from sampling variation.
The grounding-only configuration also remained close to the complete
configuration. This result does not imply that imagery is unnecessary for the
broader RoadFlood-VLM framework. Instead, it shows that the reference responses
in the present benchmark are strongly determined by the structured
transportation attributes included in the prompt. Imagery may still provide
essential visual verification, spatial context, and evidence for cases where
structured attributes are incomplete or unavailable.
""".strip()
discussion_path = (
    TEXT_DIR
    / "discussion_ablation.md"
)
discussion_path.write_text(
    discussion_text,
    encoding="utf-8",
)
limitations_text = """
## Ablation Limitations
This experiment is an inference-time input ablation rather than a comparison
of independently trained modality-specific models. All configurations used
the same adapter that was trained with Sentinel-2 imagery, Sentinel-1 imagery,
and transportation-grounding context. Therefore, the findings quantify
sensitivity to missing inputs under a fixed trained model.
The held-out split contains 24 instruction records rather than 24 independent
scenes. Multiple instructions may originate from the same scene, and the
effective number of geographically independent test cases is therefore
smaller than the record count. The paired bootstrap analysis operates at the
instruction-record level and should be interpreted accordingly.
The grounded configurations include supplied scene-level labels and
transportation attributes. Their flood-burden and disruption consistency
scores measure conditioned response generation rather than independent
classification from imagery. A label-hidden evaluation is required to measure
independent category inference.
The reference responses were produced from deterministic grounded templates.
Consequently, overlap-based language metrics may reward reproduction of the
reference style and wording. Numeric F1, structured-value accuracy, qualitative
error analysis, and independent expert review should accompany BLEU and ROUGE
when evaluating factual transportation reasoning.
""".strip()
limitations_path = (
    TEXT_DIR
    / "limitations_ablation.md"
)
limitations_path.write_text(
    limitations_text,
    encoding="utf-8",
)
methods_text = """
## Ablation Method
An inference-time input ablation study was performed using one fixed
fine-tuned RoadFlood-VLM adapter. Five configurations were evaluated:
(1) the complete Sentinel-2, Sentinel-1, and transportation-grounding input;
(2) Sentinel-2 with transportation grounding; (3) Sentinel-1 with
transportation grounding; (4) Sentinel-2 and Sentinel-1 without transportation
grounding; and (5) transportation grounding without imagery. The model was
not retrained between configurations.
Each configuration was evaluated on the same held-out instruction records.
Generation was deterministic, and the same maximum generation length was
used across configurations. Performance was evaluated using BLEU, ROUGE,
Token F1, Numeric F1, JSON validity, JSON key recall, and JSON value accuracy.
Paired bootstrap confidence intervals were computed from record-level score
differences relative to the full-input configuration. Wilcoxon signed-rank
tests and paired standardized effect sizes were also calculated where valid.
""".strip()
methods_path = (
    TEXT_DIR
    / "methods_ablation.md"
)
methods_path.write_text(
    methods_text,
    encoding="utf-8",
)
# =============================================================================
# 14. OUTPUT MANIFEST
# =============================================================================
manifest = {
    "source_run": str(RUN_DIR),
    "paper_output_directory": str(PAPER_DIR),
    "records_by_configuration": record_counts,
    "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
    "random_seed": RANDOM_SEED,
    "scipy_available": wilcoxon is not None,
    "tables": {
        "main_table_csv": str(main_table_csv),
        "main_table_latex": str(latex_path),
        "overall_detailed": str(paper_table_csv),
        "task_family": str(task_family_csv),
        "paired_statistics": str(statistics_csv),
    },
    "figures": {
        "grouped_metrics": str(main_figure_path),
        "percent_change": str(change_figure_path),
        "metric_heatmap": str(heatmap_path),
        "grounding_impact": str(grounding_figure_path),
        "task_family_heatmaps": task_heatmap_paths,
    },
    "qualitative": {
        "all_records": str(all_qualitative_csv),
        "largest_vision_drops": str(
            QUALITATIVE_DIR
            / "qualitative_largest_vision_drops.csv"
        ),
        "largest_sar_improvements": str(
            QUALITATIVE_DIR
            / "qualitative_largest_sar_improvements.csv"
        ),
        "best_full_predictions": str(
            QUALITATIVE_DIR
            / "qualitative_best_full_predictions.csv"
        ),
        "worst_full_predictions": str(
            QUALITATIVE_DIR
            / "qualitative_worst_full_predictions.csv"
        ),
    },
    "manuscript_text": {
        "methods": str(methods_path),
        "results": str(results_path),
        "discussion": str(discussion_path),
        "limitations": str(limitations_path),
    },
}
manifest_path = (
    PAPER_DIR
    / "paper_output_manifest.json"
)
manifest_path.write_text(
    json.dumps(
        manifest,
        indent=2,
    ),
    encoding="utf-8",
)
# =============================================================================
# 15. FINAL REPORT
# =============================================================================
print("\n" + "=" * 90)
print("MAIN ABLATION TABLE")
print("=" * 90)
print(
    main_table_df
    .round(4)
    .to_string(index=False)
)
print("\n" + "=" * 90)
print("PAIRED STATISTICAL RESULTS")
print("=" * 90)
if statistics_df.empty:
    print("No paired statistics were available.")
else:
    print(
        statistics_df[
            [
                "configuration_label",
                "metric",
                "paired_records",
                "mean_difference",
                "ci_95_lower",
                "ci_95_upper",
                "wilcoxon_p_value",
                "cohens_dz",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
print("\n" + "=" * 90)
print("GENERATED PAPER OUTPUTS")
print("=" * 90)
for category, outputs in manifest.items():
    if isinstance(outputs, dict):
        print(f"\n{category.upper()}")
        for name, path in outputs.items():
            if isinstance(path, list):
                for item in path:
                    print(f"  {name}: {item}")
            else:
                print(f"  {name}: {path}")
print(f"\nManifest: {manifest_path}")
print("=" * 90)
