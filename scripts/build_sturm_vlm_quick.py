from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm
PROJECT_ROOT = Path.cwd().resolve()
SOURCE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "sturm_fusion_24"
    / "extracted"
    / "Dataset"
)
METADATA_PATH = (
    SOURCE_ROOT
    / "metadata"
    / "metadata.csv"
)
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sturm_vlm_quick"
)
S1_PNG_DIR = OUTPUT_ROOT / "images" / "s1"
S2_PNG_DIR = OUTPUT_ROOT / "images" / "s2"
MASK_PNG_DIR = OUTPUT_ROOT / "images" / "masks"
SPLIT_DIR = OUTPUT_ROOT / "splits"
for directory in [
    S1_PNG_DIR,
    S2_PNG_DIR,
    MASK_PNG_DIR,
    SPLIT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
def robust_scale(
    band: np.ndarray,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
) -> np.ndarray:
    band = np.asarray(band, dtype=np.float32)
    finite = np.isfinite(band)
    if not finite.any():
        return np.zeros(band.shape, dtype=np.uint8)
    values = band[finite]
    lower = float(
        np.percentile(values, lower_percentile)
    )
    upper = float(
        np.percentile(values, upper_percentile)
    )
    if upper <= lower:
        upper = lower + 1e-6
    scaled = (band - lower) / (upper - lower)
    scaled = np.clip(scaled, 0.0, 1.0)
    scaled[~finite] = 0.0
    return np.round(scaled * 255).astype(np.uint8)
def read_raster(path: Path) -> np.ndarray:
    with rasterio.open(path) as source:
        return source.read()
def make_s1_rgb(array: np.ndarray) -> np.ndarray:
    if array.shape[0] < 2:
        raise ValueError(
            f"Expected two S1 bands, received {array.shape}"
        )
    vv = array[0]
    vh = array[1]
    red = robust_scale(vv)
    green = robust_scale(vh)
    blue = robust_scale(vv - vh)
    return np.stack(
        [red, green, blue],
        axis=-1,
    )
def make_s2_rgb(array: np.ndarray) -> np.ndarray:
    if array.shape[0] < 3:
        raise ValueError(
            f"Expected at least three S2 bands, received {array.shape}"
        )
    # The first three supplied STURM optical channels are
    # interpreted as blue, green, and red. Reverse them to RGB.
    red = robust_scale(array[2])
    green = robust_scale(array[1])
    blue = robust_scale(array[0])
    return np.stack(
        [red, green, blue],
        axis=-1,
    )
def process_mask(
    array: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    mask = np.asarray(array[0])
    # Dataset documentation describes a binary flood mask:
    # 0 = non-flood, 1 = flood.
    #
    # Some files also contain 2, and nodata is 99.
    # We exclude 2 and 99 rather than silently treating them as flood.
    valid = np.isin(mask, [0, 1])
    flooded = mask == 1
    valid_pixels = int(valid.sum())
    flood_pixels = int((flooded & valid).sum())
    flood_share = (
        flood_pixels / valid_pixels
        if valid_pixels > 0
        else 0.0
    )
    valid_share = (
        valid_pixels / mask.size
        if mask.size > 0
        else 0.0
    )
    visualization = np.zeros(
        (*mask.shape, 3),
        dtype=np.uint8,
    )
    visualization[mask == 0] = [0, 0, 0]
    visualization[mask == 1] = [255, 255, 255]
    visualization[~valid] = [127, 127, 127]
    return visualization, flood_share, valid_share
def flood_burden(flood_share: float) -> str:
    if flood_share == 0:
        return "None"
    if flood_share <= 0.10:
        return "Low"
    if flood_share <= 0.30:
        return "Moderate"
    return "High"
def make_response(
    burden: str,
    flood_percent: float,
) -> str:
    return json.dumps(
        {
            "scene_flood_burden": burden,
            "flood_area_percent": round(
                flood_percent,
                1,
            ),
        },
        separators=(",", ":"),
    )
metadata = pd.read_csv(METADATA_PATH)
if "Unnamed: 0" in metadata.columns:
    metadata = metadata.drop(
        columns=["Unnamed: 0"]
    )
required_columns = {
    "ems_code",
    "aoi_code",
    "floodmap_id",
    "event_type",
    "country",
    "tile_id",
}
missing = required_columns - set(metadata.columns)
if missing:
    raise KeyError(
        f"Missing metadata columns: {sorted(missing)}"
    )
records: list[dict] = []
failures: list[dict] = []
for row in tqdm(
    metadata.itertuples(index=False),
    total=len(metadata),
    desc="Converting STURM",
):
    tile_id = str(row.tile_id)
    tile_stem = Path(tile_id).stem
    s1_source = SOURCE_ROOT / "S1" / tile_id
    s2_source = SOURCE_ROOT / "S2" / tile_id
    mask_source = SOURCE_ROOT / "floodmaps" / tile_id
    try:
        for path in [
            s1_source,
            s2_source,
            mask_source,
        ]:
            if not path.exists():
                raise FileNotFoundError(path)
        s1_array = read_raster(s1_source)
        s2_array = read_raster(s2_source)
        mask_array = read_raster(mask_source)
        s1_rgb = make_s1_rgb(s1_array)
        s2_rgb = make_s2_rgb(s2_array)
        (
            mask_rgb,
            flood_share,
            valid_share,
        ) = process_mask(mask_array)
        s1_png = S1_PNG_DIR / f"{tile_stem}_s1.png"
        s2_png = S2_PNG_DIR / f"{tile_stem}_s2.png"
        mask_png = MASK_PNG_DIR / f"{tile_stem}_mask.png"
        Image.fromarray(
            s1_rgb,
            mode="RGB",
        ).save(s1_png)
        Image.fromarray(
            s2_rgb,
            mode="RGB",
        ).save(s2_png)
        Image.fromarray(
            mask_rgb,
            mode="RGB",
        ).save(mask_png)
        burden = flood_burden(flood_share)
        flood_percent = 100.0 * flood_share
        records.append(
            {
                "instruction_id": (
                    f"sturm_{tile_stem}"
                ),
                "scene_id": tile_stem,
                "source_dataset": (
                    "STURM-Fusion-24"
                ),
                "ems_code": row.ems_code,
                "aoi_code": row.aoi_code,
                "floodmap_id": row.floodmap_id,
                "event_type": row.event_type,
                "country": row.country,
                "tile_id": tile_id,
                "s2_png_path": str(
                    s2_png.resolve()
                ),
                "s1_png_path": str(
                    s1_png.resolve()
                ),
                "mask_png_path": str(
                    mask_png.resolve()
                ),
                "flood_pixel_share": flood_share,
                "flood_area_percent": flood_percent,
                "valid_mask_share": valid_share,
                "scene_flood_burden": burden,
                "system_prompt": (
                    "You are RoadFlood-VLM, a multimodal "
                    "satellite flood-assessment assistant. "
                    "Use Sentinel-2 optical and Sentinel-1 "
                    "radar imagery to classify visible flood "
                    "burden. Return only valid JSON."
                ),
                "instruction_text": (
                    "Analyze the Sentinel-2 optical image and "
                    "Sentinel-1 radar image. Classify the visible "
                    "scene flood burden as None, Low, Moderate, "
                    "or High and estimate the flooded-area "
                    "percentage. Return only JSON with the keys "
                    "scene_flood_burden and flood_area_percent."
                ),
                "response_text": make_response(
                    burden,
                    flood_percent,
                ),
            }
        )
    except Exception as error:
        failures.append(
            {
                "tile_id": tile_id,
                "error": repr(error),
            }
        )
dataset = pd.DataFrame(records)
if dataset.empty:
    raise RuntimeError(
        "No STURM records were successfully converted."
    )
# Remove tiles with too little valid label coverage.
dataset = dataset.loc[
    dataset["valid_mask_share"] >= 0.80
].reset_index(drop=True)
# -------------------------------------------------------------------
# Event-level split to prevent spatial/event leakage.
# Approximately 70% train, 15% validation, 15% test.
# -------------------------------------------------------------------
groups = dataset["floodmap_id"].astype(str)
outer_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=42,
)
train_indices, remaining_indices = next(
    outer_split.split(
        dataset,
        groups=groups,
    )
)
train_df = dataset.iloc[
    train_indices
].copy()
remaining_df = dataset.iloc[
    remaining_indices
].copy()
inner_groups = (
    remaining_df["floodmap_id"]
    .astype(str)
)
inner_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42,
)
validation_relative, test_relative = next(
    inner_split.split(
        remaining_df,
        groups=inner_groups,
    )
)
validation_df = remaining_df.iloc[
    validation_relative
].copy()
test_df = remaining_df.iloc[
    test_relative
].copy()
for split_name, split_df in [
    ("train", train_df),
    ("validation", validation_df),
    ("test", test_df),
]:
    split_df = split_df.reset_index(drop=True)
    split_df["split"] = split_name
    csv_path = SPLIT_DIR / f"{split_name}.csv"
    jsonl_path = SPLIT_DIR / f"{split_name}.jsonl"
    split_df.to_csv(
        csv_path,
        index=False,
    )
    with jsonl_path.open(
        "w",
        encoding="utf-8",
    ) as output:
        for record in split_df.to_dict(
            orient="records"
        ):
            output.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )
master = pd.concat(
    [
        train_df.assign(split="train"),
        validation_df.assign(split="validation"),
        test_df.assign(split="test"),
    ],
    ignore_index=True,
)
master.to_csv(
    OUTPUT_ROOT / "master.csv",
    index=False,
)
pd.DataFrame(failures).to_csv(
    OUTPUT_ROOT / "conversion_failures.csv",
    index=False,
)
summary = {
    "metadata_rows": int(len(metadata)),
    "converted_records_before_filter": int(
        len(records)
    ),
    "usable_records": int(len(master)),
    "conversion_failures": int(len(failures)),
    "split_records": {
        split: int(
            (master["split"] == split).sum()
        )
        for split in [
            "train",
            "validation",
            "test",
        ]
    },
    "split_events": {
        split: int(
            master.loc[
                master["split"] == split,
                "floodmap_id",
            ].nunique()
        )
        for split in [
            "train",
            "validation",
            "test",
        ]
    },
    "class_counts": (
        master["scene_flood_burden"]
        .value_counts()
        .to_dict()
    ),
    "split_class_counts": (
        pd.crosstab(
            master["split"],
            master["scene_flood_burden"],
        )
        .to_dict()
    ),
}
(
    OUTPUT_ROOT
    / "dataset_summary.json"
).write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)
print("\n" + "=" * 90)
print("STURM QUICK VLM DATASET COMPLETE")
print("=" * 90)
print(json.dumps(summary, indent=2))
print("\nSplit × class:")
print(
    pd.crosstab(
        master["split"],
        master["scene_flood_burden"],
        margins=True,
    )
)
print("\nOutput:")
print(OUTPUT_ROOT)
