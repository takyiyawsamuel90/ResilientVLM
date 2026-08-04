from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
ROOT = Path(
    "data/external/sturm_fusion_24/extracted/Dataset"
).resolve()
METADATA_PATH = ROOT / "metadata" / "metadata.csv"
OUTPUT_PATH = Path(
    "data/processed/sturm_roadflood/"
    "sturm_layout_report.json"
).resolve()
if not ROOT.exists():
    raise FileNotFoundError(ROOT)
if not METADATA_PATH.exists():
    raise FileNotFoundError(METADATA_PATH)
metadata = pd.read_csv(METADATA_PATH)
required_columns = {
    "ems_code",
    "aoi_code",
    "floodmap_id",
    "event_type",
    "country",
    "tile_id",
    "floodmap_date",
    "sentinel2_timestamp",
    "sentinel1_timestamp",
}
missing_columns = required_columns - set(metadata.columns)
if missing_columns:
    raise KeyError(
        "Metadata is missing columns: "
        + ", ".join(sorted(missing_columns))
    )
# ---------------------------------------------------------------------
# Count files by top-level folder
# ---------------------------------------------------------------------
folder_counts = Counter()
extension_counts = Counter()
folder_examples: dict[str, list[str]] = {}
for path in ROOT.rglob("*"):
    if not path.is_file():
        continue
    relative = path.relative_to(ROOT)
    top_folder = relative.parts[0]
    folder_counts[top_folder] += 1
    extension_counts[path.suffix.lower()] += 1
    folder_examples.setdefault(top_folder, [])
    if len(folder_examples[top_folder]) < 10:
        folder_examples[top_folder].append(str(relative))
# ---------------------------------------------------------------------
# Inspect one or more TIFFs in every top-level folder
# ---------------------------------------------------------------------
raster_reports = []
for folder in sorted(
    path
    for path in ROOT.iterdir()
    if path.is_dir()
):
    tif_files = sorted(folder.rglob("*.tif"))
    for tif_path in tif_files[:3]:
        try:
            with rasterio.open(tif_path) as src:
                array = src.read()
                band_reports = []
                for band_index in range(src.count):
                    band = array[band_index]
                    finite = band[np.isfinite(band)]
                    if finite.size == 0:
                        minimum = None
                        maximum = None
                        mean = None
                        unique_preview = []
                    else:
                        minimum = float(finite.min())
                        maximum = float(finite.max())
                        mean = float(finite.mean())
                        unique_values = np.unique(finite)
                        unique_preview = [
                            float(value)
                            for value in unique_values[:20]
                        ]
                    band_reports.append(
                        {
                            "band": band_index + 1,
                            "minimum": minimum,
                            "maximum": maximum,
                            "mean": mean,
                            "unique_value_count": int(
                                len(np.unique(finite))
                            ) if finite.size else 0,
                            "unique_preview": unique_preview,
                        }
                    )
                raster_reports.append(
                    {
                        "folder": folder.name,
                        "path": str(tif_path.relative_to(ROOT)),
                        "bands": src.count,
                        "height": src.height,
                        "width": src.width,
                        "dtypes": list(src.dtypes),
                        "crs": str(src.crs),
                        "nodata": src.nodata,
                        "band_statistics": band_reports,
                    }
                )
        except Exception as error:
            raster_reports.append(
                {
                    "folder": folder.name,
                    "path": str(tif_path.relative_to(ROOT)),
                    "error": repr(error),
                }
            )
# ---------------------------------------------------------------------
# Verify whether metadata tile IDs occur in each folder
# ---------------------------------------------------------------------
tile_ids = metadata["tile_id"].astype(str).tolist()
folder_matching = {}
for folder in sorted(
    path
    for path in ROOT.iterdir()
    if path.is_dir()
):
    existing_names = {
        path.name
        for path in folder.rglob("*.tif")
    }
    matched = sum(
        tile_id in existing_names
        for tile_id in tile_ids
    )
    folder_matching[folder.name] = {
        "metadata_tiles_matched": int(matched),
        "metadata_tile_count": int(len(tile_ids)),
        "match_rate": float(
            matched / len(tile_ids)
        ) if tile_ids else 0.0,
    }
report = {
    "root": str(ROOT),
    "metadata_path": str(METADATA_PATH),
    "metadata_rows": int(len(metadata)),
    "metadata_columns": metadata.columns.tolist(),
    "unique_events": int(metadata["ems_code"].nunique()),
    "unique_aois": int(
        metadata[["ems_code", "aoi_code"]]
        .drop_duplicates()
        .shape[0]
    ),
    "countries": (
        metadata["country"]
        .value_counts()
        .to_dict()
    ),
    "event_types": (
        metadata["event_type"]
        .value_counts()
        .to_dict()
    ),
    "folder_counts": dict(folder_counts),
    "extension_counts": dict(extension_counts),
    "folder_examples": folder_examples,
    "folder_metadata_matching": folder_matching,
    "raster_reports": raster_reports,
}
OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)
OUTPUT_PATH.write_text(
    json.dumps(
        report,
        indent=2,
    ),
    encoding="utf-8",
)
print("=" * 100)
print("STURM-FUSION-24 LAYOUT INSPECTION")
print("=" * 100)
print(f"Metadata rows : {len(metadata):,}")
print(f"Unique events : {metadata['ems_code'].nunique():,}")
print(
    "Unique AOIs   : "
    f"{metadata[['ems_code', 'aoi_code']].drop_duplicates().shape[0]:,}"
)
print("\nTOP-LEVEL FOLDER COUNTS")
print("-" * 100)
for folder, count in sorted(folder_counts.items()):
    print(f"{folder:<35} {count:>10,}")
print("\nMETADATA TILE MATCHING")
print("-" * 100)
for folder, values in folder_matching.items():
    print(
        f"{folder:<35} "
        f"{values['metadata_tiles_matched']:>6,}/"
        f"{values['metadata_tile_count']:<6,} "
        f"({100 * values['match_rate']:.1f}%)"
    )
print("\nRASTER EXAMPLES")
print("-" * 100)
for raster in raster_reports:
    print(f"\nFolder : {raster['folder']}")
    print(f"File   : {raster['path']}")
    if "error" in raster:
        print(f"Error  : {raster['error']}")
        continue
    print(
        f"Shape  : {raster['bands']} bands × "
        f"{raster['height']} × {raster['width']}"
    )
    print(f"Dtypes : {raster['dtypes']}")
    print(f"Nodata : {raster['nodata']}")
    for band in raster["band_statistics"]:
        print(
            f"  Band {band['band']}: "
            f"min={band['minimum']}, "
            f"max={band['maximum']}, "
            f"mean={band['mean']}, "
            f"unique={band['unique_value_count']}, "
            f"preview={band['unique_preview'][:8]}"
        )
print("\nSaved report:")
print(OUTPUT_PATH)
print("=" * 100)
