# ResilientVLM

> **A vision-language and transportation knowledge framework for understanding flood impacts on roadway networks**

**Developed by Samuel Takyi and Justice Adjei-Owusu**

ResilientVLM is a multimodal transportation resilience project that combines satellite imagery, flood masks, roadway networks, transportation knowledge graphs, and vision-language modeling to identify flood-affected roads and support network-level resilience analysis.

The project integrates Sentinel-1 and Sentinel-2 imagery from the SEN1FLOODS11 dataset with OpenStreetMap roadway data and transportation-specific knowledge representations. Its notebook-based workflow covers data acquisition, scene validation, transportation knowledge graph construction, road-flood grounding, instruction generation, multimodal dataset preparation, model development, evaluation, and ablation testing.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Workflow](#system-workflow)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Notebook Pipeline](#notebook-pipeline)
- [Data Sources](#data-sources)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Running on an HPC System](#running-on-an-hpc-system)
- [Development Notes](#development-notes)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Authors](#authors)
- [License](#license)

---

## Overview

Flooding can disrupt transportation systems by closing roads, isolating communities, increasing travel times, and reducing access to essential services. Traditional flood-mapping models can identify inundated pixels, but they do not always explain how those flood conditions affect transportation infrastructure.

ResilientVLM addresses this gap by combining:

- satellite-based flood observations;
- roadway network geometry;
- road-level flood exposure;
- transportation knowledge graphs;
- structured scene metrics;
- multimodal instruction data;
- vision-language reasoning;
- graph-based resilience analysis.

The project is designed to move beyond simple flood segmentation by grounding model predictions in transportation features such as roads, intersections, connectivity, accessibility, and network disruption.

---

## Key Features

- Sentinel-1 and Sentinel-2 flood-scene processing.
- SEN1FLOODS11 dataset integration.
- Automated scene discovery and validation.
- Recovery workflow for missing imagery downloads.
- Scene quality and eligibility screening.
- OpenStreetMap roadway network extraction.
- Transportation knowledge graph generation.
- GraphML and GeoPackage output support.
- Road-flood spatial overlay and grounding.
- Scene-level roadway disruption metrics.
- Transportation-aware instruction generation.
- Multimodal vision-language dataset construction.
- Vision-language model training workflow.
- Transportation knowledge adapter development.
- Graph-based resilience analysis.
- Evaluation and ablation experiments.
- Notebook execution and recovery utilities.
- Support for local and HPC-based workflows.

---

## System Workflow

```text
SEN1FLOODS11 Scene Inventory
              │
              ▼
Satellite Data Acquisition
Sentinel-1 + Sentinel-2 + Flood Labels
              │
              ▼
Scene Quality Screening
Coverage, Valid Pixels, Eligibility
              │
              ▼
OpenStreetMap Road Network Extraction
              │
              ▼
Transportation Knowledge Graph
Roads, Nodes, Edges, Attributes
              │
              ▼
Road-Flood Spatial Grounding
              │
              ▼
Scene and Roadway Metrics
Flood Exposure, Connectivity, Disruption
              │
              ▼
Transportation Instruction Generation
              │
              ▼
Multimodal VLM Dataset
Images + Questions + Answers + Knowledge
              │
              ▼
Vision Encoder + Knowledge Adapter
              │
              ▼
Multimodal Fusion and VLM Training
              │
              ▼
Road-Flood Classification and Reasoning
              │
              ▼
Resilience Graph Analysis
```

---

## Architecture

```text
┌───────────────────────────────────────────────────────────────┐
│                       INPUT DATA                              │
├──────────────────────┬──────────────────────┬─────────────────┤
│ Sentinel-1 SAR       │ Sentinel-2 Optical   │ Flood Labels    │
│ imagery              │ imagery              │ and masks       │
└──────────┬───────────┴──────────┬───────────┴────────┬────────┘
           │                      │                    │
           └──────────────────────┼────────────────────┘
                                  ▼
                     ┌─────────────────────────┐
                     │ Scene Quality Pipeline  │
                     │ validation + filtering  │
                     └────────────┬────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │ Visual Representation   │
                     │ vision encoder          │
                     └────────────┬────────────┘
                                  │
        ┌─────────────────────────┼────────────────────────┐
        │                         │                        │
        ▼                         ▼                        ▼
┌──────────────┐       ┌──────────────────┐      ┌─────────────────┐
│ OSM Network  │       │ Road-Flood       │      │ Scene Metrics   │
│ roads/nodes  │       │ Spatial Overlay  │      │ coverage/risk   │
└──────┬───────┘       └─────────┬────────┘      └────────┬────────┘
       │                         │                        │
       └─────────────────────────┼────────────────────────┘
                                 ▼
                    ┌──────────────────────────┐
                    │ Transportation Knowledge │
                    │ Graph / Adapter          │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Multimodal Fusion        │
                    │ visual + graph knowledge │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Vision-Language Model    │
                    │ classification/reasoning │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Resilience Assessment    │
                    │ road impacts + network   │
                    └──────────────────────────┘
```

---

## Repository Structure

```text
ResilientVLM/
├── configs/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── outputs/
│
├── executed_notebooks/
│
├── logs/
│
├── notebooks/
│   ├── 01_data_discovery.ipynb
│   ├── 02_dataset_acquisition.ipynb
│   ├── 02A_recover_missing_s1_s2_downloads.ipynb
│   ├── 03_scene_quality_selection.ipynb
│   ├── 04_transportation_knowledge_graph.ipynb
│   ├── 05_transport_knowledge.ipynb
│   ├── 06_road_flood_grounding.ipynb
│   ├── 07_instruction_generation.ipynb
│   ├── 08_build_vlm_dataset.ipynb
│   └── additional model, training, evaluation, and ablation notebooks
│
├── outputs/
│   ├── figures/
│   ├── tables/
│   ├── maps/
│   ├── checkpoints/
│   └── reports/
│
├── scripts/
│
├── src/
│   ├── data/
│   │   ├── flood.py
│   │   ├── osm.py
│   │   ├── fema.py
│   │   └── census.py
│   │
│   ├── models/
│   │   ├── visual_encoder.py
│   │   ├── knowledge_adapter.py
│   │   ├── fusion.py
│   │   └── resilience_graph.py
│   │
│   └── utils/
│       ├── spatial.py
│       └── metrics.py
│
├── create_notebook14.py
├── inventory.py
├── patch_continue_from_adapter.py
├── patch_notebook10_sturm_classification_prompt.py
├── patch_notebook10_sturm_paths.py
├── patch_notebook12.py
├── requirements.txt
├── .gitignore
└── README.md
```

Some notebook backup files and recovery scripts are retained in the repository for development continuity. These can be archived or removed before a production release.

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming language | Python |
| Deep learning | PyTorch, TorchVision |
| Vision-language models | Hugging Face Transformers |
| Vision backbones | TIMM |
| Geospatial analysis | GeoPandas, Rasterio, Fiona, Shapely |
| Data processing | NumPy, Pandas, SciPy |
| Visualization | Matplotlib, Seaborn |
| Configuration | PyYAML, python-dotenv |
| Notebook environment | Jupyter, IPython |
| Testing and development | Pytest, Black, Flake8 |

The project currently declares the following main dependencies:

```text
numpy
pandas
scipy
geopandas
rasterio
shapely
fiona
torch
torchvision
transformers
timm
pillow
pyyaml
python-dotenv
tqdm
matplotlib
seaborn
jupyter
ipython
pytest
black
flake8
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/takyiyawsamuel90/ResilientVLM.git
cd ResilientVLM
```

### 2. Create a virtual environment

#### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify the environment

```bash
python -c "import torch, geopandas, rasterio, transformers; print('Environment ready')"
```

### GPU verification

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

For CUDA-based training, install a PyTorch build compatible with the CUDA version available on your workstation or HPC cluster.

---

## Quick Start

Launch Jupyter from the repository root:

```bash
jupyter notebook
```

Run the notebooks in sequence, beginning with:

```text
01_data_discovery.ipynb
02_dataset_acquisition.ipynb
03_scene_quality_selection.ipynb
04_transportation_knowledge_graph.ipynb
05_transport_knowledge.ipynb
06_road_flood_grounding.ipynb
07_instruction_generation.ipynb
08_build_vlm_dataset.ipynb
```

Continue with the model-development, training, evaluation, graph-analysis, and ablation notebooks included in the repository.

Do not skip directly to model training unless the required processed datasets, transportation graphs, grounded road metrics, and instruction files already exist.

---

## Notebook Pipeline

### 01 — Data Discovery

```text
notebooks/01_data_discovery.ipynb
```

Purpose:

- inspect available datasets;
- identify scene directories and files;
- review dataset organization;
- establish project paths;
- summarize available flood scenes.

Typical outputs:

- scene inventory;
- file counts;
- dataset availability summaries;
- initial quality diagnostics.

---

### 02 — Dataset Acquisition

```text
notebooks/02_dataset_acquisition.ipynb
```

Purpose:

- acquire Sentinel-1 and Sentinel-2 scene files;
- organize SEN1FLOODS11 data;
- prepare source imagery and masks;
- record download and processing status.

---

### 02A — Missing Download Recovery

```text
notebooks/02A_recover_missing_s1_s2_downloads.ipynb
```

Purpose:

- detect missing Sentinel-1 or Sentinel-2 downloads;
- retry failed downloads;
- validate recovered files;
- log recovery outcomes;
- synchronize scene inventory counts.

Use this notebook only when the acquisition stage reports incomplete scenes.

---

### 03 — Scene Quality Selection

```text
notebooks/03_scene_quality_selection.ipynb
```

Purpose:

- calculate valid-pixel percentages;
- screen scenes for completeness;
- identify eligible scenes;
- exclude low-quality or unusable samples;
- generate scene-quality summaries.

Typical quality fields may include:

```text
scene_id
country
valid_pct
eligible
s1_exists
s2_exists
label_exists
```

---

### 04 — Transportation Knowledge Graph

```text
notebooks/04_transportation_knowledge_graph.ipynb
```

Purpose:

- extract roadway networks from OpenStreetMap;
- construct transportation graph objects;
- create road and node inventories;
- attach spatial and network attributes;
- export GraphML and GeoPackage files;
- create processing and recovery logs.

Typical outputs:

```text
data/processed/transportation_knowledge_graphs/graphml/
data/processed/transportation_knowledge_graphs/geopackages/
```

The scene inventory can include fields such as:

```text
scene_id
graphml_exists
roads_exists
ready_to_process
```

---

### 05 — Transportation Knowledge

```text
notebooks/05_transport_knowledge.ipynb
```

Purpose:

- derive transportation-domain features;
- summarize road and graph characteristics;
- prepare knowledge representations for model integration;
- generate scene-level transportation metrics.

---

### 06 — Road-Flood Grounding

```text
notebooks/06_road_flood_grounding.ipynb
```

Purpose:

- overlay flood information with roadway geometry;
- identify flooded and non-flooded road segments;
- compute road-level exposure;
- generate grounded multimodal samples;
- connect visual evidence to transportation features.

---

### 07 — Instruction Generation

```text
notebooks/07_instruction_generation.ipynb
```

Purpose:

- generate transportation-aware questions and answers;
- construct instruction-following samples;
- create classification and reasoning prompts;
- connect road-level evidence to natural-language tasks.

Example task types may include:

- flood-status classification;
- road-disruption identification;
- accessibility reasoning;
- network-impact comparison;
- transportation resilience interpretation.

---

### 08 — Build the VLM Dataset

```text
notebooks/08_build_vlm_dataset.ipynb
```

Purpose:

- combine images, transportation knowledge, instructions, and labels;
- create training, validation, and test records;
- validate dataset paths;
- generate model-ready multimodal examples.

A model-ready record may contain:

```json
{
  "scene_id": "USA_58086",
  "image_path": "data/processed/images/USA_58086.png",
  "question": "Are major roadway segments affected by flooding?",
  "answer": "Yes",
  "label": "flooded",
  "transportation_metrics": {},
  "graph_path": "data/processed/transportation_knowledge_graphs/graphml/USA_58086.graphml"
}
```

The exact schema should match the notebook output used by the training pipeline.

---

### Model Development and Training

Later notebooks in the project support:

- visual baseline development;
- transportation knowledge adapter training;
- vision-language fusion;
- full ResilientVLM training;
- checkpoint recovery;
- continuation from saved adapters;
- classification and reasoning evaluation.

---

### Resilience Graph Analysis

The graph-analysis workflow uses model predictions and roadway network structure to examine:

- affected roadway segments;
- connectivity loss;
- component fragmentation;
- isolated road groups;
- critical links;
- network-level resilience indicators.

---

### Experiments and Ablations

The experiment workflow can compare:

- visual-only baselines;
- transportation-knowledge-only models;
- fused vision and transportation models;
- adapter configurations;
- classification prompts;
- scene-selection thresholds;
- alternative training and evaluation settings.

---

## Data Sources

### SEN1FLOODS11

The primary flood dataset provides multimodal flood scenes that may include:

- Sentinel-1 synthetic aperture radar imagery;
- Sentinel-2 optical imagery;
- manually labeled flood masks;
- scenes from multiple countries and flood events.

The repository does not redistribute the complete external dataset. Download and use the data according to its original access conditions and license.

### OpenStreetMap

OpenStreetMap provides roadway geometry and network structure used to build transportation knowledge graphs.

Possible attributes include:

- road class;
- road name;
- segment length;
- node degree;
- edge connectivity;
- bridge or tunnel indicators;
- network position.

### FEMA

FEMA flood information can provide additional flood-hazard and contextual data for U.S.-based scenes.

### American Community Survey

ACS data can support socioeconomic and community vulnerability analysis when geographic coverage and spatial linkage are available.

---

## Configuration

The main configuration file is:

```text
configs/config.yaml
```

Use it to define:

- project and data directories;
- raw, interim, and processed-data paths;
- output directories;
- scene-selection settings;
- model architecture;
- pretrained model identifiers;
- training hyperparameters;
- checkpoint paths;
- logging settings;
- device configuration.

Example structure:

```yaml
paths:
  raw_data: data/raw
  interim_data: data/interim
  processed_data: data/processed
  outputs: outputs

model:
  vision_backbone: default
  hidden_dim: 768
  use_transport_adapter: true

training:
  batch_size: 8
  learning_rate: 0.0001
  epochs: 20

logging:
  directory: logs
```

Treat this as an example only. Use the keys already defined in the repository's actual `config.yaml`.

---

## Outputs

Generated artifacts are stored under `outputs/`, `data/outputs/`, or notebook-specific processed-data directories.

Common outputs include:

```text
outputs/
├── figures/
├── tables/
├── maps/
├── checkpoints/
├── predictions/
├── evaluations/
└── reports/
```

Transportation graph outputs are commonly written to:

```text
data/processed/transportation_knowledge_graphs/
├── graphml/
└── geopackages/
```

Other possible outputs include:

- scene inventories;
- download logs;
- recovery logs;
- scene-quality tables;
- road-flood overlay layers;
- transportation metrics;
- generated instructions;
- VLM dataset records;
- model checkpoints;
- classification results;
- resilience graphs;
- ablation summaries.

---

## Running on an HPC System

### 1. Clone the repository

```bash
git clone https://github.com/takyiyawsamuel90/ResilientVLM.git
cd ResilientVLM
```

### 2. Create an isolated environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On clusters using modules:

```bash
module load python
module load cuda
```

Module names vary by HPC system.

### 3. Keep environment files out of Git

Do not commit:

```text
.venv/
venv/
myenv/
__pycache__/
.ipynb_checkpoints/
```

### 4. Run notebooks non-interactively

```bash
jupyter nbconvert \
  --to notebook \
  --execute notebooks/04_transportation_knowledge_graph.ipynb \
  --output executed_notebooks/04_transportation_knowledge_graph.executed.ipynb
```

### 5. Use a batch script

Example Slurm script:

```bash
#!/bin/bash
#SBATCH --job-name=resilientvlm
#SBATCH --output=logs/resilientvlm_%j.out
#SBATCH --error=logs/resilientvlm_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

source .venv/bin/activate

jupyter nbconvert \
  --to notebook \
  --execute notebooks/08_build_vlm_dataset.ipynb \
  --output executed_notebooks/08_build_vlm_dataset.executed.ipynb
```

Adjust resources and paths to match the target cluster.

---

## Development Notes

### Run from the project root

Always launch notebooks and scripts from the repository root so relative paths resolve consistently.

```bash
cd ResilientVLM
jupyter notebook
```

### Do not commit local environments

The repository currently contains a `myenv/` directory. Local environments should generally be removed from version control and added to `.gitignore`.

Recommended entries:

```gitignore
.venv/
venv/
myenv/
__pycache__/
*.pyc
.ipynb_checkpoints/
.env
```

### Notebook backups

The repository contains `.bak` notebook copies and patch utilities. These are useful during active development but can make the public repository difficult to navigate.

For a cleaner release:

1. move backups into an archive directory;
2. retain only the latest working notebook;
3. document recovery scripts separately;
4. tag stable releases in Git.

### Large generated data

Do not commit large:

- satellite rasters;
- GeoPackages;
- GraphML collections;
- model checkpoints;
- generated dataset files;
- notebook outputs.

Use external storage, Git LFS, Hugging Face Hub, Zenodo, or another artifact repository.

### Reproducibility

For reproducible runs:

- fix random seeds;
- record dataset versions;
- record scene-selection thresholds;
- save configuration files with checkpoints;
- record the exact package environment;
- preserve train, validation, and test scene IDs;
- document hardware used for training.

---

## Roadmap

- [ ] Add a command-line pipeline runner.
- [ ] Convert core notebook logic into reusable Python modules.
- [ ] Add automated dataset validation.
- [ ] Add end-to-end scene processing from one command.
- [ ] Add downloadable sample data.
- [ ] Add pretrained model checkpoints.
- [ ] Add Hugging Face dataset and model integration.
- [ ] Add interactive road-flood maps.
- [ ] Add automated network resilience reports.
- [ ] Add confidence and uncertainty estimates.
- [ ] Add cross-country generalization evaluation.
- [ ] Add additional flood datasets.
- [ ] Add model explainability and visual grounding.
- [ ] Add structured experiment tracking.
- [ ] Add `pytest` test coverage.
- [ ] Add continuous integration.
- [ ] Add Docker support.
- [ ] Remove local virtual environments and notebook backups from Git.
- [ ] Add a formal release workflow.

---

## Contributing

Contributions are welcome.

### Standard workflow

1. Fork the repository.
2. Create a feature branch.

```bash
git checkout -b feature/your-feature-name
```

3. Make and test your changes.
4. Format the code.

```bash
black src scripts
```

5. Run checks.

```bash
flake8 src scripts
pytest
```

6. Commit the changes.

```bash
git add .
git commit -m "Add your feature description"
```

7. Push the branch.

```bash
git push origin feature/your-feature-name
```

8. Open a pull request.

Do not include local environments, credentials, raw datasets, large checkpoints, or machine-specific paths.

---

## Authors

### Samuel Takyi

Project development, transportation resilience, geospatial analysis, flood-impact modeling, roadway-network analysis, and vision-language system design.

### Justice Adjei-Owusu

Project development, multimodal AI, model implementation, transportation knowledge integration, and technical collaboration.

---

## Repository

```text
https://github.com/takyiyawsamuel90/ResilientVLM
```
