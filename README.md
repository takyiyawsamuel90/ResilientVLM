# ResilientVLM

A Vision-Language Model approach for understanding resilience in transportation networks under flood conditions.

## Overview

ResilientVLM integrates multiple data sources to assess and predict the resilience of transportation networks to flooding events. The project combines:

- **Sentinel-1 Flood Data** (sen1floods11)
- **OpenStreetMap Network Data** (OSM)
- **FEMA Flood Information**
- **American Community Survey Data** (ACS)

## Project Structure

```
ResilientVLM/
├── data/              # Data storage (raw, interim, processed)
├── notebooks/         # Jupyter notebooks for exploration and experiments
├── src/              # Source code
│   ├── data/         # Data loading and processing modules
│   ├── models/       # Model architectures
│   └── utils/        # Utility functions
├── configs/          # Configuration files
├── outputs/          # Generated outputs (figures, tables, maps, checkpoints)
├── requirements.txt  # Python dependencies
└── README.md        # This file
```

## Getting Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/takyiyawsamuel90/ResilientVLM.git
cd ResilientVLM
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Quick Start

Run the notebooks in order:

1. `01_data_discovery.ipynb` - Explore available data sources
2. `02_sen1floods11_exploration.ipynb` - Analyze Sentinel-1 flood data
3. `03_osm_network_extraction.ipynb` - Extract transportation networks
4. `04_road_flood_overlay.ipynb` - Overlay floods on road networks
5. `05_fema_context.ipynb` - Incorporate FEMA data
6. `06_acs_vulnerability.ipynb` - Add socioeconomic vulnerability data
7. `07_master_dataset.ipynb` - Combine all data sources
8. `08_vision_baseline.ipynb` - Train vision baseline model
9. `09_transportation_knowledge_adapter.ipynb` - Develop knowledge adapter
10. `10_resiliencevlm.ipynb` - Train full ResilientVLM model
11. `11_resilience_graph.ipynb` - Build and analyze resilience graphs
12. `12_experiments_ablation.ipynb` - Run ablation studies and experiments

## Key Components

### Data Modules (`src/data/`)
- `flood.py` - Sentinel-1 flood data processing
- `osm.py` - OpenStreetMap network extraction
- `fema.py` - FEMA flood information integration
- `census.py` - American Community Survey data processing

### Models (`src/models/`)
- `visual_encoder.py` - Vision-based feature extraction
- `knowledge_adapter.py` - Domain knowledge integration
- `fusion.py` - Multi-modal fusion strategy
- `resilience_graph.py` - Graph-based resilience modeling

### Utilities (`src/utils/`)
- `spatial.py` - Geospatial operations and analysis
- `metrics.py` - Evaluation metrics and analysis functions

## Configuration

Edit `configs/config.yaml` to customize:
- Data directories
- Model architecture parameters
- Training hyperparameters
- Logging settings

## Requirements

Key dependencies include:
- PyTorch and TorchVision for deep learning
- GeoPandas and Rasterio for geospatial data
- Transformers (Hugging Face) for pre-trained models
- PyYAML for configuration management

See `requirements.txt` for complete list and versions.

## License

MIT License - See LICENSE file for details

## Contact

For questions or issues, please open an issue on GitHub.
