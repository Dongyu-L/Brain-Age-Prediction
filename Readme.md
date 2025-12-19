# Brain Age Prediction Pipeline

A universal deep learning pipeline for brain age prediction from structural MRI data.

## Overview

This pipeline implements an end-to-end workflow for training brain age prediction models on structural MRI data. The system is designed to work with any dataset structure without manual configuration.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)

See `requirements.txt` for complete dependencies.

## Installation

```bash
pip install -r requirements.txt
```

Or use the provided setup scripts:
```bash
./setup.sh      # Linux/Mac
setup.bat       # Windows
```

## Quick Start

```bash
# 1. Create configuration
python main.py --create_example_config

# 2. Edit configuration with your data paths
nano example_config.yaml

# 3. Run pipeline
python main.py --config example_config.yaml --run_all
```

## Pipeline Components

### 1. Data Indexing
```bash
python -m data.indexer \
  --metadata_file data/IXI/IXI.xls \
  --t1_root data/IXI/IXI-T1 \
  --output raw_index.csv
```

### 2. Preprocessing
```bash
python -m data.preprocessor \
  --input_csv raw_index.csv \
  --output_dir preprocessed \
  --template MNI152_T1_1mm.nii.gz
```

### 3. Dataset Splitting
```bash
python -m data.splitter \
  --input_csv preprocessed/preprocessed_index.csv \
  --output_dir splits \
  --ratios 0.6 0.2 0.2 \
  --stratify_age
```

### 4. Training
```bash
python -m training.trainer \
  --train_csv splits/train_split.csv \
  --val_csv splits/val_split.csv \
  --test_csv splits/test_split.csv \
  --output_dir experiments/exp001
```

## Configuration

Configuration files use YAML format. Generate template:

```bash
python config.py --create_example --output my_config.yaml
```

Example configuration:
```yaml
paths:
  metadata_file: data/IXI/IXI.xls
  t1_root: data/IXI/IXI-T1
  template: templates/MNI152_T1_1mm.nii.gz
  output_root: outputs

preprocessor:
  registration_type: Rigid
  n4_iterations: [50, 50, 50, 50]

splitter:
  train_ratio: 0.6
  val_ratio: 0.2
  test_ratio: 0.2
  stratify_age: true

trainer:
  batch_size: 2
  epochs: 500
  lr: 0.0001
```

## Project Structure

```
Brain_Age_Prediction/
├── data/               # Data processing modules
├── training/           # Training components
├── utils/              # Validation utilities
├── config.py           # Configuration management
├── main.py             # Pipeline orchestrator
└── requirements.txt    # Dependencies
```

## Output Structure

```
outputs/
├── raw_index.csv
├── preprocessed/
│   └── preprocessed_index.csv
├── splits/
│   ├── train_split.csv
│   ├── val_split.csv
│   └── test_split.csv
└── experiments/
    └── exp001/
        ├── checkpoints/
        ├── training_history.csv
        └── test_results.json
```

## Validation

Validate data quality at each step:

```bash
python -m utils.validation --raw raw_index.csv
python -m utils.validation --preprocessed preprocessed/preprocessed_index.csv
python -m utils.validation --splits splits/train_split.csv splits/val_split.csv splits/test_split.csv
```
