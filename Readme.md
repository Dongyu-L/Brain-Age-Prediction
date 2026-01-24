# Brain Age Prediction Pipeline

End-to-end brain age prediction from MRI data using pretrained DenseNet models.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run pipeline
python main.py \
    --metadata_file data/participants.csv \
    --image_root data/T1_images \
    --model_path models/best_model.pt \
    --output_root results
```

## What It Does

1. **Index** - Scans dataset and matches images to metadata
2. **Preprocess** - ANTs-based brain MRI preprocessing
3. **Split** - Stratified train/val/test split (70/15/15)
4. **Predict** - Age prediction using pretrained model

## Requirements

- Python 3.8+
- PyTorch 2.0+
- MONAI 1.0+
- ANTs (for preprocessing)
- CUDA GPU (recommended)

## Input Data

**Metadata file** (CSV/TSV/Excel) with columns:
- Subject ID (e.g., `subject_id`, `participant_id`, `ID`)
- Age (e.g., `age`, `Age`)

Example `participants.csv`:
```csv
subject_id,age
sub-001,45
sub-002,62
sub-003,28
```

**MRI images** in NIfTI format (.nii/.nii.gz)

Example structure:
```
data/
├── participants.csv          # Your metadata file (any name)
└── T1_images/                # Your images folder (any name)
    ├── sub-001_T1.nii.gz
    ├── sub-002_T1.nii.gz
    └── sub-003_T1.nii.gz
```

## Output

```
results/
├── dataset_index.csv
├── preprocessed/
├── splits/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── predictions/
    ├── predictions_test.csv
    └── metrics_test.csv
```

## Command-Line Options

```bash
--metadata_file    Path to metadata (CSV/TSV/Excel)
--image_root       MRI images directory
--model_path       Pretrained model (.pt file)
--output_root      Output directory (default: results)
--skip_existing    Skip completed steps
```

## Example

```bash
# Example with your own data
python main.py \
    --metadata_file data/my_subjects.csv \
    --image_root data/brain_scans \
    --model_path best_model.pt \
    --output_root my_results \
    --skip_existing
```

## Results

**Predictions** (`predictions_test.csv`):
```csv
ID,Age,predicted_age,age_error
sub-001,45,47.2,-2.2
sub-002,62,59.8,2.2
```

**Metrics** (`metrics_test.csv`):
```csv
split,n_samples,MAE,RMSE
test,150,4.23,5.67
```

## Components

- `main.py` - Pipeline orchestrator
- `indexer.py` - Dataset indexing
- `preprocessor.py` - MRI preprocessing
- `splitter.py` - Data splitting

