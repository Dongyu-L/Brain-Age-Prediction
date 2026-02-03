# Brain Age Prediction

Predict brain age from MRI using pretrained DenseNet121.

## Quick Start

```bash
pip install -r Requirements.txt

python main.py \
    --metadata_file participants.csv \
    --image_root T1_images/ \
    --model_path pretrained.pt
```

## Input Requirements

| Input | Description |
|-------|-------------|
| Metadata file | CSV/TSV/Excel with `ID` and `Age` columns |
| MRI images | NIfTI files (`.nii`/`.nii.gz`) |
| Pretrained model | PyTorch checkpoint (`.pt`) |

**Metadata example:**
```csv
ID,Age
sub-001,45
sub-002,62
```

## Output

```
results/
├── preprocessed/              # N4 + MNI registered images
├── splits/
│   ├── train_split.csv
│   ├── val_split.csv
│   └── test_split.csv
└── predictions/
    ├── predictions_*.csv      # ID, Age, predicted_age, age_error
    └── metrics_*.csv          # MAE, RMSE
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--metadata_file` | Yes | Metadata file path |
| `--image_root` | Yes | MRI images directory |
| `--model_path` | Yes | Pretrained model path |
| `--output_root` | No | Output directory (default: `results`) |
| `--skip_existing` | No | Skip completed steps |

## Pipeline

1. **Indexing** - Match metadata IDs with image files
2. **Preprocessing** - N4 bias correction + MNI registration
3. **Splitting** - Age-stratified split (70/15/15)
4. **Prediction** - DenseNet121 inference

## Training

To train your own model, see [training/README.md](training/README.md).

## Requirements

- Python 3.8+
- PyTorch 2.0+
- MONAI 1.3+
- ANTsPy 0.4+
- CUDA GPU (recommended)
