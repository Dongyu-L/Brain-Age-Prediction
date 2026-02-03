# Training

Train your own brain age prediction model.

## Prerequisites

Prepare your data using the main pipeline first:

```bash
python main.py \
    --metadata_file participants.csv \
    --image_root T1_images/ \
    --model_path dummy.pt \
    --output_root results
```

This creates the split CSVs needed for training (prediction step will be skipped if model doesn't exist).

## Train

```bash
python -m training.trainer \
    --train_csv results/splits/train_split.csv \
    --val_csv results/splits/val_split.csv \
    --test_csv results/splits/test_split.csv \
    --output_dir experiments/exp001
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--train_csv` | Required | Training split CSV |
| `--val_csv` | Required | Validation split CSV |
| `--test_csv` | Required | Test split CSV |
| `--output_dir` | Required | Output directory |
| `--modality` | `T1` | Image modality |
| `--epochs` | `500` | Max training epochs |
| `--batch_size` | `2` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--weight_decay` | `1e-5` | Weight decay |
| `--early_stopping_patience` | `15` | Early stopping patience |
| `--seed` | `42` | Random seed |
| `--resume` | - | Resume from checkpoint |
| `--test_only` | - | Run test evaluation only |

## Output

```
experiments/exp001/
├── checkpoints/
│   ├── best_checkpoint.pt    # Best model (use this)
│   └── last_checkpoint.pt
├── logs/
│   └── training.log
└── results/
    ├── training_history.csv
    └── test_results.csv
```

## Run Inference with Trained Model

```bash
python main.py \
    --metadata_file new_data.csv \
    --image_root new_images/ \
    --model_path experiments/exp001/checkpoints/best_checkpoint.pt \
    --output_root results
```
