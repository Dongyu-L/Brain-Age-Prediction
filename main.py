"""
Brain Age Prediction - Inference Pipeline
==========================================

Simplified pipeline for brain age prediction using pretrained model:
1. Index dataset from metadata
2. Preprocess MRI images
3. Split dataset
4. Predict ages using pretrained model

Usage:
    python main.py \
        --metadata_file data/IXI/IXI.xls \
        --image_root data/IXI/IXI-T1 \
        --model_path models/best_model.pt \
        --output_root results
"""

import argparse
import logging
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Import data processing modules
from data.indexer import UniversalIndexer
from data.preprocessor import UniversalPreprocessor
from data.splitter import UniversalSplitter

# Import deep learning modules
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    Orientationd, ScaleIntensityRanged, CropForegroundd, EnsureTyped
)
from tqdm import tqdm


class BrainAgeInferenceDataset(Dataset):
    """Dataset for brain age prediction inference"""

    def __init__(self, csv_file: str, transforms=None):
        self.data = pd.read_csv(csv_file)
        self.transforms = transforms

        # Find image path column (prioritize preprocessed paths)
        self.image_col = self._find_column([
            'T1_preprocessed_path', 'T2_preprocessed_path',
            'preprocessed_path', 'image_path',
            'T1_path', 't1_path', 'path'
        ])
        if not self.image_col:
            raise ValueError(f"Could not find image path column in {csv_file}")

        # Find age column (optional for prediction)
        self.age_col = self._find_column(['age', 'Age'])

    def _find_column(self, candidates):
        for col in candidates:
            if col in self.data.columns:
                # Check if column has valid paths
                valid_count = (~self.data[col].isna() & (self.data[col] != '')).sum()
                if valid_count > 0:
                    return col
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        data_dict = {'image': row[self.image_col]}

        # Handle age: check if column exists and value is not NaN/empty
        if self.age_col and self.age_col in row:
            age_val = row[self.age_col]
            if pd.notna(age_val) and age_val != '':
                data_dict['age'] = float(age_val)
                data_dict['has_age'] = True
            else:
                data_dict['age'] = -1.0  # placeholder
                data_dict['has_age'] = False
        else:
            data_dict['age'] = -1.0
            data_dict['has_age'] = False

        if self.transforms:
            data_dict = self.transforms(data_dict)

        return data_dict


class BrainAgePipeline:
    """Brain age prediction pipeline with pretrained model"""

    def __init__(self,
                 metadata_file: str,
                 image_root: str,
                 model_path: str,
                 output_root: str,
                 skip_existing: bool = False):
        """
        Args:
            metadata_file: Path to metadata CSV/Excel/TSV
            image_root: Root directory containing MRI images
            model_path: Path to pretrained model (.pt file)
            output_root: Output directory for all results
            skip_existing: Skip steps if outputs exist
        """
        self.metadata_file = metadata_file
        self.image_root = image_root
        self.model_path = model_path
        self.output_root = Path(output_root)
        self.skip_existing = skip_existing

        # Setup
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._setup_logging()

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logging.info("=" * 70)
        logging.info("BRAIN AGE PREDICTION PIPELINE")
        logging.info("=" * 70)
        logging.info(f"Metadata: {metadata_file}")
        logging.info(f"Images: {image_root}")
        logging.info(f"Model: {model_path}")
        logging.info(f"Output: {self.output_root}")
        logging.info(f"Device: {self.device}")
        logging.info("")

    def _setup_logging(self):
        """Setup logging"""
        log_dir = self.output_root / "logs"
        log_dir.mkdir(exist_ok=True)

        log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logging.info(f"Log file: {log_file}")

    def run(self):
        """Run complete pipeline"""
        try:
            # Step 1: Index dataset
            index_csv = self._run_indexing()

            # Step 2: Preprocess
            preprocessed_csv = self._run_preprocessing(index_csv)

            # Step 3: Split dataset
            train_csv, val_csv, test_csv = self._run_splitting(preprocessed_csv)

            # Step 4: Predict on all splits
            self._run_prediction(train_csv, "train")
            self._run_prediction(val_csv, "val")
            self._run_prediction(test_csv, "test")

            logging.info("")
            logging.info("=" * 70)
            logging.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logging.info("=" * 70)
            logging.info(f"Results saved to: {self.output_root}")

        except Exception as e:
            logging.error(f"Pipeline failed: {str(e)}")
            raise

    def _run_indexing(self) -> str:
        """Step 1: Create dataset index"""
        logging.info("")
        logging.info("=" * 70)
        logging.info("STEP 1: INDEXING DATASET")
        logging.info("=" * 70)

        output_csv = self.output_root / "dataset_index.csv"

        if self.skip_existing and output_csv.exists():
            logging.info(f"✓ Index already exists: {output_csv}")
            return str(output_csv)

        indexer = UniversalIndexer(
            metadata_file=self.metadata_file,
            image_roots={'T1': self.image_root},
            output_csv=str(output_csv),
            recursive=True,
            min_age=0.0,
            max_age=120.0,
        )

        indexer.create_index()
        logging.info(f"✓ Index created: {output_csv}")

        return str(output_csv)

    def _run_preprocessing(self, index_csv: str) -> str:
        """Step 2: Preprocess MRI images"""
        logging.info("")
        logging.info("=" * 70)
        logging.info("STEP 2: PREPROCESSING")
        logging.info("=" * 70)

        preproc_dir = self.output_root / "preprocessed"
        output_csv = self.output_root / "preprocessed_index.csv"

        if self.skip_existing and output_csv.exists():
            logging.info(f"✓ Preprocessed data already exists: {output_csv}")
            return str(output_csv)

        # Note: template_path should point to actual MNI152 template file
        # Default location: templates/MNI152_T1_1mm.nii.gz
        template_path = Path(r"T1_Template.nii\T1_Template.nii")
        if not template_path.exists():
            logging.warning(f"Template not found at {template_path}, using default ANTs template")
            template_path = r"T1_Template.nii\T1_Template.nii"  # ANTs may have built-in template

        preprocessor = UniversalPreprocessor(
            input_csv=index_csv,
            output_dir=str(preproc_dir),
            template_path=str(template_path),
            skip_existing=self.skip_existing,
        )

        # Run preprocessing - this generates preprocessed_index.csv automatically
        output_df = preprocessor.run()

        # The preprocessor creates its own output CSV at preproc_dir/preprocessed_index.csv
        generated_csv = preproc_dir / "preprocessed_index.csv"
        if generated_csv.exists():
            # Copy to expected location
            import shutil
            shutil.copy(generated_csv, output_csv)
        else:
            # Fallback: manually create output CSV if preprocessor didn't
            output_df.to_csv(output_csv, index=False)

        logging.info(f"✓ Preprocessing completed: {output_csv}")

        return str(output_csv)

    def _run_splitting(self, preprocessed_csv: str) -> tuple:
        """Step 3: Split dataset"""
        logging.info("")
        logging.info("=" * 70)
        logging.info("STEP 3: SPLITTING DATASET")
        logging.info("=" * 70)

        split_dir = self.output_root / "splits"
        # Note: UniversalSplitter generates files with _split suffix
        train_csv = split_dir / "train_split.csv"
        val_csv = split_dir / "val_split.csv"
        test_csv = split_dir / "test_split.csv"

        if self.skip_existing and all([train_csv.exists(), val_csv.exists(), test_csv.exists()]):
            logging.info(f"✓ Splits already exist")
            return str(train_csv), str(val_csv), str(test_csv)

        splitter = UniversalSplitter(
            input_csv=preprocessed_csv,
            output_dir=str(split_dir),
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            stratify_age=True,  # Fixed: was stratify_by_age
            seed=42,  # Fixed: was random_seed
        )

        splitter.run()  # Fixed: was split()

        logging.info(f"✓ Dataset split completed")
        logging.info(f"  Train: {train_csv} ({pd.read_csv(train_csv).__len__()} samples)")
        logging.info(f"  Val: {val_csv} ({pd.read_csv(val_csv).__len__()} samples)")
        logging.info(f"  Test: {test_csv} ({pd.read_csv(test_csv).__len__()} samples)")

        return str(train_csv), str(val_csv), str(test_csv)

    def _run_prediction(self, csv_file: str, split_name: str):
        """Step 4: Predict ages using pretrained model"""
        logging.info("")
        logging.info("=" * 70)
        logging.info(f"STEP 4: PREDICTING - {split_name.upper()} SET")
        logging.info("=" * 70)

        # Load model
        logging.info(f"Loading model from: {self.model_path}")
        model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=1).to(self.device)

        checkpoint = torch.load(self.model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        logging.info("✓ Model loaded successfully")

        # Data transforms (no augmentation for inference)
        transforms = Compose([
            LoadImaged(keys=['image']),
            EnsureChannelFirstd(keys=['image']),
            Spacingd(keys=['image'], pixdim=(1.5, 1.5, 1.5), mode='bilinear'),
            Orientationd(keys=['image'], axcodes='RAS'),
            ScaleIntensityRanged(keys=['image'], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=['image'], source_key='image'),
            EnsureTyped(keys=['image']),
        ])

        # Dataset and loader
        dataset = BrainAgeInferenceDataset(csv_file, transforms=transforms)
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)

        # Run predictions
        predictions = []
        true_ages = []
        has_age_flags = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Predicting {split_name}"):
                images = batch['image'].to(self.device)
                outputs = model(images)
                predictions.extend(outputs.cpu().numpy().flatten())

                if 'age' in batch:
                    true_ages.extend(batch['age'].numpy())
                if 'has_age' in batch:
                    has_age_flags.extend(batch['has_age'].numpy())

        # Save results
        results_dir = self.output_root / "predictions"
        results_dir.mkdir(exist_ok=True)

        results_df = pd.read_csv(csv_file)
        results_df['predicted_age'] = predictions

        # Handle partial age labels
        if true_ages and has_age_flags:
            results_df['true_age'] = true_ages
            results_df['has_age_label'] = has_age_flags

            # Calculate error only for subjects with age labels
            valid_mask = np.array(has_age_flags, dtype=bool)
            results_df['age_error'] = np.where(
                valid_mask,
                np.array(predictions) - np.array(true_ages),
                np.nan
            )

            # Calculate metrics only for subjects with valid age labels
            n_with_age = valid_mask.sum()
            if n_with_age > 0:
                valid_errors = results_df.loc[valid_mask, 'age_error']
                mae = np.mean(np.abs(valid_errors))
                rmse = np.sqrt(np.mean(valid_errors**2))

                logging.info(f"✓ {split_name.upper()} SET RESULTS:")
                logging.info(f"  Total subjects: {len(results_df)}")
                logging.info(f"  Subjects with age GT: {n_with_age}")
                logging.info(f"  Subjects without age GT: {len(results_df) - n_with_age}")
                logging.info(f"  MAE (on {n_with_age} subjects): {mae:.2f} years")
                logging.info(f"  RMSE (on {n_with_age} subjects): {rmse:.2f} years")

                # Save metrics
                metrics = {
                    'split': split_name,
                    'n_total': len(results_df),
                    'n_with_age': n_with_age,
                    'MAE': mae,
                    'RMSE': rmse,
                }
                metrics_file = results_dir / f"metrics_{split_name}.csv"
                pd.DataFrame([metrics]).to_csv(metrics_file, index=False)
            else:
                logging.info(f"✓ {split_name.upper()} SET: {len(results_df)} subjects predicted (no age GT available)")

        # Save predictions
        output_file = results_dir / f"predictions_{split_name}.csv"
        results_df.to_csv(output_file, index=False)
        logging.info(f"✓ Predictions saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Brain Age Prediction Pipeline (Inference Only)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python main.py \\
      --metadata_file data/IXI/IXI.xls \\
      --image_root data/IXI/IXI-T1 \\
      --model_path models/best_model.pt \\
      --output_root results

  # Skip existing outputs
  python main.py \\
      --metadata_file data/metadata.csv \\
      --image_root data/images \\
      --model_path best_model.pt \\
      --output_root results \\
      --skip_existing
        """
    )

    parser.add_argument('--metadata_file', required=True,
                       help='Path to metadata file (CSV/TSV/Excel)')
    parser.add_argument('--image_root', required=True,
                       help='Root directory containing MRI images')
    parser.add_argument('--model_path', required=True,
                       help='Path to pretrained model (.pt file)')
    parser.add_argument('--output_root', default='results',
                       help='Output directory (default: results)')
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip steps if outputs already exist')

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.metadata_file).exists():
        print(f"ERROR: Metadata file not found: {args.metadata_file}")
        sys.exit(1)

    if not Path(args.image_root).exists():
        print(f"ERROR: Image root not found: {args.image_root}")
        sys.exit(1)

    if not Path(args.model_path).exists():
        print(f"ERROR: Model file not found: {args.model_path}")
        sys.exit(1)

    # Run pipeline
    pipeline = BrainAgePipeline(
        metadata_file=args.metadata_file,
        image_root=args.image_root,
        model_path=args.model_path,
        output_root=args.output_root,
        skip_existing=args.skip_existing,
    )

    pipeline.run()


if __name__ == '__main__':
    main()