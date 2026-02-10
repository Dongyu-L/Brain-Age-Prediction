"""
Universal Dataset Splitter

Splits preprocessed dataset into train/val/test sets with stratification options.

Usage:
    # Basic usage (60/20/20 split)
    python data_splitter.py \
        --input_csv preprocessed_index.csv \
        --output_dir splits \
        --ratios 0.6 0.2 0.2
    
    # With age stratification
    python data_splitter.py \
        --input_csv preprocessed_index.csv \
        --output_dir splits \
        --ratios 0.6 0.2 0.2 \
        --stratify_age \
        --age_bins 5
    
    # With random seed for reproducibility
    python data_splitter.py \
        --input_csv preprocessed_index.csv \
        --output_dir splits \
        --ratios 0.6 0.2 0.2 \
        --seed 42
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime


class UniversalSplitter:
    """Universal dataset splitter with stratification support"""

    def __init__(
        self,
        input_csv: str,
        output_dir: str,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        stratify_age: bool = False,
        stratify_gender: bool = False,
        age_bins: int = 5,
        seed: int = 42,
        require_complete: bool = True,
    ):
        """
        Args:
            input_csv: Path to preprocessed CSV (from data_preprocessor.py)
            output_dir: Output directory for split CSVs
            train_ratio: Training set ratio (default: 0.6)
            val_ratio: Validation set ratio (default: 0.2)
            test_ratio: Test set ratio (default: 0.2)
            stratify_age: Whether to stratify by age groups
            stratify_gender: Whether to stratify by gender
            age_bins: Number of age bins for stratification
            seed: Random seed for reproducibility
            require_complete: Only include subjects with all preprocessed modalities
        """
        self.input_csv = Path(input_csv)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.stratify_age = stratify_age
        self.stratify_gender = stratify_gender
        self.age_bins = age_bins
        self.seed = seed
        self.require_complete = require_complete

        self.df: Optional[pd.DataFrame] = None
        self.preprocessed_cols: List[str] = []
        self.has_gender: bool = False

        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(
                f"Ratios must sum to 1.0, got {total_ratio}\n"
                f"train={train_ratio}, val={val_ratio}, test={test_ratio}"
            )
    
    def run(self) -> Dict[str, pd.DataFrame]:
        """
        Run dataset splitting pipeline.
        
        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        logging.info("=" * 60)
        logging.info("Starting Dataset Splitting")
        logging.info("=" * 60)
        
        # Step 1: Load and validate
        self._load_csv()
        self._validate_data()
        
        # Step 2: Filter subjects
        self._filter_subjects()
        
        # Step 3: Split dataset
        splits = self._split_dataset()
        
        # Step 4: Save splits
        self._save_splits(splits)
        
        # Step 5: Summary
        self._print_summary(splits)
        
        return splits
    
    # ========== Step 1: Load and Validate ==========
    
    def _load_csv(self) -> None:
        """Load input CSV"""
        logging.info(f"Loading input CSV: {self.input_csv}")
        
        if not self.input_csv.exists():
            raise FileNotFoundError(
                f"Input CSV not found: {self.input_csv}\n"
                f"Suggestion: Run data_preprocessor.py first to create preprocessed_index.csv"
            )
        
        self.df = pd.read_csv(self.input_csv)
        logging.info(f"  Loaded {len(self.df)} subjects")
        
        # Validate required columns
        required_cols = {'ID', 'Age'}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(
                f"CSV missing required columns: {required_cols - set(self.df.columns)}\n"
                f"Available columns: {list(self.df.columns)}\n"
                f"Suggestion: Ensure CSV was created by data_preprocessor.py"
            )
    
    def _validate_data(self) -> None:
        """Validate data quality"""
        logging.info("Validating data...")

        # Find preprocessed path columns
        self.preprocessed_cols = [
            col for col in self.df.columns
            if col.endswith('_preprocessed_path')
        ]

        if not self.preprocessed_cols:
            raise ValueError(
                f"No preprocessed path columns found in CSV.\n"
                f"Expected columns like 'T1_preprocessed_path', 'T2_preprocessed_path', etc.\n"
                f"Available columns: {list(self.df.columns)}\n"
                f"Suggestion: Ensure CSV was created by data_preprocessor.py"
            )

        logging.info(f"  Found {len(self.preprocessed_cols)} preprocessed modalities:")
        for col in self.preprocessed_cols:
            modality = col.replace('_preprocessed_path', '')
            # Fixed: use & instead of + to correctly count non-empty and non-null values
            n_complete = ((self.df[col] != '') & (~self.df[col].isna())).sum()
            pct = 100 * n_complete / len(self.df)
            logging.info(f"    {modality}: {n_complete}/{len(self.df)} ({pct:.1f}%)")

        # Check for gender column
        self.has_gender = 'Gender' in self.df.columns
        if self.has_gender:
            n_valid_gender = (self.df['Gender'] >= 0).sum()
            n_male = (self.df['Gender'] == 1).sum()
            n_female = (self.df['Gender'] == 0).sum()
            logging.info(f"  Gender data available: {n_valid_gender}/{len(self.df)}")
            logging.info(f"    Male: {n_male}, Female: {n_female}")
            if self.stratify_gender and n_valid_gender == 0:
                logging.warning("Gender stratification requested but no valid gender data found. "
                              "Falling back to age-only stratification.")
                self.stratify_gender = False
        elif self.stratify_gender:
            logging.warning("Gender stratification requested but Gender column not found. "
                          "Falling back to age-only stratification.")
            self.stratify_gender = False

        # Validate ages (warn but don't filter - allow missing ages for inference)
        missing_ages = self.df['Age'].isna().sum()
        out_of_range = ((self.df['Age'] < 0) | (self.df['Age'] > 120)).sum()

        if missing_ages > 0:
            logging.warning(
                f"Found {missing_ages} subjects with missing ages. "
                f"These will be kept for inference but excluded from stratification."
            )
        if out_of_range > 0:
            logging.warning(
                f"Found {out_of_range} subjects with out-of-range ages (not in 0-120). "
            )

        # Count subjects with valid ages for reference
        valid_ages = self.df[
            (self.df['Age'] >= 0) &
            (self.df['Age'] <= 120) &
            (~self.df['Age'].isna())
        ]
        logging.info(f"  Subjects with valid ages: {len(valid_ages)}/{len(self.df)}")
    
    def _filter_subjects(self) -> None:
        """Filter subjects based on requirements"""
        if not self.require_complete:
            logging.info("No filtering applied (require_complete=False)")
            return
        
        logging.info("Filtering subjects with incomplete preprocessed data...")
        
        before = len(self.df)
        
        # Keep only subjects with all preprocessed modalities
        for col in self.preprocessed_cols:
            self.df = self.df[
                (self.df[col] != '') & 
                (~self.df[col].isna())
            ]
        
        after = len(self.df)
        removed = before - after
        
        if removed > 0:
            pct = 100 * removed / before
            logging.info(f"  Removed {removed}/{before} subjects with incomplete data ({pct:.1f}%)")
        else:
            logging.info(f"  All {after} subjects have complete data")
        
        if after == 0:
            raise ValueError(
                "No subjects remain after filtering.\n"
                f"Suggestion: Set --no_require_complete to include subjects with partial data, "
                f"or check preprocessing logs for failures."
            )
    
    # ========== Step 3: Split Dataset ==========
    
    def _split_dataset(self) -> Dict[str, pd.DataFrame]:
        """Split dataset into train/val/test"""
        logging.info("")
        logging.info("Splitting dataset...")
        logging.info(f"  Ratios: train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio}")
        logging.info(f"  Random seed: {self.seed}")
        logging.info(f"  Stratify by age: {self.stratify_age}")
        logging.info(f"  Stratify by gender: {self.stratify_gender}")

        # Set random seed
        np.random.seed(self.seed)

        if self.stratify_age or self.stratify_gender:
            return self._stratified_split()
        else:
            return self._random_split()
    
    def _random_split(self) -> Dict[str, pd.DataFrame]:
        """Random split without stratification"""
        # Get unique subject IDs and shuffle
        subject_ids = self.df['ID'].unique().tolist()
        np.random.shuffle(subject_ids)
        
        n_total = len(subject_ids)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)
        
        # Split IDs
        train_ids = subject_ids[:n_train]
        val_ids = subject_ids[n_train:n_train + n_val]
        test_ids = subject_ids[n_train + n_val:]
        
        # Create dataframes
        train_df = self.df[self.df['ID'].isin(train_ids)].copy()
        val_df = self.df[self.df['ID'].isin(val_ids)].copy()
        test_df = self.df[self.df['ID'].isin(test_ids)].copy()
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df,
        }
    
    def _stratified_split(self) -> Dict[str, pd.DataFrame]:
        """Stratified split by age groups and/or gender"""
        strat_cols = []

        # Create age bins if stratifying by age
        if self.stratify_age:
            logging.info(f"  Using {self.age_bins} age bins for stratification")
            self.df['age_bin'] = pd.cut(
                self.df['Age'],
                bins=self.age_bins,
                labels=False,
                duplicates='drop'
            )
            strat_cols.append('age_bin')

        # Use gender for stratification if available
        if self.stratify_gender and self.has_gender:
            logging.info("  Using gender for stratification")
            # Create gender bin (only valid genders)
            self.df['gender_bin'] = self.df['Gender'].apply(
                lambda x: int(x) if x >= 0 else -1
            )
            strat_cols.append('gender_bin')

        # Create combined stratification key
        if len(strat_cols) > 1:
            self.df['strat_key'] = self.df[strat_cols].apply(
                lambda x: '_'.join(str(v) for v in x), axis=1
            )
        elif len(strat_cols) == 1:
            self.df['strat_key'] = self.df[strat_cols[0]].astype(str)
        else:
            # Fallback to random split
            return self._random_split()

        # Initialize split dataframes
        train_dfs = []
        val_dfs = []
        test_dfs = []

        # Split within each stratification group
        for strat_key in self.df['strat_key'].unique():
            if pd.isna(strat_key) or '-1' in str(strat_key) or 'nan' in str(strat_key).lower():
                # Handle subjects with unknown gender/age separately (put in random split)
                continue

            group_df = self.df[self.df['strat_key'] == strat_key]
            subject_ids = group_df['ID'].unique().tolist()
            np.random.shuffle(subject_ids)

            n_total = len(subject_ids)
            n_train = max(1, int(n_total * self.train_ratio))
            n_val = max(0, int(n_total * self.val_ratio))

            # Ensure at least some subjects in each split if possible
            if n_total >= 3:
                n_val = max(1, n_val)

            # Split IDs
            train_ids = subject_ids[:n_train]
            val_ids = subject_ids[n_train:n_train + n_val]
            test_ids = subject_ids[n_train + n_val:]

            # Add to split dataframes
            train_dfs.append(group_df[group_df['ID'].isin(train_ids)])
            val_dfs.append(group_df[group_df['ID'].isin(val_ids)])
            test_dfs.append(group_df[group_df['ID'].isin(test_ids)])

        # Handle subjects with unknown stratification (e.g., unknown gender or missing age)
        unknown_mask = (
            self.df['strat_key'].isna() |
            self.df['strat_key'].str.contains('-1', na=False) |
            self.df['strat_key'].str.contains('nan', case=False, na=False)
        )
        if unknown_mask.sum() > 0:
            unknown_df = self.df[unknown_mask]
            subject_ids = unknown_df['ID'].unique().tolist()
            np.random.shuffle(subject_ids)

            n_total = len(subject_ids)
            n_train = max(1, int(n_total * self.train_ratio))
            n_val = max(0, int(n_total * self.val_ratio))

            train_ids = subject_ids[:n_train]
            val_ids = subject_ids[n_train:n_train + n_val]
            test_ids = subject_ids[n_train + n_val:]

            train_dfs.append(unknown_df[unknown_df['ID'].isin(train_ids)])
            val_dfs.append(unknown_df[unknown_df['ID'].isin(val_ids)])
            test_dfs.append(unknown_df[unknown_df['ID'].isin(test_ids)])

        # Concatenate all groups
        cols_to_drop = ['age_bin', 'gender_bin', 'strat_key']
        cols_to_drop = [c for c in cols_to_drop if c in self.df.columns]

        train_df = pd.concat(train_dfs, ignore_index=True).drop(columns=cols_to_drop, errors='ignore')
        val_df = pd.concat(val_dfs, ignore_index=True).drop(columns=cols_to_drop, errors='ignore')
        test_df = pd.concat(test_dfs, ignore_index=True).drop(columns=cols_to_drop, errors='ignore')

        return {
            'train': train_df,
            'val': val_df,
            'test': test_df,
        }
    
    # ========== Step 4: Save Splits ==========
    
    def _save_splits(self, splits: Dict[str, pd.DataFrame]) -> None:
        """Save split CSVs and metadata"""
        logging.info("")
        logging.info(f"Saving splits to: {self.output_dir}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CSV files
        for split_name, split_df in splits.items():
            output_path = self.output_dir / f"{split_name}_split.csv"
            split_df.to_csv(output_path, index=False)
            logging.info(f"  {split_name}: {len(split_df)} subjects -> {output_path.name}")
        
        # Save split metadata
        metadata = {
            'created_date': datetime.now().isoformat(),
            'input_csv': str(self.input_csv),
            'total_subjects': len(self.df),
            'ratios': {
                'train': self.train_ratio,
                'val': self.val_ratio,
                'test': self.test_ratio,
            },
            'stratify_age': self.stratify_age,
            'age_bins': self.age_bins if self.stratify_age else None,
            'random_seed': self.seed,
            'require_complete': self.require_complete,
            'split_sizes': {
                'train': len(splits['train']),
                'val': len(splits['val']),
                'test': len(splits['test']),
            },
            'preprocessed_modalities': [
                col.replace('_preprocessed_path', '') 
                for col in self.preprocessed_cols
            ],
        }
        
        import json
        metadata_path = self.output_dir / "split_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logging.info(f"  Metadata saved: {metadata_path.name}")
    
    # ========== Step 5: Summary ==========
    
    def _print_summary(self, splits: Dict[str, pd.DataFrame]) -> None:
        """Print detailed summary"""
        logging.info("")
        logging.info("=" * 60)
        logging.info("SPLIT SUMMARY")
        logging.info("=" * 60)

        total_subjects = sum(len(df) for df in splits.values())

        for split_name in ['train', 'val', 'test']:
            split_df = splits[split_name]
            n_subjects = len(split_df)
            pct = 100 * n_subjects / total_subjects if total_subjects > 0 else 0

            # Count subjects with valid ages
            valid_ages = split_df['Age'].dropna()
            n_with_age = len(valid_ages)
            n_without_age = n_subjects - n_with_age

            logging.info(f"\n{split_name.upper()}:")
            logging.info(f"  Subjects: {n_subjects} ({pct:.1f}%)")

            if n_with_age > 0:
                age_mean = valid_ages.mean()
                age_std = valid_ages.std()
                age_min = valid_ages.min()
                age_max = valid_ages.max()
                logging.info(f"  Age (n={n_with_age}): {age_mean:.1f} ± {age_std:.1f} (range: {age_min:.1f}-{age_max:.1f})")
            if n_without_age > 0:
                logging.info(f"  Subjects without age: {n_without_age}")

            # Gender statistics if available
            if self.has_gender and 'Gender' in split_df.columns:
                n_male = (split_df['Gender'] == 1).sum()
                n_female = (split_df['Gender'] == 0).sum()
                n_unknown = (split_df['Gender'] < 0).sum()
                logging.info(f"  Gender: Male={n_male}, Female={n_female}, Unknown={n_unknown}")

        logging.info("")
        logging.info(f"Total subjects: {total_subjects}")
        logging.info("=" * 60)

        # Check for age distribution balance
        if self.stratify_age:
            self._check_stratification_quality(splits)

        # Check for gender distribution balance
        if self.stratify_gender and self.has_gender:
            self._check_gender_stratification_quality(splits)

    def _check_gender_stratification_quality(self, splits: Dict[str, pd.DataFrame]) -> None:
        """Check if gender stratification worked well"""
        logging.info("\nGender distribution check:")

        gender_ratios = []
        for split_name, split_df in splits.items():
            if 'Gender' not in split_df.columns:
                continue
            n_male = (split_df['Gender'] == 1).sum()
            n_female = (split_df['Gender'] == 0).sum()
            n_total = n_male + n_female
            if n_total > 0:
                male_ratio = n_male / n_total
                gender_ratios.append(male_ratio)
                logging.info(f"  {split_name}: Male ratio = {male_ratio*100:.1f}%")

        # Calculate variance between splits
        if len(gender_ratios) > 1:
            variance = np.var(gender_ratios)
            if variance < 0.01:
                logging.info(f"  Stratification quality: GOOD (variance={variance:.4f})")
            else:
                logging.warning(f"  Stratification quality: MODERATE (variance={variance:.4f})")
    
    def _check_stratification_quality(self, splits: Dict[str, pd.DataFrame]) -> None:
        """Check if stratification worked well"""
        logging.info("\nAge distribution check (subjects with valid ages only):")

        means = []
        for split_name, split_df in splits.items():
            valid_ages = split_df['Age'].dropna()
            if len(valid_ages) > 0:
                mean_age = valid_ages.mean()
                means.append(mean_age)
                logging.info(f"  {split_name}: mean age = {mean_age:.2f} (n={len(valid_ages)})")
            else:
                logging.info(f"  {split_name}: no valid ages")

        # Calculate variance between splits
        if len(means) > 1:
            variance = np.var(means)
            if variance < 1.0:
                logging.info(f"  Stratification quality: GOOD (variance={variance:.2f})")
            else:
                logging.warning(f"  Stratification quality: MODERATE (variance={variance:.2f})")


def main():
    parser = argparse.ArgumentParser(
        description='Universal Dataset Splitter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic 60/20/20 split
  python data_splitter.py \\
    --input_csv preprocessed/preprocessed_index.csv \\
    --output_dir splits \\
    --ratios 0.6 0.2 0.2
  
  # With age stratification (recommended for brain age)
  python data_splitter.py \\
    --input_csv preprocessed/preprocessed_index.csv \\
    --output_dir splits \\
    --ratios 0.6 0.2 0.2 \\
    --stratify_age \\
    --age_bins 5
  
  # Custom ratios (70/15/15)
  python data_splitter.py \\
    --input_csv preprocessed/preprocessed_index.csv \\
    --output_dir splits \\
    --ratios 0.7 0.15 0.15 \\
    --seed 123
  
  # Include subjects with partial data
  python data_splitter.py \\
    --input_csv preprocessed/preprocessed_index.csv \\
    --output_dir splits \\
    --ratios 0.6 0.2 0.2 \\
    --no_require_complete
        """
    )
    
    # Required arguments
    parser.add_argument('--input_csv', required=True,
                        help='Input CSV from data_preprocessor.py')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for split CSVs')
    
    # Split ratios
    parser.add_argument('--ratios', type=float, nargs=3,
                        default=[0.6, 0.2, 0.2],
                        metavar=('TRAIN', 'VAL', 'TEST'),
                        help='Split ratios (must sum to 1.0, default: 0.6 0.2 0.2)')
    
    # Stratification options
    parser.add_argument('--stratify_age', action='store_true',
                        help='Stratify splits by age groups (recommended for brain age)')
    parser.add_argument('--stratify_gender', action='store_true',
                        help='Stratify splits by gender (ensures balanced gender distribution)')
    parser.add_argument('--age_bins', type=int, default=5,
                        help='Number of age bins for stratification (default: 5)')
    
    # Other options
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--no_require_complete', action='store_true',
                        help='Include subjects with incomplete preprocessed data')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s: %(message)s'
    )
    
    # Validate ratios
    ratios_sum = sum(args.ratios)
    if not np.isclose(ratios_sum, 1.0):
        logging.error(f"Ratios must sum to 1.0, got {ratios_sum}")
        logging.error(f"Provided: train={args.ratios[0]}, val={args.ratios[1]}, test={args.ratios[2]}")
        return 1
    
    # Create splitter
    splitter = UniversalSplitter(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        train_ratio=args.ratios[0],
        val_ratio=args.ratios[1],
        test_ratio=args.ratios[2],
        stratify_age=args.stratify_age,
        stratify_gender=args.stratify_gender,
        age_bins=args.age_bins,
        seed=args.seed,
        require_complete=not args.no_require_complete,
    )
    
    # Run splitting
    try:
        splitter.run()
    except Exception as e:
        logging.error(f"\nFATAL ERROR: {str(e)}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())