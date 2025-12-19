"""
Data Validation Utilities for Brain Age Prediction Pipeline

This module provides comprehensive validation for:
- Image file integrity
- Metadata completeness
- Preprocessing quality
- Dataset statistics

Usage:
    # Validate raw data
    from utils.validation import validate_raw_data
    validate_raw_data('raw_index.csv')
    
    # Validate preprocessed data
    from utils.validation import validate_preprocessed_data
    validate_preprocessed_data('preprocessed_index.csv')
    
    # Check single image
    from utils.validation import check_image_quality
    check_image_quality('path/to/image.nii.gz')
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import nibabel as nib
from dataclasses import dataclass
import warnings


@dataclass
class ValidationReport:
    """Container for validation results"""
    total_subjects: int = 0
    valid_subjects: int = 0
    failed_subjects: List[str] = None
    warnings: List[str] = None
    errors: List[str] = None
    statistics: Dict = None
    
    def __post_init__(self):
        if self.failed_subjects is None:
            self.failed_subjects = []
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []
        if self.statistics is None:
            self.statistics = {}
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_subjects == 0:
            return 0.0
        return 100 * self.valid_subjects / self.total_subjects
    
    @property
    def is_valid(self) -> bool:
        """Check if validation passed"""
        return len(self.errors) == 0 and self.valid_subjects > 0
    
    def print_summary(self) -> None:
        """Print validation summary"""
        print("\n" + "=" * 60)
        print("VALIDATION REPORT")
        print("=" * 60)
        print(f"Total subjects: {self.total_subjects}")
        print(f"Valid subjects: {self.valid_subjects} ({self.success_rate:.1f}%)")
        print(f"Failed subjects: {len(self.failed_subjects)}")
        
        if self.errors:
            print(f"\nERRORS ({len(self.errors)}):")
            for error in self.errors[:5]:  # Show first 5
                print(f"  - {error}")
            if len(self.errors) > 5:
                print(f"  ... and {len(self.errors) - 5} more")
        
        if self.warnings:
            print(f"\nWARNINGS ({len(self.warnings)}):")
            for warning in self.warnings[:5]:
                print(f"  - {warning}")
            if len(self.warnings) > 5:
                print(f"  ... and {len(self.warnings) - 5} more")
        
        if self.statistics:
            print("\nSTATISTICS:")
            for key, value in self.statistics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
        
        print("=" * 60)
        print(f"Status: {'✓ PASSED' if self.is_valid else '✗ FAILED'}")
        print("=" * 60 + "\n")


# ==================== Image Quality Checks ====================

def check_image_quality(
    image_path: str,
    expected_dims: int = 3,
    min_intensity: float = -1000,
    max_intensity: float = 10000,
) -> Tuple[bool, List[str]]:
    """
    Check quality of a single NIfTI image.
    
    Args:
        image_path: Path to NIfTI file
        expected_dims: Expected number of spatial dimensions
        min_intensity: Minimum valid intensity value
        max_intensity: Maximum valid intensity value
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    image_path = Path(image_path)
    
    # Check file exists
    if not image_path.exists():
        issues.append(f"File not found: {image_path}")
        return False, issues
    
    try:
        # Load image
        img = nib.load(str(image_path))
        data = img.get_fdata()
        
        # Check dimensions
        if data.ndim != expected_dims:
            issues.append(f"Wrong dimensions: expected {expected_dims}D, got {data.ndim}D")
        
        # Check for NaN or Inf
        if np.any(np.isnan(data)):
            issues.append("Contains NaN values")
        
        if np.any(np.isinf(data)):
            issues.append("Contains Inf values")
        
        # Check intensity range
        min_val = float(np.min(data))
        max_val = float(np.max(data))
        
        if min_val < min_intensity:
            issues.append(f"Intensity too low: min={min_val:.2f}")
        
        if max_val > max_intensity:
            issues.append(f"Intensity too high: max={max_val:.2f}")
        
        # Check if image is empty (all zeros)
        if np.all(data == 0):
            issues.append("Image is all zeros")
        
        # Check variance (constant images are suspicious)
        if np.var(data) < 1e-6:
            issues.append("Image has near-zero variance")
        
        # Check shape consistency
        shape = data.shape
        if min(shape) < 10:
            issues.append(f"Suspiciously small dimensions: {shape}")
        
    except Exception as e:
        issues.append(f"Failed to load image: {str(e)}")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def get_image_statistics(image_path: str) -> Dict:
    """
    Get detailed statistics for an image.
    
    Args:
        image_path: Path to NIfTI file
    
    Returns:
        Dictionary with statistics
    """
    try:
        img = nib.load(str(image_path))
        data = img.get_fdata()
        
        stats = {
            'shape': data.shape,
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'median': float(np.median(data)),
            'non_zero_fraction': float(np.count_nonzero(data) / data.size),
        }
        
        return stats
    
    except Exception as e:
        return {'error': str(e)}


# ==================== CSV Validation ====================

def validate_csv_structure(
    csv_path: str,
    required_columns: List[str] = None,
) -> Tuple[bool, List[str]]:
    """
    Validate CSV structure.
    
    Args:
        csv_path: Path to CSV file
        required_columns: List of required column names
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    csv_path = Path(csv_path)
    
    if required_columns is None:
        required_columns = ['ID', 'Age']
    
    # Check file exists
    if not csv_path.exists():
        issues.append(f"CSV file not found: {csv_path}")
        return False, issues
    
    try:
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Check not empty
        if len(df) == 0:
            issues.append("CSV is empty")
            return False, issues
        
        # Check required columns
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        # Check for duplicate IDs
        if 'ID' in df.columns:
            duplicates = df['ID'].duplicated()
            if duplicates.any():
                n_dups = duplicates.sum()
                dup_ids = df.loc[duplicates, 'ID'].tolist()[:5]
                issues.append(f"Found {n_dups} duplicate IDs: {dup_ids}")
        
        # Check Age column
        if 'Age' in df.columns:
            # Check for NaN
            if df['Age'].isna().any():
                n_nan = df['Age'].isna().sum()
                issues.append(f"Age column contains {n_nan} NaN values")
            
            # Check for invalid ages
            valid_ages = df['Age'].dropna()
            if len(valid_ages) > 0:
                if (valid_ages < 0).any() or (valid_ages > 120).any():
                    issues.append("Age column contains invalid values (<0 or >120)")
        
    except Exception as e:
        issues.append(f"Failed to load CSV: {str(e)}")
    
    is_valid = len(issues) == 0
    return is_valid, issues


# ==================== Raw Data Validation ====================

def validate_raw_data(
    csv_path: str,
    check_images: bool = True,
    sample_size: int = 10,
) -> ValidationReport:
    """
    Validate raw dataset index CSV.
    
    Args:
        csv_path: Path to raw index CSV
        check_images: Whether to check actual image files
        sample_size: Number of images to sample check
    
    Returns:
        ValidationReport object
    """
    report = ValidationReport()
    
    logging.info(f"Validating raw data: {csv_path}")
    
    # Check CSV structure
    is_valid, issues = validate_csv_structure(csv_path)
    if not is_valid:
        report.errors.extend(issues)
        return report
    
    # Load CSV
    df = pd.read_csv(csv_path)
    report.total_subjects = len(df)
    
    # Find image path columns
    path_cols = [col for col in df.columns if col.endswith('_path')]
    
    if not path_cols:
        report.errors.append("No image path columns found (expected columns like 'T1_path')")
        return report
    
    # Check for missing paths
    for col in path_cols:
        n_missing = (df[col].isna() | (df[col] == '')).sum()
        if n_missing > 0:
            pct = 100 * n_missing / len(df)
            report.warnings.append(f"{col}: {n_missing}/{len(df)} missing ({pct:.1f}%)")
    
    # Sample check images if requested
    if check_images and len(path_cols) > 0:
        sample_col = path_cols[0]  # Check first modality
        valid_paths = df[df[sample_col] != ''][sample_col]
        
        if len(valid_paths) > 0:
            sample = valid_paths.sample(n=min(sample_size, len(valid_paths)), random_state=42)
            
            n_valid_images = 0
            for img_path in sample:
                is_valid, issues = check_image_quality(img_path)
                if is_valid:
                    n_valid_images += 1
                else:
                    report.warnings.append(f"{Path(img_path).name}: {', '.join(issues)}")
            
            report.statistics['sample_image_quality'] = f"{n_valid_images}/{len(sample)} valid"
    
    # Age statistics
    if 'Age' in df.columns:
        ages = df['Age'].dropna()
        report.statistics['age_mean'] = float(ages.mean())
        report.statistics['age_std'] = float(ages.std())
        report.statistics['age_range'] = f"{float(ages.min()):.1f} - {float(ages.max()):.1f}"
    
    # Count valid subjects (have all path columns filled)
    has_all_paths = df[path_cols].apply(lambda row: all(row != '') and all(~row.isna()), axis=1)
    report.valid_subjects = has_all_paths.sum()
    
    return report


# ==================== Preprocessed Data Validation ====================

def validate_preprocessed_data(
    csv_path: str,
    check_registration: bool = True,
    sample_size: int = 10,
) -> ValidationReport:
    """
    Validate preprocessed dataset CSV.
    
    Args:
        csv_path: Path to preprocessed index CSV
        check_registration: Whether to check registration quality
        sample_size: Number of images to sample check
    
    Returns:
        ValidationReport object
    """
    report = ValidationReport()
    
    logging.info(f"Validating preprocessed data: {csv_path}")
    
    # Check CSV structure
    required_cols = ['ID', 'Age']
    is_valid, issues = validate_csv_structure(csv_path, required_cols)
    if not is_valid:
        report.errors.extend(issues)
        return report
    
    # Load CSV
    df = pd.read_csv(csv_path)
    report.total_subjects = len(df)
    
    # Find preprocessed path columns
    prep_cols = [col for col in df.columns if col.endswith('_preprocessed_path')]
    
    if not prep_cols:
        report.errors.append(
            "No preprocessed path columns found "
            "(expected columns like 'T1_preprocessed_path')"
        )
        return report
    
    # Check for missing preprocessed paths
    for col in prep_cols:
        n_missing = (df[col].isna() | (df[col] == '')).sum()
        if n_missing > 0:
            pct = 100 * n_missing / len(df)
            report.warnings.append(f"{col}: {n_missing}/{len(df)} missing ({pct:.1f}%)")
    
    # Check preprocessed images exist and quality
    if len(prep_cols) > 0:
        sample_col = prep_cols[0]
        valid_paths = df[df[sample_col] != ''][sample_col]
        
        if len(valid_paths) > 0:
            sample = valid_paths.sample(n=min(sample_size, len(valid_paths)), random_state=42)
            
            n_valid = 0
            shapes = []
            intensities = []
            
            for img_path in sample:
                is_valid, issues = check_image_quality(img_path)
                
                if is_valid:
                    n_valid += 1
                    stats = get_image_statistics(img_path)
                    if 'shape' in stats:
                        shapes.append(stats['shape'])
                        intensities.append((stats['min'], stats['max']))
                else:
                    report.warnings.append(f"{Path(img_path).name}: {', '.join(issues)}")
            
            report.statistics['sample_valid'] = f"{n_valid}/{len(sample)}"
            
            # Check shape consistency
            if shapes:
                unique_shapes = list(set(shapes))
                if len(unique_shapes) > 1:
                    report.warnings.append(f"Inconsistent shapes found: {unique_shapes}")
                else:
                    report.statistics['preprocessed_shape'] = str(shapes[0])
            
            # Check intensity normalization
            if intensities and check_registration:
                min_vals = [x[0] for x in intensities]
                max_vals = [x[1] for x in intensities]
                
                avg_min = np.mean(min_vals)
                avg_max = np.mean(max_vals)
                
                report.statistics['intensity_range'] = f"{avg_min:.2f} - {avg_max:.2f}"
                
                # Check if normalized to [0, 1] range (common for preprocessing)
                if avg_min < -0.1 or avg_max > 1.1:
                    report.warnings.append(
                        f"Images may not be normalized (range: {avg_min:.2f} - {avg_max:.2f})"
                    )
    
    # Count valid subjects
    has_all_paths = df[prep_cols].apply(lambda row: all(row != '') and all(~row.isna()), axis=1)
    report.valid_subjects = has_all_paths.sum()
    
    # Age statistics
    if 'Age' in df.columns:
        ages = df['Age'].dropna()
        report.statistics['age_mean'] = float(ages.mean())
        report.statistics['age_std'] = float(ages.std())
        report.statistics['age_range'] = f"{float(ages.min()):.1f} - {float(ages.max()):.1f}"
    
    return report


# ==================== Split Validation ====================

def validate_splits(
    train_csv: str,
    val_csv: str,
    test_csv: str,
) -> ValidationReport:
    """
    Validate train/val/test splits.
    
    Args:
        train_csv: Path to training CSV
        val_csv: Path to validation CSV
        test_csv: Path to test CSV
    
    Returns:
        ValidationReport object
    """
    report = ValidationReport()
    
    logging.info("Validating dataset splits")
    
    try:
        # Load splits
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        test_df = pd.read_csv(test_csv)
        
        # Check all have required columns
        for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            if 'ID' not in df.columns or 'Age' not in df.columns:
                report.errors.append(f"{name} split missing required columns")
                return report
        
        # Check for ID overlap
        train_ids = set(train_df['ID'])
        val_ids = set(val_df['ID'])
        test_ids = set(test_df['ID'])
        
        train_val_overlap = train_ids & val_ids
        train_test_overlap = train_ids & test_ids
        val_test_overlap = val_ids & test_ids
        
        if train_val_overlap:
            report.errors.append(f"Train/Val overlap: {len(train_val_overlap)} IDs")
        
        if train_test_overlap:
            report.errors.append(f"Train/Test overlap: {len(train_test_overlap)} IDs")
        
        if val_test_overlap:
            report.errors.append(f"Val/Test overlap: {len(val_test_overlap)} IDs")
        
        # Statistics
        total = len(train_df) + len(val_df) + len(test_df)
        report.total_subjects = total
        report.valid_subjects = total
        
        report.statistics['train_size'] = len(train_df)
        report.statistics['val_size'] = len(val_df)
        report.statistics['test_size'] = len(test_df)
        report.statistics['train_ratio'] = f"{100*len(train_df)/total:.1f}%"
        report.statistics['val_ratio'] = f"{100*len(val_df)/total:.1f}%"
        report.statistics['test_ratio'] = f"{100*len(test_df)/total:.1f}%"
        
        # Age distribution comparison
        train_age = train_df['Age'].mean()
        val_age = val_df['Age'].mean()
        test_age = test_df['Age'].mean()
        
        report.statistics['train_age_mean'] = f"{train_age:.1f}"
        report.statistics['val_age_mean'] = f"{val_age:.1f}"
        report.statistics['test_age_mean'] = f"{test_age:.1f}"
        
        # Check if age distributions are similar (good stratification)
        age_variance = np.var([train_age, val_age, test_age])
        if age_variance > 5.0:
            report.warnings.append(
                f"Age distributions vary significantly across splits (variance={age_variance:.2f}). "
                "Consider using --stratify_age"
            )
        
    except Exception as e:
        report.errors.append(f"Failed to validate splits: {str(e)}")
    
    return report


# ==================== Command-line Interface ====================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Data Validation Tools',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate raw data index
  python -m utils.validation --raw raw_index.csv
  
  # Validate preprocessed data
  python -m utils.validation --preprocessed preprocessed/preprocessed_index.csv
  
  # Validate splits
  python -m utils.validation --splits splits/train_split.csv splits/val_split.csv splits/test_split.csv
  
  # Check single image
  python -m utils.validation --image path/to/image.nii.gz
        """
    )
    
    parser.add_argument('--raw', help='Validate raw data index CSV')
    parser.add_argument('--preprocessed', help='Validate preprocessed data CSV')
    parser.add_argument('--splits', nargs=3, metavar=('TRAIN', 'VAL', 'TEST'),
                        help='Validate train/val/test splits')
    parser.add_argument('--image', help='Check single image quality')
    parser.add_argument('--sample_size', type=int, default=10,
                        help='Number of images to sample check (default: 10)')
    parser.add_argument('--no_image_check', action='store_true',
                        help='Skip image quality checks')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    if args.raw:
        report = validate_raw_data(
            args.raw,
            check_images=not args.no_image_check,
            sample_size=args.sample_size,
        )
        report.print_summary()
        return 0 if report.is_valid else 1
    
    elif args.preprocessed:
        report = validate_preprocessed_data(
            args.preprocessed,
            check_registration=not args.no_image_check,
            sample_size=args.sample_size,
        )
        report.print_summary()
        return 0 if report.is_valid else 1
    
    elif args.splits:
        report = validate_splits(args.splits[0], args.splits[1], args.splits[2])
        report.print_summary()
        return 0 if report.is_valid else 1
    
    elif args.image:
        is_valid, issues = check_image_quality(args.image)
        print(f"\nImage: {args.image}")
        if is_valid:
            print("✓ Image is valid")
            stats = get_image_statistics(args.image)
            print("\nStatistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        else:
            print("✗ Image has issues:")
            for issue in issues:
                print(f"  - {issue}")
        return 0 if is_valid else 1
    
    else:
        parser.print_help()
        return 0


if __name__ == '__main__':
    exit(main())