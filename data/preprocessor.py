"""
Universal MRI Preprocessing Pipeline

A flexible preprocessing pipeline that works with any dataset structure.
Performs N4 bias correction and registration to standard template space.

Usage:
    # Basic usage (auto-detect modalities)
    python data_preprocessor.py \
        --input_csv IXI_index.csv \
        --output_dir preprocessed \
        --template MNI152_T1_1mm.nii.gz
    
    # Multi-modality with specific columns
    python data_preprocessor.py \
        --input_csv IXI_index.csv \
        --output_dir preprocessed \
        --template MNI152_T1_1mm.nii.gz \
        --modalities T1 T2
    
    # Process subset (useful for batch processing)
    python data_preprocessor.py \
        --input_csv IXI_index.csv \
        --output_dir preprocessed \
        --template MNI152_T1_1mm.nii.gz \
        --start_row 0 \
        --end_row 100

Output Structure:
    preprocessed/
    ├── IXI/                    # Dataset name from CSV filename
    │   ├── T1/                 # All T1 images for IXI dataset
    │   │   ├── 001_T1_MNI.nii.gz
    │   │   ├── 002_T1_MNI.nii.gz
    │   │   └── ...
    │   ├── T2/                 # All T2 images for IXI dataset
    │   │   ├── 001_T2_MNI.nii.gz
    │   │   └── ...
    │   ├── metadata/           # Processing metadata
    │   │   ├── 001_metadata.json
    │   │   └── ...
    │   └── IXI_preprocessed_index.csv  # Updated CSV with output paths
    └── logs/                   # Processing logs
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import ants
from datetime import datetime
import json


class UniversalPreprocessor:
    """Universal MRI preprocessing pipeline using ANTsPy"""
    
    def __init__(
        self,
        input_csv: str,
        output_dir: str,
        template_path: str,
        modalities: Optional[List[str]] = None,
        registration_type: str = "Rigid",
        n4_iterations: List[int] = [50, 50, 50, 50],
        skip_existing: bool = True,
        save_intermediate: bool = False,
    ):
        """
        Args:
            input_csv: Path to index CSV (from data_indexer.py)
            output_dir: Output directory for preprocessed images
            template_path: Path to template image (e.g., MNI152)
            modalities: List of modalities to process (e.g., ['T1', 'T2'])
                       If None, auto-detect from CSV columns
            registration_type: ANTs registration type ('Rigid', 'Affine', 'SyN')
            n4_iterations: Number of N4 iterations per level
            skip_existing: Skip subjects that already have output files
            save_intermediate: Save intermediate steps (N4 only, before registration)
        """
        self.input_csv = Path(input_csv)
        self.output_dir = Path(output_dir)
        self.template_path = Path(template_path)
        self.modalities = modalities
        self.registration_type = registration_type
        self.n4_iterations = n4_iterations
        self.skip_existing = skip_existing
        self.save_intermediate = save_intermediate
        
        # Extract dataset name from CSV filename (e.g., 'IXI_index.csv' -> 'IXI')
        self.dataset_name = self.input_csv.stem.split('_')[0]
        
        self.df: Optional[pd.DataFrame] = None
        self.template_img: Optional[ants.ANTsImage] = None
        self.stats: Dict = {
            'total': 0,
            'processed': 0,
            'skipped': 0,
            'failed': 0,
            'start_time': None,
            'end_time': None,
        }
        
    def run(
        self,
        start_row: int = 0,
        end_row: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Run preprocessing pipeline.
        
        Args:
            start_row: Start row index (inclusive)
            end_row: End row index (exclusive), None = process all
            
        Returns:
            DataFrame with preprocessed image paths added
        """
        logging.info("=" * 60)
        logging.info("Starting Universal MRI Preprocessing Pipeline")
        logging.info("=" * 60)
        
        self.stats['start_time'] = datetime.now()
        
        # Step 1: Load and validate inputs
        self._load_csv()
        self._load_template()
        self._detect_modalities()
        self._validate_inputs()
        
        # Step 2: Setup output directory
        self._setup_output_dir()
        
        # Step 3: Process subjects
        self._process_batch(start_row, end_row)
        
        # Step 4: Generate output CSV
        output_df = self._create_output_csv()
        
        # Step 5: Summary
        self._print_summary()
        
        return output_df
    
    # ========== Step 1: Load and Validate ==========
    
    def _load_csv(self) -> None:
        """Load input CSV"""
        logging.info(f"Loading input CSV: {self.input_csv}")
        
        if not self.input_csv.exists():
            raise FileNotFoundError(
                f"Input CSV not found: {self.input_csv}\n"
                f"Suggestion: Run data_indexer.py first to create the index CSV."
            )
        
        self.df = pd.read_csv(self.input_csv)
        logging.info(f"  Loaded {len(self.df)} subjects")
        
        # Validate required columns
        required_cols = {'ID', 'Age'}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(
                f"CSV missing required columns: {required_cols - set(self.df.columns)}\n"
                f"Available columns: {list(self.df.columns)}\n"
                f"Suggestion: Ensure CSV was created by data_indexer.py"
            )
    
    def _load_template(self) -> None:
        """Load template image"""
        logging.info(f"Loading template: {self.template_path}")
        
        if not self.template_path.exists():
            raise FileNotFoundError(
                f"Template file not found: {self.template_path}\n"
                f"Suggestion: Download a standard template like MNI152_T1_1mm.nii.gz"
            )
        
        try:
            self.template_img = ants.image_read(str(self.template_path))
            logging.info(f"  Template shape: {self.template_img.shape}")
            logging.info(f"  Template spacing: {self.template_img.spacing}")
        except Exception as e:
            raise ValueError(
                f"Failed to load template: {str(e)}\n"
                f"Suggestion: Ensure template is a valid NIfTI image."
            )
    
    def _detect_modalities(self) -> None:
        """Auto-detect modalities from CSV columns"""
        if self.modalities is not None:
            logging.info(f"Using specified modalities: {self.modalities}")
            return
        
        # Find columns ending with '_path'
        path_cols = [col for col in self.df.columns if col.endswith('_path')]
        
        if not path_cols:
            raise ValueError(
                f"No modality columns found in CSV.\n"
                f"Expected columns like 'T1_path', 'T2_path', etc.\n"
                f"Available columns: {list(self.df.columns)}\n"
                f"Suggestion: Ensure CSV was created by data_indexer.py"
            )
        
        # Extract modality names (e.g., 'T1_path' -> 'T1')
        self.modalities = [col.replace('_path', '') for col in path_cols]
        logging.info(f"Auto-detected modalities: {self.modalities}")
    
    def _validate_inputs(self) -> None:
        """Validate that image paths exist"""
        logging.info("Validating image paths...")
        
        missing_counts = {}
        for modality in self.modalities:
            col_name = f'{modality}_path'
            
            if col_name not in self.df.columns:
                raise ValueError(
                    f"Modality column '{col_name}' not found in CSV.\n"
                    f"Available columns: {list(self.df.columns)}\n"
                    f"Suggestion: Check modality names or re-run data_indexer.py"
                )
            
            # Count missing paths
            n_missing = (self.df[col_name] == '').sum() + self.df[col_name].isna().sum()
            missing_counts[modality] = n_missing
            
            if n_missing > 0:
                pct = 100 * n_missing / len(self.df)
                logging.warning(f"  {modality}: {n_missing}/{len(self.df)} missing ({pct:.1f}%)")
            else:
                logging.info(f"  {modality}: all paths present")
        
        # Check if at least one modality has all paths
        complete_modalities = [m for m, count in missing_counts.items() if count == 0]
        if not complete_modalities:
            raise ValueError(
                f"No modality has complete paths for all subjects.\n"
                f"Missing counts: {missing_counts}\n"
                f"Suggestion: Filter subjects with complete data or fix missing paths."
            )
    
    def _setup_output_dir(self) -> None:
        """Create output directory structure"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log directory
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Setup file logging
        log_file = log_dir / f"preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
        
        logging.info(f"Output directory: {self.output_dir}")
        logging.info(f"Log file: {log_file}")
    
    # ========== Step 3: Process Batch ==========
    
    def _process_batch(self, start_row: int, end_row: Optional[int]) -> None:
        """Process a batch of subjects"""
        n_rows = len(self.df)
        if end_row is None or end_row > n_rows:
            end_row = n_rows
        
        subset = self.df.iloc[start_row:end_row]
        
        logging.info("")
        logging.info("=" * 60)
        logging.info(f"Processing rows {start_row} to {end_row-1} ({len(subset)} subjects)")
        logging.info("=" * 60)
        
        self.stats['total'] = len(subset)
        
        for idx, row in subset.iterrows():
            try:
                success = self._process_subject(idx, row)
                if success:
                    self.stats['processed'] += 1
                else:
                    self.stats['skipped'] += 1
            except Exception as e:
                self.stats['failed'] += 1
                logging.error(f"[{idx}] Subject {row['ID']}: {str(e)}")
    
    def _process_subject(self, idx: int, row: pd.Series) -> bool:
        """
        Process a single subject.
        
        Returns:
            True if processed, False if skipped
        """
        subject_id = str(row['ID'])
        
        # Check if already processed (check all modalities)
        if self.skip_existing and self._is_subject_complete(subject_id):
            logging.info(f"[{idx}] {subject_id}: Already processed, skipping")
            return False
        
        logging.info(f"[{idx}] Processing {subject_id}")
        
        # Process each modality
        for modality in self.modalities:
            col_name = f'{modality}_path'
            img_path = row[col_name]
            
            # Skip if path is missing
            if pd.isna(img_path) or img_path == '':
                logging.info(f"  {modality}: No image path, skipping")
                continue
            
            img_path = Path(img_path)
            if not img_path.exists():
                logging.warning(f"  {modality}: File not found: {img_path}")
                continue
            
            logging.info(f"  {modality}: {img_path.name}")
            
            # Run preprocessing pipeline
            self._preprocess_single_image(
                img_path=img_path,
                subject_id=subject_id,
                modality=modality,
            )
        
        # Save processing metadata
        self._save_metadata(subject_id, row)
        
        return True
    
    def _preprocess_single_image(
        self,
        img_path: Path,
        output_dir: Path,
        subject_id: str,
        modality: str,
    ) -> None:
        """
        Preprocess a single image: N4 correction + registration.
        
        Args:
            img_path: Path to input image
            output_dir: Output directory for this subject
            subject_id: Subject ID
            modality: Modality name (e.g., 'T1', 'T2')
        """
        # Output filenames
        n4_output = output_dir / f"{subject_id}_{modality}_N4.nii.gz"
        final_output = output_dir / f"{subject_id}_{modality}_MNI.nii.gz"
        transform_prefix = output_dir / f"{subject_id}_{modality}_"
        
        # Step 1: Read image
        try:
            img = ants.image_read(str(img_path))
        except Exception as e:
            raise ValueError(f"Failed to read image: {str(e)}")
        
        # Step 2: N4 bias field correction
        logging.info(f"    - N4 bias correction")
        try:
            img_n4 = ants.n4_bias_field_correction(
                img,
                convergence={'iters': self.n4_iterations, 'tol': 1e-7},
            )
        except Exception as e:
            raise ValueError(f"N4 correction failed: {str(e)}")
        
        # Save N4 result if requested
        if self.save_intermediate:
            ants.image_write(img_n4, str(n4_output))
            logging.info(f"    - Saved N4: {n4_output.name}")
        
        # Step 3: Registration to template
        logging.info(f"    - Registration ({self.registration_type})")
        try:
            reg = ants.registration(
                fixed=self.template_img,
                moving=img_n4,
                type_of_transform=self.registration_type,
                outprefix=str(transform_prefix),
                verbose=False,
            )
        except Exception as e:
            raise ValueError(f"Registration failed: {str(e)}")
        
        # Step 4: Apply transforms
        logging.info(f"    - Applying transforms")
        try:
            img_registered = ants.apply_transforms(
                fixed=self.template_img,
                moving=img_n4,
                transformlist=reg['fwdtransforms'],
                interpolator='linear',
            )
        except Exception as e:
            raise ValueError(f"Transform application failed: {str(e)}")
        
        # Step 5: Save final result
        ants.image_write(img_registered, str(final_output))
        logging.info(f"    - Saved: {final_output.name}")
    
    def _is_subject_complete(self, subject_id: str) -> bool:
        """Check if subject has already been processed"""
        for modality in self.modalities:
            expected_file = self.output_dir / self.dataset_name / modality / f"{subject_id}_{modality}_MNI.nii.gz"
            if not expected_file.exists():
                return False
        return True
    
    def _save_metadata(self, subject_id: str, row: pd.Series) -> None:
        """Save processing metadata for this subject"""
        metadata_dir = self.output_dir / self.dataset_name / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Build output paths for each modality
        output_paths = {}
        for modality in self.modalities:
            n4_path = self.output_dir / self.dataset_name / modality / f"{subject_id}_{modality}_N4.nii.gz"
            mni_path = self.output_dir / self.dataset_name / modality / f"{subject_id}_{modality}_MNI.nii.gz"
            
            output_paths[modality] = {
                'n4_corrected': str(n4_path) if self.save_intermediate and n4_path.exists() else None,
                'mni_registered': str(mni_path) if mni_path.exists() else None,
            }
        
        metadata = {
            'subject_id': subject_id,
            'age': float(row['Age']),
            'processed_date': datetime.now().isoformat(),
            'template': str(self.template_path),
            'registration_type': self.registration_type,
            'modalities': self.modalities,
            'n4_iterations': self.n4_iterations,
            'dataset': self.dataset_name,
            'output_paths': output_paths,
        }
        
        metadata_file = metadata_dir / f"{subject_id}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    # ========== Step 4: Create Output CSV ==========
    
    def _create_output_csv(self) -> pd.DataFrame:
        """Create output CSV with preprocessed image paths"""
        logging.info("")
        logging.info("Creating output CSV...")
        
        # Start with original dataframe
        output_df = self.df.copy()
        
        # Add preprocessed path columns
        for modality in self.modalities:
            new_col = f'{modality}_preprocessed_path'
            output_df[new_col] = ''
        
        # Fill in preprocessed paths
        for idx, row in output_df.iterrows():
            subject_id = str(row['ID'])
            subj_dir = self.output_dir / subject_id
            
            for modality in self.modalities:
                preprocessed_file = subj_dir / f"{subject_id}_{modality}_MNI.nii.gz"
                
                if preprocessed_file.exists():
                    new_col = f'{modality}_preprocessed_path'
                    output_df.at[idx, new_col] = str(preprocessed_file)
        
        # Save output CSV
        output_csv = self.output_dir / "preprocessed_index.csv"
        output_df.to_csv(output_csv, index=False)
        
        logging.info(f"Output CSV saved: {output_csv}")
        
        # Statistics
        for modality in self.modalities:
            col = f'{modality}_preprocessed_path'
            n_complete = (output_df[col] != '').sum()
            pct = 100 * n_complete / len(output_df)
            logging.info(f"  {modality}: {n_complete}/{len(output_df)} completed ({pct:.1f}%)")
        
        return output_df
    
    # ========== Step 5: Summary ==========
    
    def _print_summary(self) -> None:
        """Print processing summary"""
        self.stats['end_time'] = datetime.now()
        duration = self.stats['end_time'] - self.stats['start_time']
        
        logging.info("")
        logging.info("=" * 60)
        logging.info("PREPROCESSING COMPLETE")
        logging.info("=" * 60)
        logging.info(f"Total subjects: {self.stats['total']}")
        logging.info(f"Processed: {self.stats['processed']}")
        logging.info(f"Skipped (already done): {self.stats['skipped']}")
        logging.info(f"Failed: {self.stats['failed']}")
        logging.info(f"Duration: {duration}")
        
        if self.stats['total'] > 0:
            success_rate = 100 * self.stats['processed'] / self.stats['total']
            logging.info(f"Success rate: {success_rate:.1f}%")
        
        logging.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Universal MRI Preprocessing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (auto-detect modalities)
  python data_preprocessor.py \\
    --input_csv raw_index.csv \\
    --output_dir preprocessed \\
    --template MNI152_T1_1mm.nii.gz
  
  # Specify modalities explicitly
  python data_preprocessor.py \\
    --input_csv raw_index.csv \\
    --output_dir preprocessed \\
    --template MNI152_T1_1mm.nii.gz \\
    --modalities T1 T2
  
  # Process subset (for batch processing)
  python data_preprocessor.py \\
    --input_csv raw_index.csv \\
    --output_dir preprocessed \\
    --template MNI152_T1_1mm.nii.gz \\
    --start_row 0 \\
    --end_row 100
  
  # Use affine registration instead of rigid
  python data_preprocessor.py \\
    --input_csv raw_index.csv \\
    --output_dir preprocessed \\
    --template MNI152_T1_1mm.nii.gz \\
    --registration_type Affine
        """
    )
    
    # Required arguments
    parser.add_argument('--input_csv', required=True,
                        help='Input CSV from data_indexer.py')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for preprocessed images')
    parser.add_argument('--template', required=True,
                        help='Template image path (e.g., MNI152_T1_1mm.nii.gz)')
    
    # Optional arguments
    parser.add_argument('--modalities', nargs='+',
                        help='Modalities to process (e.g., T1 T2). If not specified, auto-detect from CSV.')
    parser.add_argument('--registration_type', default='Rigid',
                        choices=['Rigid', 'Affine', 'SyN', 'SyNRA', 'SyNOnly'],
                        help='ANTs registration type (default: Rigid)')
    parser.add_argument('--n4_iterations', type=int, nargs=4,
                        default=[50, 50, 50, 50],
                        help='N4 iterations per level (default: 50 50 50 50)')
    parser.add_argument('--start_row', type=int, default=0,
                        help='Start row index (inclusive, default: 0)')
    parser.add_argument('--end_row', type=int,
                        help='End row index (exclusive, default: all rows)')
    parser.add_argument('--no_skip_existing', action='store_true',
                        help='Reprocess subjects even if output exists')
    parser.add_argument('--save_intermediate', action='store_true',
                        help='Save intermediate results (N4 only)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s: %(message)s'
    )
    
    # Create preprocessor
    preprocessor = UniversalPreprocessor(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        template_path=args.template,
        modalities=args.modalities,
        registration_type=args.registration_type,
        n4_iterations=args.n4_iterations,
        skip_existing=not args.no_skip_existing,
        save_intermediate=args.save_intermediate,
    )
    
    # Run preprocessing
    try:
        preprocessor.run(
            start_row=args.start_row,
            end_row=args.end_row,
        )
    except Exception as e:
        logging.error(f"\nFATAL ERROR: {str(e)}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())