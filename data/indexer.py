"""
Universal MRI Dataset Indexer

Automatically creates a CSV index for any MRI dataset by:
1. Auto-detecting metadata format (Excel/CSV/TSV/JSON)
2. Auto-identifying ID and Age columns
3. Recursively scanning for image files
4. Intelligently matching IDs across different naming conventions
5. Validating data completeness

Usage:
    python data_indexer.py \
        --metadata_file data/IXI/IXI.xls \
        --image_root data/IXI/IXI-T1 \
        --output ixi_index.csv
    
    # Multi-modality support
    python data_indexer.py \
        --metadata_file data/IXI/IXI.xls \
        --t1_root data/IXI/IXI-T1 \
        --t2_root data/IXI/IXI-T2 \
        --output ixi_full_index.csv
"""

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import pandas as pd


class UniversalIndexer:
    """Universal indexer for MRI datasets"""

    # Common column name patterns for fuzzy matching
    ID_PATTERNS = ['id', 'patient', 'subject', 'participant', 'sub', 'case']
    AGE_PATTERNS = ['age', 'anos', 'alter', 'yas']
    GENDER_PATTERNS = ['sex', 'gender', 'male', 'female', 'geschlecht', 'sexo']

    # Gender value mappings (normalize to 0=Female, 1=Male)
    GENDER_MALE_VALUES = ['m', 'male', '1', 'man', 'männlich', 'masculino', 'homme']
    GENDER_FEMALE_VALUES = ['f', 'female', '2', 'woman', 'weiblich', 'femenino', 'femme', '0']
    
    # Common filename patterns for ID extraction
    FILENAME_PATTERNS = [
        r'IXI(\d+)',              # IXI002-Guys-0828-T1.nii.gz
        r'sub-([a-zA-Z0-9]+)',    # BIDS: sub-001_T1w.nii.gz
        r'subject[_-]([a-zA-Z0-9]+)',  # subject_0012.nii
        r'patient[_-]([a-zA-Z0-9]+)',  # patient_001.nii
        r'([a-zA-Z0-9]{3,})',     # Any alphanumeric 3+ chars
    ]
    
    # Valid image extensions
    IMAGE_EXTENSIONS = ['.nii', '.nii.gz', '.nrrd', '.mha', '.mhd']
    
    def __init__(self,
                 metadata_file: str,
                 image_roots: Dict[str, str],
                 output_csv: str,
                 recursive: bool = True,
                 min_age: float = 0.0,
                 max_age: float = 120.0,
                 require_all_modalities: bool = False,
                 extract_gender: bool = True):
        """
        Args:
            metadata_file: Path to metadata file (Excel/CSV/TSV/JSON)
            image_roots: Dict mapping modality -> root path, e.g. {'T1': 'path/to/t1', 'T2': 'path/to/t2'}
            output_csv: Output CSV path
            recursive: Whether to search subdirectories
            min_age: Minimum valid age
            max_age: Maximum valid age
            require_all_modalities: If True, only include subjects with all modalities
            extract_gender: Whether to extract gender information from metadata
        """
        self.metadata_file = Path(metadata_file)
        self.image_roots = {k: Path(v) for k, v in image_roots.items()}
        self.output_csv = Path(output_csv)
        self.recursive = recursive
        self.min_age = min_age
        self.max_age = max_age
        self.require_all_modalities = require_all_modalities
        self.extract_gender = extract_gender

        self.metadata_df: Optional[pd.DataFrame] = None
        self.id_col: Optional[str] = None
        self.age_col: Optional[str] = None
        self.gender_col: Optional[str] = None
        
    def create_index(self) -> pd.DataFrame:
        """Main pipeline: load, match, validate, save"""
        logging.info("=" * 60)
        logging.info("Starting Universal MRI Dataset Indexer")
        logging.info("=" * 60)
        
        # Step 1: Load metadata
        self.metadata_df = self._auto_load_metadata()
        logging.info(f"Loaded metadata: {len(self.metadata_df)} rows, {len(self.metadata_df.columns)} columns")
        
        # Step 2: Detect key columns
        self.id_col, self.age_col = self._auto_detect_columns()
        logging.info(f"Detected columns: ID='{self.id_col}', Age='{self.age_col}'")
        
        # Step 3: Scan images
        image_maps = self._scan_all_modalities()
        
        # Step 4: Match IDs
        result_df = self._match_and_merge(image_maps)
        
        # Step 5: Validate and save
        self._validate_and_save(result_df)
        
        return result_df
    
    # ========== Step 1: Load Metadata ==========
    
    def _auto_load_metadata(self) -> pd.DataFrame:
        """Auto-detect and load metadata file"""
        if not self.metadata_file.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_file}\n"
                f"Suggestion: Check the file path and ensure it exists."
            )
        
        ext = self.metadata_file.suffix.lower()
        
        try:
            if ext in ['.xls', '.xlsx']:
                return pd.read_excel(self.metadata_file)
            elif ext == '.csv':
                return pd.read_csv(self.metadata_file)
            elif ext == '.tsv':
                return pd.read_csv(self.metadata_file, sep='\t')
            elif ext == '.json':
                return pd.read_json(self.metadata_file)
            else:
                # Try common formats
                for reader_func in [pd.read_csv, pd.read_excel]:
                    try:
                        return reader_func(self.metadata_file)
                    except Exception:
                        continue
                
                raise ValueError(f"Unsupported file format: {ext}")
                
        except Exception as e:
            raise ValueError(
                f"Failed to load metadata file: {self.metadata_file}\n"
                f"Error: {str(e)}\n"
                f"Suggestion: Ensure the file is a valid Excel/CSV/TSV/JSON file."
            )
    
    # ========== Step 2: Detect Columns ==========
    
    def _auto_detect_columns(self) -> Tuple[str, str]:
        """Auto-detect ID and Age columns with fuzzy matching"""
        id_col = self._fuzzy_find_column(self.metadata_df.columns, self.ID_PATTERNS)
        age_col = self._fuzzy_find_column(self.metadata_df.columns, self.AGE_PATTERNS)

        if not id_col:
            raise ValueError(
                f"Could not auto-detect ID column in metadata.\n"
                f"Available columns: {list(self.metadata_df.columns)}\n"
                f"Suggestion: Ensure metadata has a column containing 'id', 'patient', 'subject', or 'participant'.\n"
                f"Or manually specify with --id_column argument."
            )

        if not age_col:
            raise ValueError(
                f"Could not auto-detect Age column in metadata.\n"
                f"Available columns: {list(self.metadata_df.columns)}\n"
                f"Suggestion: Ensure metadata has a column containing 'age'.\n"
                f"Or manually specify with --age_column argument."
            )

        # Validate Age column is numeric
        try:
            pd.to_numeric(self.metadata_df[age_col], errors='coerce')
        except Exception:
            raise ValueError(
                f"Age column '{age_col}' contains non-numeric values.\n"
                f"Suggestion: Clean the age data or specify a different column."
            )

        # Detect gender column (optional)
        if self.extract_gender:
            gender_col = self._fuzzy_find_column(self.metadata_df.columns, self.GENDER_PATTERNS)
            if gender_col:
                self.gender_col = gender_col
                logging.info(f"Detected gender column: '{gender_col}'")
            else:
                logging.warning(
                    f"Could not auto-detect Gender column in metadata.\n"
                    f"Available columns: {list(self.metadata_df.columns)}\n"
                    f"Gender prediction will not be available for this dataset."
                )

        return id_col, age_col

    def _normalize_gender(self, value) -> Optional[int]:
        """
        Normalize gender value to binary format.
        Returns: 0 for Female, 1 for Male, None if unknown
        """
        if pd.isna(value):
            return None

        value_str = str(value).lower().strip()

        if value_str in self.GENDER_MALE_VALUES:
            return 1
        elif value_str in self.GENDER_FEMALE_VALUES:
            return 0
        else:
            # Try numeric interpretation
            try:
                num_val = float(value_str)
                if num_val == 1:
                    return 1  # Male
                elif num_val == 2 or num_val == 0:
                    return 0  # Female
            except ValueError:
                pass
            return None
    
    def _fuzzy_find_column(self, columns: List[str], patterns: List[str]) -> Optional[str]:
        """Fuzzy match column names (case-insensitive, ignore separators)"""
        for col in columns:
            col_clean = col.lower().replace('_', '').replace('-', '').replace(' ', '')
            for pattern in patterns:
                if pattern in col_clean:
                    return col
        return None
    
    # ========== Step 3: Scan Images ==========
    
    def _scan_all_modalities(self) -> Dict[str, Dict[str, str]]:
        """Scan all image roots and build ID->path mappings"""
        image_maps = {}
        
        for modality, root in self.image_roots.items():
            logging.info(f"Scanning {modality} images in: {root}")
            
            if not root.exists():
                logging.warning(
                    f"Image root not found: {root}\n"
                    f"Suggestion: Check the path or skip this modality."
                )
                image_maps[modality] = {}
                continue
            
            file_map = self._scan_image_directory(root)
            logging.info(f"  Found {len(file_map)} unique IDs for {modality}")
            image_maps[modality] = file_map
        
        return image_maps
    
    def _scan_image_directory(self, root: Path) -> Dict[str, str]:
        """
        Recursively scan directory for images and build ID->path mapping.
        Returns dict with multiple ID variants as keys pointing to same path.
        """
        file_map = {}
        
        # Find all image files
        if self.recursive:
            image_files = []
            for ext in self.IMAGE_EXTENSIONS:
                image_files.extend(root.rglob(f'*{ext}'))
        else:
            image_files = []
            for ext in self.IMAGE_EXTENSIONS:
                image_files.extend(root.glob(f'*{ext}'))
        
        logging.info(f"  Found {len(image_files)} image files")
        
        # Extract IDs from filenames
        for img_path in image_files:
            extracted_id = self._extract_id_from_filename(img_path.name)
            
            if not extracted_id:
                logging.debug(f"  Could not extract ID from: {img_path.name}")
                continue
            
            # Generate ID variants for robust matching
            id_variants = self._generate_id_variants(extracted_id)
            
            # Map all variants to this file path
            for variant in id_variants:
                if variant not in file_map:
                    file_map[variant] = str(img_path)
        
        return file_map
    
    def _extract_id_from_filename(self, filename: str) -> Optional[str]:
        """Try multiple patterns to extract ID from filename"""
        for pattern in self.FILENAME_PATTERNS:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Fallback: use stem without extension
        return Path(filename).stem
    
    def _generate_id_variants(self, raw_id: str) -> Set[str]:
        """
        Generate multiple ID variants for robust matching.
        Examples:
            '002' -> {'002', '2', 'IXI002', 'sub-002', ...}
            'IXI012' -> {'IXI012', '012', '12', 'sub-012', ...}
        """
        variants = set()
        
        # Original ID
        variants.add(str(raw_id))
        variants.add(str(raw_id).upper())
        variants.add(str(raw_id).lower())
        
        # Extract numeric part
        digits = re.findall(r'\d+', str(raw_id))
        if digits:
            num_str = digits[0]
            num_int = int(num_str)
            
            # Different numeric formats
            variants.add(num_str)                    # '002'
            variants.add(str(num_int))              # '2'
            variants.add(num_str.zfill(3))          # '002'
            variants.add(num_str.zfill(4))          # '0002'
            variants.add(num_str.zfill(5))          # '00002'
            
            # Common prefixes
            for prefix in ['IXI', 'sub-', 'subject_', 'patient_', 'case_']:
                variants.add(f'{prefix}{num_str}')
                variants.add(f'{prefix}{num_int}')
                variants.add(f'{prefix}{num_str.zfill(3)}')
        
        # Extract alphabetic prefix + numeric suffix (e.g., 'IXI002')
        match = re.match(r'([a-zA-Z]+)(\d+)', str(raw_id))
        if match:
            prefix, num = match.groups()
            variants.add(num)
            variants.add(str(int(num)))
            variants.add(f'{prefix}{num}')
            variants.add(f'{prefix}{int(num)}')
        
        return variants
    
    # ========== Step 4: Match and Merge ==========
    
    def _match_and_merge(self, image_maps: Dict[str, Dict[str, str]]) -> pd.DataFrame:
        """Match metadata IDs with image files and merge into single dataframe"""
        logging.info("Matching IDs between metadata and images...")

        records = []
        match_stats = {mod: {'matched': 0, 'missing': 0} for mod in image_maps.keys()}
        gender_stats = {'male': 0, 'female': 0, 'unknown': 0}

        for idx, row in self.metadata_df.iterrows():
            raw_id = str(row[self.id_col]).strip()
            age = row[self.age_col]

            # Generate ID variants for this subject
            id_variants = self._generate_id_variants(raw_id)

            # Try to find images for each modality
            record = {
                'ID': raw_id,
                'Age': age,
            }

            # Extract gender if available
            if self.gender_col and self.gender_col in row:
                gender = self._normalize_gender(row[self.gender_col])
                record['Gender'] = gender if gender is not None else -1  # -1 for unknown
                if gender == 1:
                    gender_stats['male'] += 1
                elif gender == 0:
                    gender_stats['female'] += 1
                else:
                    gender_stats['unknown'] += 1

            match_confidence = 'high'

            for modality, file_map in image_maps.items():
                col_name = f'{modality}_path'

                # Try all ID variants
                matched_path = None
                for variant in id_variants:
                    if variant in file_map:
                        matched_path = file_map[variant]
                        match_stats[modality]['matched'] += 1
                        break

                if matched_path:
                    record[col_name] = matched_path
                else:
                    record[col_name] = ''
                    match_stats[modality]['missing'] += 1
                    match_confidence = 'low'

            record['match_confidence'] = match_confidence
            records.append(record)
        
        # Log matching statistics
        logging.info("Matching statistics:")
        for modality, stats in match_stats.items():
            total = stats['matched'] + stats['missing']
            pct = 100 * stats['matched'] / total if total > 0 else 0
            logging.info(f"  {modality}: {stats['matched']}/{total} matched ({pct:.1f}%)")

        # Log gender statistics if extracted
        if self.gender_col:
            total_gender = sum(gender_stats.values())
            logging.info("Gender statistics:")
            logging.info(f"  Male: {gender_stats['male']}/{total_gender} ({100*gender_stats['male']/total_gender:.1f}%)")
            logging.info(f"  Female: {gender_stats['female']}/{total_gender} ({100*gender_stats['female']/total_gender:.1f}%)")
            if gender_stats['unknown'] > 0:
                logging.info(f"  Unknown: {gender_stats['unknown']}/{total_gender} ({100*gender_stats['unknown']/total_gender:.1f}%)")

        result_df = pd.DataFrame(records)
        
        # Filter by modality requirements
        if self.require_all_modalities:
            before = len(result_df)
            path_cols = [col for col in result_df.columns if col.endswith('_path')]
            result_df = result_df[result_df[path_cols].apply(lambda x: all(x != ''), axis=1)]
            after = len(result_df)
            logging.info(f"Filtered to subjects with all modalities: {after}/{before} remaining")
        
        return result_df
    
    # ========== Step 5: Validate and Save ==========
    
    def _validate_and_save(self, df: pd.DataFrame) -> None:
        """Validate data quality and save to CSV"""
        logging.info("Validating data...")
        
        # Check for empty dataframe
        if len(df) == 0:
            raise ValueError(
                "No subjects matched between metadata and images.\n"
                "Suggestions:\n"
                "  1. Check that image filenames contain subject IDs\n"
                "  2. Verify metadata IDs match filename patterns\n"
                "  3. Use --recursive flag if images are in subdirectories\n"
                "  4. Check image file extensions (.nii, .nii.gz, etc.)"
            )
        
        # Validate ages
        invalid_ages = df[(df['Age'] < self.min_age) | (df['Age'] > self.max_age)]
        if len(invalid_ages) > 0:
            logging.warning(
                f"Found {len(invalid_ages)} subjects with invalid ages (range: {self.min_age}-{self.max_age})\n"
                f"First few: {invalid_ages['ID'].head().tolist()}"
            )
        
        # Check for missing paths
        path_cols = [col for col in df.columns if col.endswith('_path')]
        for col in path_cols:
            n_missing = (df[col] == '').sum()
            if n_missing > 0:
                pct = 100 * n_missing / len(df)
                logging.warning(f"{col}: {n_missing}/{len(df)} missing ({pct:.1f}%)")
        
        # Check for duplicates
        dup_ids = df[df['ID'].duplicated()]
        if len(dup_ids) > 0:
            raise ValueError(
                f"Found {len(dup_ids)} duplicate IDs in metadata: {dup_ids['ID'].tolist()}\n"
                f"Suggestion: Clean metadata to have unique subject IDs."
            )
        
        # Verify file existence (sample check)
        sample_size = min(10, len(df))
        sample_df = df.sample(n=sample_size, random_state=42)
        
        for _, row in sample_df.iterrows():
            for col in path_cols:
                path_str = row[col]
                if path_str and not Path(path_str).exists():
                    raise FileNotFoundError(
                        f"Image file not found: {path_str}\n"
                        f"Subject ID: {row['ID']}\n"
                        f"Suggestion: Verify image paths are correct."
                    )
        
        # Save CSV
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_csv, index=False)
        
        logging.info("=" * 60)
        logging.info(f"SUCCESS: Created index with {len(df)} subjects")
        logging.info(f"Output saved to: {self.output_csv}")
        logging.info("=" * 60)
        
        # Summary statistics
        logging.info("\nSummary:")
        logging.info(f"  Total subjects: {len(df)}")
        logging.info(f"  Age range: {df['Age'].min():.1f} - {df['Age'].max():.1f}")
        logging.info(f"  Mean age: {df['Age'].mean():.1f} ± {df['Age'].std():.1f}")
        
        high_conf = (df['match_confidence'] == 'high').sum()
        logging.info(f"  High confidence matches: {high_conf}/{len(df)} ({100*high_conf/len(df):.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Universal MRI Dataset Indexer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single modality (T1 only)
  python data_indexer.py \\
    --metadata_file data/IXI/IXI.xls \\
    --image_root data/IXI/IXI-T1 \\
    --output ixi_index.csv
  
  # Multi-modality (T1 and T2)
  python data_indexer.py \\
    --metadata_file data/IXI/IXI.xls \\
    --t1_root data/IXI/IXI-T1 \\
    --t2_root data/IXI/IXI-T2 \\
    --output ixi_full_index.csv \\
    --require_all_modalities
  
  # BIDS dataset
  python data_indexer.py \\
    --metadata_file data/participants.tsv \\
    --image_root data/derivatives/preprocessed \\
    --output bids_index.csv \\
    --recursive
        """
    )
    
    # Required arguments
    parser.add_argument('--metadata_file', required=True,
                        help='Path to metadata file (Excel/CSV/TSV/JSON)')
    parser.add_argument('--output', required=True,
                        help='Output CSV file path')
    
    # Image root options (use either --image_root OR specific modality roots)
    image_group = parser.add_mutually_exclusive_group(required=True)
    image_group.add_argument('--image_root',
                             help='Single image root directory (for single modality)')
    image_group.add_argument('--t1_root',
                             help='T1 image root directory')
    
    parser.add_argument('--t2_root', help='T2 image root directory (optional)')
    parser.add_argument('--flair_root', help='FLAIR image root directory (optional)')
    
    # Optional arguments
    parser.add_argument('--id_column', help='Override auto-detected ID column name')
    parser.add_argument('--age_column', help='Override auto-detected Age column name')
    parser.add_argument('--gender_column', help='Override auto-detected Gender column name')
    parser.add_argument('--no_gender', action='store_true',
                        help='Do not extract gender information')
    parser.add_argument('--recursive', action='store_true', default=True,
                        help='Recursively search subdirectories (default: True)')
    parser.add_argument('--no_recursive', action='store_false', dest='recursive',
                        help='Do not search subdirectories')
    parser.add_argument('--require_all_modalities', action='store_true',
                        help='Only include subjects with all specified modalities')
    parser.add_argument('--min_age', type=float, default=0.0,
                        help='Minimum valid age (default: 0.0)')
    parser.add_argument('--max_age', type=float, default=120.0,
                        help='Maximum valid age (default: 120.0)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s: %(message)s'
    )
    
    # Build image_roots dict
    image_roots = {}
    if args.image_root:
        image_roots['T1'] = args.image_root
    else:
        if args.t1_root:
            image_roots['T1'] = args.t1_root
        if args.t2_root:
            image_roots['T2'] = args.t2_root
        if args.flair_root:
            image_roots['FLAIR'] = args.flair_root
    
    # Create indexer
    indexer = UniversalIndexer(
        metadata_file=args.metadata_file,
        image_roots=image_roots,
        output_csv=args.output,
        recursive=args.recursive,
        min_age=args.min_age,
        max_age=args.max_age,
        require_all_modalities=args.require_all_modalities,
        extract_gender=not args.no_gender,
    )

    # Override column detection if specified
    if args.id_column:
        indexer.ID_PATTERNS = [args.id_column.lower()]
    if args.age_column:
        indexer.AGE_PATTERNS = [args.age_column.lower()]
    if args.gender_column:
        indexer.GENDER_PATTERNS = [args.gender_column.lower()]
    
    # Run indexer
    try:
        indexer.create_index()
    except Exception as e:
        logging.error(f"\nFATAL ERROR: {str(e)}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())