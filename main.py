"""
Universal Brain Age Prediction Pipeline

Complete end-to-end pipeline from raw data to trained model.

Usage:
    # Run complete pipeline
    python main.py \
        --config configs/ixi_config.yaml \
        --run_all
    
    # Run specific steps
    python main.py \
        --config configs/ixi_config.yaml \
        --steps index preprocess split train
    
    # Skip completed steps
    python main.py \
        --config configs/ixi_config.yaml \
        --steps preprocess split train \
        --skip_existing
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Import pipeline components
try:
    from data_indexer import UniversalIndexer
    from data_preprocessor import UniversalPreprocessor
    from data_splitter import UniversalSplitter
    from trainer import BrainAgeTrainer
except ImportError:
    # If running as standalone script, add parent dir to path
    sys.path.insert(0, str(Path(__file__).parent))
    from data_indexer import UniversalIndexer
    from data_preprocessor import UniversalPreprocessor
    from data_splitter import UniversalSplitter
    from trainer import BrainAgeTrainer


class BrainAgePipeline:
    """Complete brain age prediction pipeline orchestrator"""
    
    STEPS = ['index', 'preprocess', 'split', 'train']
    
    def __init__(self, config: Dict, skip_existing: bool = False):
        """
        Args:
            config: Configuration dictionary
            skip_existing: Skip steps if output already exists
        """
        self.config = config
        self.skip_existing = skip_existing
        
        # Create output directories
        self.output_root = Path(config['output_root'])
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        logging.info("=" * 70)
        logging.info("BRAIN AGE PREDICTION PIPELINE")
        logging.info("=" * 70)
        logging.info(f"Output root: {self.output_root}")
        logging.info(f"Skip existing: {skip_existing}")
        logging.info("")
    
    def _setup_logging(self) -> None:
        """Setup logging to file and console"""
        log_dir = self.output_root / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter('%(levelname)s: %(message)s')
        )
        
        # Configure root logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        logging.info(f"Log file: {log_file}")
    
    def run(self, steps: Optional[List[str]] = None) -> None:
        """
        Run pipeline steps.
        
        Args:
            steps: List of steps to run. If None, run all steps.
        """
        if steps is None:
            steps = self.STEPS
        
        # Validate steps
        invalid_steps = set(steps) - set(self.STEPS)
        if invalid_steps:
            raise ValueError(
                f"Invalid steps: {invalid_steps}\n"
                f"Valid steps: {self.STEPS}"
            )
        
        logging.info(f"Steps to run: {steps}")
        logging.info("")
        
        # Run steps in order
        for step in self.STEPS:
            if step not in steps:
                logging.info(f"Skipping step: {step}")
                continue
            
            method_name = f"_run_{step}"
            if not hasattr(self, method_name):
                raise NotImplementedError(f"Step '{step}' not implemented")
            
            logging.info("")
            logging.info("=" * 70)
            logging.info(f"STEP: {step.upper()}")
            logging.info("=" * 70)
            
            try:
                method = getattr(self, method_name)
                method()
                logging.info(f"Step '{step}' completed successfully")
            except Exception as e:
                logging.error(f"Step '{step}' failed: {str(e)}")
                raise
        
        logging.info("")
        logging.info("=" * 70)
        logging.info("PIPELINE COMPLETED SUCCESSFULLY")
        logging.info("=" * 70)
    
    # ========== Step 1: Create Index ==========
    
    def _run_index(self) -> None:
        """Step 1: Create raw data index"""
        config = self.config['indexer']
        
        output_csv = self.output_root / "raw_index.csv"
        
        # Skip if exists
        if self.skip_existing and output_csv.exists():
            logging.info(f"Index already exists: {output_csv}")
            return
        
        # Build image_roots dict
        image_roots = {}
        for modality in config.get('modalities', ['T1']):
            key = f'{modality.lower()}_root'
            if key in config:
                image_roots[modality] = config[key]
        
        if not image_roots:
            raise ValueError(
                "No image roots specified in config.\n"
                "Example: indexer.t1_root, indexer.t2_root"
            )
        
        # Create indexer
        indexer = UniversalIndexer(
            metadata_file=config['metadata_file'],
            image_roots=image_roots,
            output_csv=str(output_csv),
            recursive=config.get('recursive', True),
            min_age=config.get('min_age', 0.0),
            max_age=config.get('max_age', 120.0),
            require_all_modalities=config.get('require_all_modalities', False),
        )
        
        # Run
        indexer.create_index()
        
        # Save output path to config for next steps
        self.config['_raw_index_csv'] = str(output_csv)
    
    # ========== Step 2: Preprocess ==========
    
    def _run_preprocess(self) -> None:
        """Step 2: Preprocess images"""
        config = self.config['preprocessor']
        
        # Get input CSV from previous step or config
        if '_raw_index_csv' in self.config:
            input_csv = self.config['_raw_index_csv']
        else:
            input_csv = config.get('input_csv')
            if not input_csv:
                raise ValueError(
                    "No input CSV specified for preprocessing.\n"
                    "Either run 'index' step first or specify preprocessor.input_csv in config."
                )
        
        output_dir = self.output_root / "preprocessed"
        preprocessed_csv = output_dir / "preprocessed_index.csv"
        
        # Skip if exists
        if self.skip_existing and preprocessed_csv.exists():
            logging.info(f"Preprocessed data already exists: {output_dir}")
            self.config['_preprocessed_csv'] = str(preprocessed_csv)
            return
        
        # Create preprocessor
        preprocessor = UniversalPreprocessor(
            input_csv=input_csv,
            output_dir=str(output_dir),
            template_path=config['template'],
            modalities=config.get('modalities'),
            registration_type=config.get('registration_type', 'Rigid'),
            n4_iterations=config.get('n4_iterations', [50, 50, 50, 50]),
            skip_existing=config.get('skip_existing', True),
            save_intermediate=config.get('save_intermediate', False),
        )
        
        # Run
        preprocessor.run(
            start_row=config.get('start_row', 0),
            end_row=config.get('end_row'),
        )
        
        # Save output path for next steps
        self.config['_preprocessed_csv'] = str(preprocessed_csv)
    
    # ========== Step 3: Split Dataset ==========
    
    def _run_split(self) -> None:
        """Step 3: Split into train/val/test"""
        config = self.config['splitter']
        
        # Get input CSV from previous step or config
        if '_preprocessed_csv' in self.config:
            input_csv = self.config['_preprocessed_csv']
        else:
            input_csv = config.get('input_csv')
            if not input_csv:
                raise ValueError(
                    "No input CSV specified for splitting.\n"
                    "Either run 'preprocess' step first or specify splitter.input_csv in config."
                )
        
        output_dir = self.output_root / "splits"
        train_csv = output_dir / "train_split.csv"
        val_csv = output_dir / "val_split.csv"
        test_csv = output_dir / "test_split.csv"
        
        # Skip if exists
        if self.skip_existing and all(p.exists() for p in [train_csv, val_csv, test_csv]):
            logging.info(f"Splits already exist: {output_dir}")
            self.config['_train_csv'] = str(train_csv)
            self.config['_val_csv'] = str(val_csv)
            self.config['_test_csv'] = str(test_csv)
            return
        
        # Get ratios
        ratios = config.get('ratios', [0.6, 0.2, 0.2])
        if len(ratios) != 3:
            raise ValueError(f"Ratios must have 3 values, got {len(ratios)}")
        
        # Create splitter
        splitter = UniversalSplitter(
            input_csv=input_csv,
            output_dir=str(output_dir),
            train_ratio=ratios[0],
            val_ratio=ratios[1],
            test_ratio=ratios[2],
            stratify_age=config.get('stratify_age', False),
            age_bins=config.get('age_bins', 5),
            seed=config.get('seed', 42),
            require_complete=config.get('require_complete', True),
        )
        
        # Run
        splitter.run()
        
        # Save output paths for next steps
        self.config['_train_csv'] = str(train_csv)
        self.config['_val_csv'] = str(val_csv)
        self.config['_test_csv'] = str(test_csv)
    
    # ========== Step 4: Train ==========
    
    def _run_train(self) -> None:
        """Step 4: Train model"""
        config = self.config['trainer']
        
        # Get input CSVs from previous step or config
        if all(k in self.config for k in ['_train_csv', '_val_csv', '_test_csv']):
            train_csv = self.config['_train_csv']
            val_csv = self.config['_val_csv']
            test_csv = self.config['_test_csv']
        else:
            train_csv = config.get('train_csv')
            val_csv = config.get('val_csv')
            test_csv = config.get('test_csv')
            
            if not all([train_csv, val_csv, test_csv]):
                raise ValueError(
                    "Training CSVs not specified.\n"
                    "Either run 'split' step first or specify trainer.train_csv, val_csv, test_csv in config."
                )
        
        output_dir = self.output_root / "experiments" / config.get('experiment_name', 'exp001')
        
        # Create trainer
        trainer = BrainAgeTrainer(
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            output_dir=str(output_dir),
            modality=config.get('modality', 'T1'),
            target_size=tuple(config.get('target_size', [160, 192, 160])),
            batch_size=config.get('batch_size', 2),
            num_workers=config.get('num_workers', 4),
            epochs=config.get('epochs', 500),
            lr=config.get('lr', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5),
            early_stopping_patience=config.get('early_stopping_patience', 15),
            lr_scheduler_patience=config.get('lr_scheduler_patience', 5),
            seed=config.get('seed', 42),
        )
        
        # Run
        trainer.run(resume=config.get('resume', False))


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def create_example_config(output_path: str = "example_config.yaml") -> None:
    """Create an example configuration file"""
    example_config = {
        'output_root': 'outputs/ixi_pipeline',
        
        'indexer': {
            'metadata_file': 'data/IXI/IXI.xls',
            'modalities': ['T1', 'T2'],
            't1_root': 'data/IXI/IXI-T1',
            't2_root': 'data/IXI/IXI-T2',
            'recursive': True,
            'min_age': 18.0,
            'max_age': 90.0,
            'require_all_modalities': False,
        },
        
        'preprocessor': {
            'template': 'templates/MNI152_T1_1mm.nii.gz',
            'modalities': None,  # Auto-detect from CSV
            'registration_type': 'Rigid',
            'n4_iterations': [50, 50, 50, 50],
            'skip_existing': True,
            'save_intermediate': False,
        },
        
        'splitter': {
            'ratios': [0.6, 0.2, 0.2],
            'stratify_age': True,
            'age_bins': 5,
            'seed': 42,
            'require_complete': True,
        },
        
        'trainer': {
            'experiment_name': 'exp001',
            'modality': 'T1',
            'target_size': [160, 192, 160],
            'batch_size': 2,
            'num_workers': 4,
            'epochs': 500,
            'lr': 1e-4,
            'weight_decay': 1e-5,
            'early_stopping_patience': 15,
            'lr_scheduler_patience': 5,
            'seed': 42,
            'resume': False,
        },
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(example_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Example configuration saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Universal Brain Age Prediction Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create example config
  python main.py --create_example_config
  
  # Run complete pipeline
  python main.py --config configs/ixi.yaml --run_all
  
  # Run specific steps
  python main.py --config configs/ixi.yaml --steps index preprocess
  
  # Skip existing outputs
  python main.py --config configs/ixi.yaml --run_all --skip_existing
  
  # Resume training
  python main.py --config configs/ixi.yaml --steps train
        """
    )
    
    parser.add_argument('--config', help='Path to configuration YAML file')
    parser.add_argument('--steps', nargs='+', choices=BrainAgePipeline.STEPS,
                        help='Steps to run (default: all)')
    parser.add_argument('--run_all', action='store_true',
                        help='Run all pipeline steps')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip steps if output already exists')
    parser.add_argument('--create_example_config', action='store_true',
                        help='Create example configuration file and exit')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Create example config if requested
    if args.create_example_config:
        create_example_config()
        return 0
    
    # Validate arguments
    if not args.config:
        parser.error("--config is required (or use --create_example_config)")
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s: %(message)s'
    )
    
    # Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        logging.error(f"Failed to load config: {str(e)}")
        return 1
    
    # Determine steps to run
    if args.run_all:
        steps = None  # Run all steps
    elif args.steps:
        steps = args.steps
    else:
        parser.error("Either --run_all or --steps must be specified")
    
    # Create and run pipeline
    try:
        pipeline = BrainAgePipeline(config, skip_existing=args.skip_existing)
        pipeline.run(steps=steps)
    except Exception as e:
        logging.error(f"\nPIPELINE FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())