"""
Configuration Management for Brain Age Prediction Pipeline

This module provides centralized configuration for all pipeline components.

Usage:
    # Use default configuration
    from config import Config
    cfg = Config()
    
    # Load from YAML
    from config import Config
    cfg = Config.from_yaml('my_config.yaml')
    
    # Override specific values
    cfg = Config(batch_size=4, epochs=100)
    
    # Save configuration
    cfg.save_yaml('saved_config.yaml')
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import yaml
from dataclasses import dataclass, field, asdict


# Type alias for paths that can be str or Path
PathLike = Union[str, Path]


@dataclass
class PathConfig:
    """File path configurations

    Note: Path fields are initially strings but may be converted to Path objects
    by Config.__post_init__() for convenience.
    """
    # Input data paths
    metadata_file: PathLike = "data/IXI/IXI.xls"
    t1_root: Optional[PathLike] = "data/IXI/IXI-T1"
    t2_root: Optional[PathLike] = None
    flair_root: Optional[PathLike] = None
    template: PathLike = "templates/MNI152_T1_1mm.nii.gz"

    # Output paths
    output_root: PathLike = "outputs"
    experiment_name: str = "exp001"


@dataclass
class IndexerConfig:
    """Configuration for data indexer"""
    modalities: List[str] = field(default_factory=lambda: ["T1"])
    recursive: bool = True
    min_age: float = 0.0
    max_age: float = 120.0
    require_all_modalities: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PreprocessorConfig:
    """Configuration for preprocessing"""
    modalities: Optional[List[str]] = None  # None = auto-detect
    registration_type: str = "Rigid"  # Rigid, Affine, SyN
    n4_iterations: List[int] = field(default_factory=lambda: [50, 50, 50, 50])
    skip_existing: bool = True
    save_intermediate: bool = False
    start_row: int = 0
    end_row: Optional[int] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SplitterConfig:
    """Configuration for dataset splitting"""
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    stratify_age: bool = True
    age_bins: int = 5
    seed: int = 42
    require_complete: bool = True
    
    def __post_init__(self):
        # Validate ratios sum to 1.0
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"Ratios must sum to 1.0, got {total} "
                f"(train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio})"
            )
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ModelConfig:
    """Configuration for model architecture"""
    model_name: str = "DenseNet121"
    spatial_dims: int = 3
    in_channels: int = 1
    out_channels: int = 1  # Regression
    pretrained: bool = False


@dataclass
class TrainerConfig:
    """Configuration for training"""
    # Data settings
    modality: str = "T1"
    target_size: Tuple[int, int, int] = (160, 192, 160)
    batch_size: int = 2
    num_workers: int = 4
    
    # Training settings
    epochs: int = 500
    lr: float = 1e-4
    weight_decay: float = 1e-5
    
    # Scheduler and early stopping
    early_stopping_patience: int = 15
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5
    
    # Reproducibility
    seed: int = 42
    
    # Execution
    resume: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Config:
    """Master configuration class"""
    paths: PathConfig = field(default_factory=PathConfig)
    indexer: IndexerConfig = field(default_factory=IndexerConfig)
    preprocessor: PreprocessorConfig = field(default_factory=PreprocessorConfig)
    splitter: SplitterConfig = field(default_factory=SplitterConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    
    # Additional metadata
    project_name: str = "Brain_Age_Prediction"
    description: str = "Universal brain age prediction pipeline"
    version: str = "1.0.0"
    
    def __post_init__(self):
        """Convert paths to Path objects"""
        self.paths.output_root = Path(self.paths.output_root)
        if self.paths.metadata_file:
            self.paths.metadata_file = Path(self.paths.metadata_file)
        if self.paths.template:
            self.paths.template = Path(self.paths.template)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            Config object
            
        Example:
            >>> cfg = Config.from_yaml('configs/ixi.yaml')
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Parse nested configurations
        paths = PathConfig(**data.get('paths', {}))
        indexer = IndexerConfig(**data.get('indexer', {}))
        preprocessor = PreprocessorConfig(**data.get('preprocessor', {}))
        splitter = SplitterConfig(**data.get('splitter', {}))
        model = ModelConfig(**data.get('model', {}))
        trainer = TrainerConfig(**data.get('trainer', {}))
        
        # Get metadata
        metadata = {
            'project_name': data.get('project_name', 'Brain_Age_Prediction'),
            'description': data.get('description', 'Universal brain age prediction pipeline'),
            'version': data.get('version', '1.0.0'),
        }
        
        return cls(
            paths=paths,
            indexer=indexer,
            preprocessor=preprocessor,
            splitter=splitter,
            model=model,
            trainer=trainer,
            **metadata
        )
    
    def save_yaml(self, yaml_path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            yaml_path: Path to save YAML file
            
        Example:
            >>> cfg = Config()
            >>> cfg.save_yaml('my_config.yaml')
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict
        data = {
            'project_name': self.project_name,
            'description': self.description,
            'version': self.version,
            'paths': asdict(self.paths),
            'indexer': self.indexer.to_dict(),
            'preprocessor': self.preprocessor.to_dict(),
            'splitter': self.splitter.to_dict(),
            'model': asdict(self.model),
            'trainer': self.trainer.to_dict(),
        }
        
        # Convert Path objects to strings for YAML serialization
        def path_to_str(obj):
            if isinstance(obj, dict):
                return {k: path_to_str(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [path_to_str(v) for v in obj]
            elif isinstance(obj, Path):
                return str(obj)
            else:
                return obj
        
        data = path_to_str(data)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        
        print(f"Configuration saved to: {yaml_path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entire config to dictionary"""
        return {
            'project_name': self.project_name,
            'description': self.description,
            'version': self.version,
            'paths': asdict(self.paths),
            'indexer': self.indexer.to_dict(),
            'preprocessor': self.preprocessor.to_dict(),
            'splitter': self.splitter.to_dict(),
            'model': asdict(self.model),
            'trainer': self.trainer.to_dict(),
        }
    
    def get_output_paths(self) -> Dict[str, Path]:
        """
        Get all output paths for the pipeline.
        
        Returns:
            Dictionary with keys: raw_index, preprocessed, splits, experiments
        """
        output_root = Path(self.paths.output_root)
        
        return {
            'raw_index': output_root / "raw_index.csv",
            'preprocessed': output_root / "preprocessed",
            'preprocessed_csv': output_root / "preprocessed" / "preprocessed_index.csv",
            'splits': output_root / "splits",
            'train_csv': output_root / "splits" / "train_split.csv",
            'val_csv': output_root / "splits" / "val_split.csv",
            'test_csv': output_root / "splits" / "test_split.csv",
            'experiments': output_root / "experiments" / self.paths.experiment_name,
            'logs': output_root / "logs",
        }
    
    def validate(self) -> None:
        """
        Validate configuration values.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate paths exist (for input files)
        if not self.paths.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.paths.metadata_file}")
        
        if not self.paths.template.exists():
            raise FileNotFoundError(f"Template file not found: {self.paths.template}")
        
        # Validate registration type
        valid_reg_types = ['Rigid', 'Affine', 'SyN', 'SyNRA', 'SyNOnly']
        if self.preprocessor.registration_type not in valid_reg_types:
            raise ValueError(
                f"Invalid registration type: {self.preprocessor.registration_type}. "
                f"Must be one of {valid_reg_types}"
            )
        
        # Validate training parameters
        if self.trainer.batch_size < 1:
            raise ValueError(f"Batch size must be >= 1, got {self.trainer.batch_size}")
        
        if self.trainer.lr <= 0:
            raise ValueError(f"Learning rate must be > 0, got {self.trainer.lr}")
        
        if self.trainer.epochs < 1:
            raise ValueError(f"Epochs must be >= 1, got {self.trainer.epochs}")
        
        print("✓ Configuration is valid")


# ==================== Preset Configurations ====================

def get_ixi_config() -> Config:
    """
    Get preset configuration for IXI dataset.
    
    Returns:
        Config object configured for IXI dataset
    """
    return Config(
        paths=PathConfig(
            metadata_file="data/IXI/IXI.xls",
            t1_root="data/IXI/IXI-T1",
            t2_root="data/IXI/IXI-T2",
            template="templates/MNI152_T1_1mm.nii.gz",
            output_root="outputs/ixi",
            experiment_name="ixi_baseline",
        ),
        indexer=IndexerConfig(
            modalities=["T1", "T2"],
            min_age=18.0,
            max_age=90.0,
        ),
        splitter=SplitterConfig(
            stratify_age=True,
            age_bins=5,
        ),
        trainer=TrainerConfig(
            batch_size=2,
            epochs=500,
        ),
    )


def get_corr_config() -> Config:
    """
    Get preset configuration for CoRR dataset.
    
    Returns:
        Config object configured for CoRR dataset
    """
    return Config(
        paths=PathConfig(
            metadata_file="data/CoRR/participants.tsv",
            t1_root="data/CoRR/derivatives/preprocessed",
            template="templates/MNI152_T1_1mm.nii.gz",
            output_root="outputs/corr",
            experiment_name="corr_baseline",
        ),
        indexer=IndexerConfig(
            modalities=["T1"],
            recursive=True,
        ),
        splitter=SplitterConfig(
            stratify_age=True,
        ),
    )


def get_quick_test_config() -> Config:
    """
    Get configuration for quick testing (small dataset, few epochs).
    
    Returns:
        Config object for quick testing
    """
    return Config(
        trainer=TrainerConfig(
            batch_size=4,
            epochs=10,
            early_stopping_patience=5,
        ),
        preprocessor=PreprocessorConfig(
            end_row=20,  # Only process first 20 subjects
        ),
    )


# ==================== Configuration Templates ====================

def create_example_config(output_path: str = "example_config.yaml") -> None:
    """
    Create an example configuration file with comments.
    
    Args:
        output_path: Where to save the example config
    """
    example_yaml = """# Brain Age Prediction Pipeline Configuration
# Generated example configuration

project_name: Brain_Age_Prediction
description: Universal brain age prediction pipeline
version: 1.0.0

# ==================== Path Configuration ====================
paths:
  # Input data paths
  metadata_file: data/IXI/IXI.xls       # Metadata file (Excel/CSV/TSV/JSON)
  t1_root: data/IXI/IXI-T1               # T1 MRI images directory
  t2_root: null                          # T2 MRI images directory (optional)
  flair_root: null                       # FLAIR images directory (optional)
  template: templates/MNI152_T1_1mm.nii.gz  # Registration template
  
  # Output paths
  output_root: outputs                   # Root output directory
  experiment_name: exp001                # Experiment identifier

# ==================== Indexer Configuration ====================
indexer:
  modalities: [T1]                       # List of modalities to index
  recursive: true                        # Search subdirectories recursively
  min_age: 0.0                          # Minimum valid age
  max_age: 120.0                        # Maximum valid age
  require_all_modalities: false          # Require all modalities present

# ==================== Preprocessor Configuration ====================
preprocessor:
  modalities: null                       # Auto-detect from CSV (or specify: [T1, T2])
  registration_type: Rigid               # Registration type: Rigid/Affine/SyN
  n4_iterations: [50, 50, 50, 50]       # N4 bias correction iterations per level
  skip_existing: true                    # Skip already processed subjects
  save_intermediate: false               # Save N4-only results
  start_row: 0                          # Start row (for batch processing)
  end_row: null                         # End row (null = all rows)

# ==================== Splitter Configuration ====================
splitter:
  train_ratio: 0.6                      # Training set ratio
  val_ratio: 0.2                        # Validation set ratio
  test_ratio: 0.2                       # Test set ratio
  stratify_age: true                    # Stratify by age (recommended)
  age_bins: 5                           # Number of age bins for stratification
  seed: 42                              # Random seed for reproducibility
  require_complete: true                # Only include complete preprocessed data

# ==================== Model Configuration ====================
model:
  model_name: DenseNet121               # Model architecture
  spatial_dims: 3                       # 3D images
  in_channels: 1                        # Single modality input
  out_channels: 1                       # Regression output
  pretrained: false                     # Use pretrained weights

# ==================== Trainer Configuration ====================
trainer:
  # Data settings
  modality: T1                          # Which modality to use for training
  target_size: [160, 192, 160]         # Target image size (H, W, D)
  batch_size: 2                         # Batch size
  num_workers: 4                        # Data loading workers
  
  # Training settings
  epochs: 500                           # Maximum epochs
  lr: 0.0001                            # Learning rate
  weight_decay: 0.00001                 # Weight decay
  
  # Scheduler and early stopping
  early_stopping_patience: 15           # Early stopping patience (epochs)
  lr_scheduler_patience: 5              # LR scheduler patience (epochs)
  lr_scheduler_factor: 0.5              # LR reduction factor
  
  # Reproducibility
  seed: 42                              # Random seed
  
  # Execution
  resume: false                         # Resume from checkpoint
"""
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(example_yaml)
    
    print(f"Example configuration saved to: {output_path}")
    print("\nTo use this configuration:")
    print(f"  python main.py --config {output_path} --run_all")


# ==================== Main for Testing ====================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Configuration Management')
    parser.add_argument('--create_example', action='store_true',
                        help='Create example configuration file')
    parser.add_argument('--preset', choices=['ixi', 'corr', 'test'],
                        help='Create preset configuration')
    parser.add_argument('--output', default='example_config.yaml',
                        help='Output path for configuration file')
    parser.add_argument('--validate', help='Validate a configuration file')
    
    args = parser.parse_args()
    
    if args.create_example:
        create_example_config(args.output)
    
    elif args.preset:
        if args.preset == 'ixi':
            cfg = get_ixi_config()
        elif args.preset == 'corr':
            cfg = get_corr_config()
        elif args.preset == 'test':
            cfg = get_quick_test_config()
        
        cfg.save_yaml(args.output)
        print(f"Preset configuration '{args.preset}' saved to: {args.output}")
    
    elif args.validate:
        try:
            cfg = Config.from_yaml(args.validate)
            cfg.validate()
            print(f"✓ Configuration is valid: {args.validate}")
        except Exception as e:
            print(f"✗ Configuration is invalid: {str(e)}")
    
    else:
        parser.print_help()