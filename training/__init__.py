"""
Training modules for brain age prediction.

This package contains:
- dataset: PyTorch Dataset class for MRI brain age data
- trainer: Complete training pipeline with DenseNet121
"""

from .dataset import BrainAgeDataset
from .trainer import BrainAgeTrainer, EarlyStopping

__all__ = [
    'BrainAgeDataset',
    'BrainAgeTrainer',
    'EarlyStopping',
]

__version__ = '1.0.0'