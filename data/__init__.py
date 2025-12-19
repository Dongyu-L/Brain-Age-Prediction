"""
Data processing modules for brain age prediction pipeline.

This package contains:
- indexer: Create CSV index from raw MRI data
- preprocessor: ANTs-based preprocessing (N4 + registration)
- splitter: Train/val/test dataset splitting with stratification
"""

from .indexer import UniversalIndexer
from .preprocessor import UniversalPreprocessor
from .splitter import UniversalSplitter

__all__ = [
    'UniversalIndexer',
    'UniversalPreprocessor', 
    'UniversalSplitter',
]

__version__ = '1.0.0'