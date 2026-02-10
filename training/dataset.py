"""
PyTorch Dataset for Brain Age Prediction

Automatically detects image path columns and loads preprocessed MRI data.
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import nibabel as nib
from monai.transforms import Compose, Resize, ScaleIntensity


class BrainAgeDataset(Dataset):
    """Universal brain age dataset that auto-detects modality columns and supports multi-task learning"""

    def __init__(
        self,
        csv_path: str,
        target_size: Tuple[int, int, int] = (160, 192, 160),
        modality: str = "T1",
        predict_gender: bool = True,
    ):
        """
        Args:
            csv_path: Path to split CSV (train/val/test)
            target_size: Target image size (H, W, D)
            modality: Modality to use (e.g., 'T1', 'T2')
            predict_gender: Whether to return gender labels for multi-task learning
        """
        self.csv_path = Path(csv_path)
        self.target_size = target_size
        self.modality = modality
        self.predict_gender = predict_gender

        # Load CSV
        self.df = pd.read_csv(csv_path)

        # Detect image path column
        self.img_col = self._detect_image_column()

        # Check for gender column
        self.has_gender = 'Gender' in self.df.columns and predict_gender

        # Validate data
        self._validate_data()

        # Setup transforms
        self.transforms = Compose([
            ScaleIntensity(minv=0.0, maxv=1.0),
            Resize(spatial_size=target_size, mode='trilinear'),
        ])

        logging.info(f"Loaded {len(self.df)} samples from {self.csv_path.name}")
        logging.info(f"  Using column: {self.img_col}")
        logging.info(f"  Target size: {self.target_size}")
        if self.has_gender:
            n_valid_gender = (self.df['Gender'] >= 0).sum()
            logging.info(f"  Gender labels available: {n_valid_gender}/{len(self.df)}")
    
    def _detect_image_column(self) -> str:
        """Auto-detect the correct image path column"""
        # Priority: preprocessed path > original path
        candidates = [
            f'{self.modality}_preprocessed_path',
            f'{self.modality}_path',
            'Path',  # Fallback for custom CSVs
        ]
        
        for col in candidates:
            if col in self.df.columns:
                # Check if column has valid paths
                n_valid = (~self.df[col].isna() & (self.df[col] != '')).sum()
                if n_valid > 0:
                    return col
        
        raise ValueError(
            f"Could not find image path column for modality '{self.modality}'.\n"
            f"Available columns: {list(self.df.columns)}\n"
            f"Suggestion: Ensure CSV contains '{self.modality}_preprocessed_path' or '{self.modality}_path'"
        )
    
    def _validate_data(self) -> None:
        """Validate dataset"""
        # Check required columns
        if 'Age' not in self.df.columns:
            raise ValueError(
                f"CSV missing 'Age' column.\n"
                f"Available columns: {list(self.df.columns)}"
            )
        
        # Check for NaN ages
        if self.df['Age'].isna().any():
            raise ValueError("Age column contains NaN values")
        
        # Check for missing image paths
        n_missing = (self.df[self.img_col].isna() | (self.df[self.img_col] == '')).sum()
        if n_missing > 0:
            raise ValueError(
                f"{n_missing}/{len(self.df)} samples have missing image paths in column '{self.img_col}'.\n"
                f"Suggestion: Run data_preprocessor.py or filter the CSV."
            )
        
        # Check if files exist (sample check)
        sample_size = min(5, len(self.df))
        for idx in range(sample_size):
            img_path = Path(self.df[self.img_col].iloc[idx])
            if not img_path.exists():
                raise FileNotFoundError(
                    f"Image file not found: {img_path}\n"
                    f"Suggestion: Verify paths in CSV are correct and files exist."
                )
    
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        """
        Returns:
            If predict_gender=False: (img, age_tensor)
            If predict_gender=True: (img, age_tensor, gender_tensor)
                - gender_tensor: 0=Female, 1=Male, -1=Unknown
        """
        row = self.df.iloc[idx]
        age = float(row['Age'])
        img_path = Path(row[self.img_col])

        # Load image
        try:
            nii_img = nib.load(str(img_path))
            img = nii_img.get_fdata(dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {str(e)}")

        # Convert to tensor and add channel dimension
        img = torch.from_numpy(img).unsqueeze(0)  # [1, H, W, D]

        # Apply transforms
        img = self.transforms(img)

        age_tensor = torch.tensor([age], dtype=torch.float32)

        if self.has_gender:
            gender = row['Gender']
            # Gender: 0=Female, 1=Male, -1=Unknown
            gender_tensor = torch.tensor([float(gender)], dtype=torch.float32)
            return img, age_tensor, gender_tensor

        return img, age_tensor