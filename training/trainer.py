"""
Universal Brain Age Prediction Trainer

A flexible training pipeline that works with any preprocessed dataset.

Usage:
    # Basic usage
    python -m training.trainer \\
        --train_csv splits/train_split.csv \\
        --val_csv splits/val_split.csv \\
        --test_csv splits/test_split.csv \\
        --output_dir experiments/exp001
    
    # With custom hyperparameters
    python -m training.trainer \\
        --train_csv splits/train_split.csv \\
        --val_csv splits/val_split.csv \\
        --test_csv splits/test_split.csv \\
        --output_dir experiments/exp001 \\
        --epochs 100 \\
        --batch_size 4 \\
        --lr 1e-4
"""

import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from monai.networks.nets import DenseNet121
from monai.utils import set_determinism

# Import dataset from separate module
try:
    from .dataset import BrainAgeDataset
except ImportError:
    from dataset import BrainAgeDataset


class MultiTaskBrainModel(nn.Module):
    """
    Multi-task model for brain age prediction and gender classification.

    Architecture:
        - Shared backbone: DenseNet121 (3D)
        - Age head: Linear layer for regression
        - Gender head: Linear layer + Sigmoid for binary classification
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        backbone_features: int = 1024,
        predict_gender: bool = True,
    ):
        """
        Args:
            spatial_dims: Number of spatial dimensions (3 for 3D images)
            in_channels: Number of input channels
            backbone_features: Number of features from backbone
            predict_gender: Whether to include gender prediction head
        """
        super().__init__()

        self.predict_gender = predict_gender

        # Shared backbone - DenseNet121 outputs backbone_features features
        self.backbone = DenseNet121(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=backbone_features,
        )

        # Age regression head
        self.age_head = nn.Linear(backbone_features, 1)

        # Gender classification head (binary: 0=Female, 1=Male)
        # Note: No Sigmoid here - use BCEWithLogitsLoss for AMP compatibility
        if predict_gender:
            self.gender_head = nn.Linear(backbone_features, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, C, H, W, D]

        Returns:
            Dictionary with keys:
                - 'age': Age predictions [B, 1]
                - 'gender': Gender predictions [B, 1] (if predict_gender=True)
        """
        # Shared features
        features = self.backbone(x)

        # Age prediction
        age_pred = self.age_head(features)

        outputs = {'age': age_pred}

        # Gender prediction (if enabled)
        if self.predict_gender:
            gender_pred = self.gender_head(features)
            outputs['gender'] = gender_pred

        return outputs


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 15, min_delta: float = 0.001, verbose: bool = True):
        """
        Args:
            patience: Number of epochs with no improvement before stopping
            min_delta: Minimum change to qualify as improvement
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> None:
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.verbose:
                logging.info(f"Initial best loss: {self.best_loss:.4f}")
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                logging.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logging.info("Early stopping triggered!")
        else:
            if self.verbose:
                logging.info(f"Validation loss improved: {self.best_loss:.4f} -> {val_loss:.4f}")
            self.best_loss = val_loss
            self.counter = 0


class BrainAgeTrainer:
    """Universal brain age prediction trainer with multi-task support"""

    def __init__(
        self,
        train_csv: str,
        val_csv: str,
        test_csv: str,
        output_dir: str,
        modality: str = "T1",
        target_size: Tuple[int, int, int] = (160, 192, 160),
        batch_size: int = 2,
        num_workers: int = 4,
        epochs: int = 500,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        early_stopping_patience: int = 15,
        lr_scheduler_patience: int = 5,
        seed: int = 42,
        predict_gender: bool = True,
        gender_loss_weight: float = 0.5,
    ):
        """
        Args:
            train_csv: Path to training CSV
            val_csv: Path to validation CSV
            test_csv: Path to test CSV
            output_dir: Output directory for checkpoints and logs
            modality: Image modality to use (e.g., 'T1', 'T2')
            target_size: Target image size for resizing
            batch_size: Batch size for training
            num_workers: Number of data loading workers
            epochs: Maximum number of training epochs
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            early_stopping_patience: Patience for early stopping
            lr_scheduler_patience: Patience for learning rate scheduler
            seed: Random seed for reproducibility
            predict_gender: Whether to enable gender prediction (multi-task)
            gender_loss_weight: Weight for gender classification loss
        """
        self.train_csv = Path(train_csv)
        self.val_csv = Path(val_csv)
        self.test_csv = Path(test_csv)
        self.output_dir = Path(output_dir)
        self.modality = modality
        self.target_size = target_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.lr_scheduler_patience = lr_scheduler_patience
        self.seed = seed
        self.predict_gender = predict_gender
        self.gender_loss_weight = gender_loss_weight

        # Setup
        self._setup_output_dir()
        self._setup_device()
        self._set_seed()

        # To be initialized
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.age_criterion: Optional[nn.Module] = None
        self.gender_criterion: Optional[nn.Module] = None
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        self.early_stopping: Optional[EarlyStopping] = None

        # Training history
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_age_losses: List[float] = []
        self.train_gender_losses: List[float] = []
        self.val_age_losses: List[float] = []
        self.val_gender_losses: List[float] = []
        self.train_gender_accs: List[float] = []
        self.val_gender_accs: List[float] = []
        self.best_val_loss: float = float('inf')
        self.start_epoch: int = 1
    
    def _setup_output_dir(self) -> None:
        """Create output directory structure"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Setup logging
        log_file = self.output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
        
        logging.info(f"Output directory: {self.output_dir}")
        logging.info(f"Checkpoint directory: {self.checkpoint_dir}")
        logging.info(f"Log file: {log_file}")
    
    def _setup_device(self) -> None:
        """Setup computation device"""
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            logging.warning("CUDA not available! Using CPU (this will be slow)")
    
    def _set_seed(self) -> None:
        """Set random seeds for reproducibility"""
        set_determinism(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        logging.info(f"Random seed set to: {self.seed}")
    
    def setup_data(self) -> None:
        """Setup datasets and dataloaders"""
        logging.info("")
        logging.info("=" * 60)
        logging.info("Setting up datasets")
        logging.info("=" * 60)

        # Create datasets
        train_ds = BrainAgeDataset(
            csv_path=str(self.train_csv),
            target_size=self.target_size,
            modality=self.modality,
            predict_gender=self.predict_gender,
        )
        val_ds = BrainAgeDataset(
            csv_path=str(self.val_csv),
            target_size=self.target_size,
            modality=self.modality,
            predict_gender=self.predict_gender,
        )
        test_ds = BrainAgeDataset(
            csv_path=str(self.test_csv),
            target_size=self.target_size,
            modality=self.modality,
            predict_gender=self.predict_gender,
        )

        # Check if gender data is actually available
        self.has_gender_data = train_ds.has_gender and self.predict_gender
        if self.predict_gender and not train_ds.has_gender:
            logging.warning("Gender prediction enabled but no gender data found in CSV. "
                          "Training will proceed with age prediction only.")
            self.predict_gender = False

        # Create dataloaders
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        logging.info(f"Train batches: {len(self.train_loader)}")
        logging.info(f"Val batches: {len(self.val_loader)}")
        logging.info(f"Test batches: {len(self.test_loader)}")
        logging.info(f"Multi-task (gender prediction): {self.has_gender_data}")
    
    def setup_model(self) -> None:
        """Setup model, optimizer, scheduler, and loss"""
        logging.info("")
        logging.info("=" * 60)
        logging.info("Setting up model")
        logging.info("=" * 60)

        # Check if we should use multi-task model
        use_multitask = self.predict_gender and self.has_gender_data

        if use_multitask:
            # Multi-task model for age + gender prediction
            self.model = MultiTaskBrainModel(
                spatial_dims=3,
                in_channels=1,
                backbone_features=1024,
                predict_gender=True,
            ).to(self.device)
            logging.info(f"Model: MultiTaskBrainModel (DenseNet121 backbone)")
            logging.info(f"  Tasks: Age regression + Gender classification")
            logging.info(f"  Gender loss weight: {self.gender_loss_weight}")
        else:
            # Single-task model for age prediction only
            self.model = DenseNet121(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,  # Regression
            ).to(self.device)
            logging.info(f"Model: DenseNet121 (single-task)")
            logging.info(f"  Task: Age regression only")

        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters())
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {n_params:,}")
        logging.info(f"Trainable parameters: {n_trainable:,}")

        # Loss functions
        self.age_criterion = nn.L1Loss()  # MAE for age regression
        logging.info(f"Age loss: L1Loss (MAE)")

        if use_multitask:
            self.gender_criterion = nn.BCEWithLogitsLoss()  # AMP-safe binary cross-entropy
            logging.info(f"Gender loss: BCEWithLogitsLoss")

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        logging.info(f"Optimizer: AdamW (lr={self.lr}, weight_decay={self.weight_decay})")

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.lr_scheduler_patience,
            factor=0.5,
        )
        logging.info(f"LR Scheduler: ReduceLROnPlateau (patience={self.lr_scheduler_patience})")

        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler()

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.early_stopping_patience,
            min_delta=0.001,
            verbose=True,
        )
        logging.info(f"Early stopping patience: {self.early_stopping_patience}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with multi-task support"""
        self.model.train()
        total_loss = 0.0
        total_age_loss = 0.0
        total_gender_loss = 0.0
        total_gender_correct = 0
        total_gender_samples = 0

        use_multitask = self.predict_gender and self.has_gender_data

        loop = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.epochs}",
            leave=False,
        )

        for batch in loop:
            if use_multitask:
                x, age, gender = batch
                gender = gender.to(self.device)
            else:
                x, age = batch
                gender = None

            x = x.to(self.device)
            age = age.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with AMP
            with torch.cuda.amp.autocast():
                if use_multitask:
                    outputs = self.model(x)
                    age_pred = outputs['age']
                    gender_pred = outputs['gender']

                    # Age loss
                    age_loss = self.age_criterion(age_pred, age)

                    # Gender loss (only for valid gender labels, i.e., gender >= 0)
                    valid_gender_mask = gender.squeeze() >= 0
                    if valid_gender_mask.sum() > 0:
                        valid_gender_pred = gender_pred[valid_gender_mask]
                        valid_gender_true = gender[valid_gender_mask]
                        gender_loss = self.gender_criterion(valid_gender_pred, valid_gender_true)

                        # Calculate accuracy (logits > 0 means prob > 0.5)
                        gender_preds_binary = (valid_gender_pred > 0).float()
                        total_gender_correct += (gender_preds_binary == valid_gender_true).sum().item()
                        total_gender_samples += valid_gender_mask.sum().item()
                    else:
                        gender_loss = torch.tensor(0.0, device=self.device)

                    # Combined loss
                    loss = age_loss + self.gender_loss_weight * gender_loss
                    total_age_loss += age_loss.item() * x.size(0)
                    total_gender_loss += gender_loss.item() * valid_gender_mask.sum().item() if valid_gender_mask.sum() > 0 else 0
                else:
                    pred = self.model(x)
                    loss = self.age_criterion(pred, age)
                    total_age_loss += loss.item() * x.size(0)

            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * x.size(0)

            if use_multitask:
                loop.set_postfix({
                    "loss": loss.item(),
                    "age": age_loss.item(),
                    "gender": gender_loss.item()
                })
            else:
                loop.set_postfix({"loss": loss.item()})

        n_samples = len(self.train_loader.dataset)
        metrics = {
            'total_loss': total_loss / n_samples,
            'age_loss': total_age_loss / n_samples,
        }

        if use_multitask and total_gender_samples > 0:
            metrics['gender_loss'] = total_gender_loss / total_gender_samples
            metrics['gender_acc'] = total_gender_correct / total_gender_samples
        elif use_multitask:
            metrics['gender_loss'] = 0.0
            metrics['gender_acc'] = 0.0

        return metrics
    
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate on a dataloader with multi-task support"""
        self.model.eval()
        total_loss = 0.0
        total_age_loss = 0.0
        total_gender_loss = 0.0
        total_gender_correct = 0
        total_gender_samples = 0

        use_multitask = self.predict_gender and self.has_gender_data

        with torch.no_grad():
            for batch in loader:
                if use_multitask:
                    x, age, gender = batch
                    gender = gender.to(self.device)
                else:
                    x, age = batch
                    gender = None

                x = x.to(self.device)
                age = age.to(self.device)

                with torch.cuda.amp.autocast():
                    if use_multitask:
                        outputs = self.model(x)
                        age_pred = outputs['age']
                        gender_pred = outputs['gender']

                        # Age loss
                        age_loss = self.age_criterion(age_pred, age)

                        # Gender loss (only for valid gender labels)
                        valid_gender_mask = gender.squeeze() >= 0
                        if valid_gender_mask.sum() > 0:
                            valid_gender_pred = gender_pred[valid_gender_mask]
                            valid_gender_true = gender[valid_gender_mask]
                            gender_loss = self.gender_criterion(valid_gender_pred, valid_gender_true)

                            # Calculate accuracy (logits > 0 means prob > 0.5)
                            gender_preds_binary = (valid_gender_pred > 0).float()
                            total_gender_correct += (gender_preds_binary == valid_gender_true).sum().item()
                            total_gender_samples += valid_gender_mask.sum().item()
                        else:
                            gender_loss = torch.tensor(0.0, device=self.device)

                        # Combined loss
                        loss = age_loss + self.gender_loss_weight * gender_loss
                        total_age_loss += age_loss.item() * x.size(0)
                        total_gender_loss += gender_loss.item() * valid_gender_mask.sum().item() if valid_gender_mask.sum() > 0 else 0
                    else:
                        pred = self.model(x)
                        loss = self.age_criterion(pred, age)
                        total_age_loss += loss.item() * x.size(0)

                total_loss += loss.item() * x.size(0)

        n_samples = len(loader.dataset)
        metrics = {
            'total_loss': total_loss / n_samples,
            'age_loss': total_age_loss / n_samples,
        }

        if use_multitask and total_gender_samples > 0:
            metrics['gender_loss'] = total_gender_loss / total_gender_samples
            metrics['gender_acc'] = total_gender_correct / total_gender_samples
        elif use_multitask:
            metrics['gender_loss'] = 0.0
            metrics['gender_acc'] = 0.0

        return metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': {
                'modality': self.modality,
                'target_size': self.target_size,
                'batch_size': self.batch_size,
                'lr': self.lr,
                'weight_decay': self.weight_decay,
                'seed': self.seed,
                'predict_gender': self.predict_gender,
                'gender_loss_weight': self.gender_loss_weight,
                'has_gender_data': self.has_gender_data if hasattr(self, 'has_gender_data') else False,
            }
        }

        # Save last checkpoint
        last_path = self.checkpoint_dir / "last_checkpoint.pt"
        torch.save(checkpoint, last_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
            logging.info(f"New best model saved (Val Loss={self.val_losses[-1]:.3f})")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load checkpoint for resuming training"""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logging.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        self.start_epoch = checkpoint['epoch'] + 1
        
        logging.info(f"Resumed from epoch {checkpoint['epoch']}")
        logging.info(f"Best val loss so far: {self.best_val_loss:.3f}")
    
    def train(self, resume: bool = False) -> None:
        """Main training loop"""
        logging.info("")
        logging.info("=" * 60)
        logging.info("Starting Training")
        logging.info("=" * 60)
        logging.info(f"Max epochs: {self.epochs}")
        logging.info(f"Early stopping patience: {self.early_stopping_patience}")
        logging.info(f"LR scheduler patience: {self.lr_scheduler_patience}")
        logging.info("")
        
        # Resume if requested
        if resume:
            last_checkpoint = self.checkpoint_dir / "last_checkpoint.pt"
            if last_checkpoint.exists():
                self.load_checkpoint(str(last_checkpoint))
            else:
                logging.warning("Resume requested but no checkpoint found. Starting from scratch.")
        
        use_multitask = self.predict_gender and self.has_gender_data

        # Training loop
        for epoch in range(self.start_epoch, self.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader)

            train_loss = train_metrics['total_loss']
            val_loss = val_metrics['total_loss']

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_age_losses.append(train_metrics['age_loss'])
            self.val_age_losses.append(val_metrics['age_loss'])

            if use_multitask:
                self.train_gender_losses.append(train_metrics.get('gender_loss', 0))
                self.val_gender_losses.append(val_metrics.get('gender_loss', 0))
                self.train_gender_accs.append(train_metrics.get('gender_acc', 0))
                self.val_gender_accs.append(val_metrics.get('gender_acc', 0))

            # Update scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log
            if use_multitask:
                logging.info(
                    f"Epoch {epoch}: "
                    f"Train Loss={train_loss:.3f} (Age={train_metrics['age_loss']:.3f}, Gender={train_metrics.get('gender_loss', 0):.3f}), "
                    f"Val Loss={val_loss:.3f} (Age={val_metrics['age_loss']:.3f}, Gender={val_metrics.get('gender_loss', 0):.3f}), "
                    f"Gender Acc: Train={train_metrics.get('gender_acc', 0)*100:.1f}%, Val={val_metrics.get('gender_acc', 0)*100:.1f}%, "
                    f"LR={current_lr:.2e}"
                )
            else:
                logging.info(
                    f"Epoch {epoch}: "
                    f"Train MAE={train_loss:.3f}, "
                    f"Val MAE={val_loss:.3f}, "
                    f"LR={current_lr:.2e}"
                )

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            self.save_checkpoint(epoch, is_best=is_best)

            # Early stopping check
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                logging.info("")
                logging.info("=" * 60)
                logging.info(f"Early stopping at epoch {epoch}")
                logging.info(f"Best validation loss: {self.best_val_loss:.3f}")
                logging.info("=" * 60)
                break

        # Save training history
        self._save_training_history()
    
    def test(self) -> Dict[str, float]:
        """Evaluate on test set using best model"""
        logging.info("")
        logging.info("=" * 60)
        logging.info("Testing")
        logging.info("=" * 60)

        # Load best checkpoint
        best_checkpoint = self.checkpoint_dir / "best_checkpoint.pt"
        if not best_checkpoint.exists():
            raise FileNotFoundError(
                f"Best checkpoint not found: {best_checkpoint}\n"
                f"Suggestion: Complete training first."
            )

        checkpoint = torch.load(best_checkpoint, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Loaded best model from epoch {checkpoint['epoch']}")

        # Evaluate
        test_metrics = self.evaluate(self.test_loader)
        use_multitask = self.predict_gender and self.has_gender_data

        logging.info(f"Test Age MAE: {test_metrics['age_loss']:.3f} years")
        if use_multitask:
            logging.info(f"Test Gender Accuracy: {test_metrics.get('gender_acc', 0)*100:.1f}%")

        # Save test results
        results = {
            'best_epoch': checkpoint['epoch'],
            'best_val_loss': checkpoint['best_val_loss'],
            'test_total_loss': test_metrics['total_loss'],
            'test_age_mae': test_metrics['age_loss'],
        }

        if use_multitask:
            results['test_gender_loss'] = test_metrics.get('gender_loss', 0)
            results['test_gender_accuracy'] = test_metrics.get('gender_acc', 0)

        with open(self.output_dir / "test_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        logging.info(f"Test results saved to {self.output_dir / 'test_results.json'}")

        return test_metrics
    
    def _save_training_history(self) -> None:
        """Save training history to CSV"""
        use_multitask = self.predict_gender and self.has_gender_data

        history_data = {
            'epoch': range(1, len(self.train_losses) + 1),
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'train_age_loss': self.train_age_losses,
            'val_age_loss': self.val_age_losses,
        }

        if use_multitask:
            history_data['train_gender_loss'] = self.train_gender_losses
            history_data['val_gender_loss'] = self.val_gender_losses
            history_data['train_gender_acc'] = self.train_gender_accs
            history_data['val_gender_acc'] = self.val_gender_accs

        history_df = pd.DataFrame(history_data)

        history_path = self.output_dir / "training_history.csv"
        history_df.to_csv(history_path, index=False)
        logging.info(f"Training history saved to {history_path}")
    
    def run(self, resume: bool = False) -> None:
        """Complete training pipeline"""
        self.setup_data()
        self.setup_model()
        self.train(resume=resume)
        self.test()


def main():
    parser = argparse.ArgumentParser(
        description='Universal Brain Age Prediction Trainer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument('--train_csv', required=True,
                        help='Training CSV path')
    parser.add_argument('--val_csv', required=True,
                        help='Validation CSV path')
    parser.add_argument('--test_csv', required=True,
                        help='Test CSV path')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for checkpoints and logs')
    
    # Data arguments
    parser.add_argument('--modality', default='T1',
                        help='Image modality to use (default: T1)')
    parser.add_argument('--target_size', type=int, nargs=3,
                        default=[160, 192, 160],
                        metavar=('H', 'W', 'D'),
                        help='Target image size (default: 160 192 160)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=500,
                        help='Maximum number of epochs (default: 500)')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size (default: 2)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (default: 1e-5)')
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                        help='Early stopping patience (default: 15)')
    parser.add_argument('--lr_scheduler_patience', type=int, default=5,
                        help='LR scheduler patience (default: 5)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    # Multi-task arguments
    parser.add_argument('--predict_gender', action='store_true', default=True,
                        help='Enable gender prediction (multi-task learning, default: True)')
    parser.add_argument('--no_gender', action='store_true',
                        help='Disable gender prediction (age-only mode)')
    parser.add_argument('--gender_loss_weight', type=float, default=0.5,
                        help='Weight for gender classification loss (default: 0.5)')
    
    # Execution arguments
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last checkpoint')
    parser.add_argument('--test_only', action='store_true',
                        help='Only run testing (requires trained model)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s: %(message)s'
    )
    
    # Determine gender prediction mode
    predict_gender = args.predict_gender and not args.no_gender

    # Create trainer
    trainer = BrainAgeTrainer(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        output_dir=args.output_dir,
        modality=args.modality,
        target_size=tuple(args.target_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        lr_scheduler_patience=args.lr_scheduler_patience,
        seed=args.seed,
        predict_gender=predict_gender,
        gender_loss_weight=args.gender_loss_weight,
    )
    
    # Run
    try:
        if args.test_only:
            trainer.setup_data()
            trainer.setup_model()
            trainer.test()
        else:
            trainer.run(resume=args.resume)
    except Exception as e:
        logging.error(f"\nFATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())