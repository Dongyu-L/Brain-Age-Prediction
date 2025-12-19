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
from .dataset import BrainAgeDataset


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
    """Universal brain age prediction trainer"""
    
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
        self.criterion: Optional[nn.Module] = None
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        self.early_stopping: Optional[EarlyStopping] = None
        
        # Training history
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
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
        )
        val_ds = BrainAgeDataset(
            csv_path=str(self.val_csv),
            target_size=self.target_size,
            modality=self.modality,
        )
        test_ds = BrainAgeDataset(
            csv_path=str(self.test_csv),
            target_size=self.target_size,
            modality=self.modality,
        )
        
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
    
    def setup_model(self) -> None:
        """Setup model, optimizer, scheduler, and loss"""
        logging.info("")
        logging.info("=" * 60)
        logging.info("Setting up model")
        logging.info("=" * 60)
        
        # Model
        self.model = DenseNet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,  # Regression
        ).to(self.device)
        
        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters())
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(f"Model: DenseNet121")
        logging.info(f"Total parameters: {n_params:,}")
        logging.info(f"Trainable parameters: {n_trainable:,}")
        
        # Loss
        self.criterion = nn.L1Loss()
        logging.info(f"Loss function: L1Loss (MAE)")
        
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
            verbose=True,
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
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        loop = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.epochs}",
            leave=False,
        )
        
        for x, age in loop:
            x = x.to(self.device)
            age = age.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with AMP
            with torch.cuda.amp.autocast():
                pred = self.model(x)
                loss = self.criterion(pred, age)
            
            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item() * x.size(0)
            loop.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / len(self.train_loader.dataset)
        return avg_loss
    
    def evaluate(self, loader: DataLoader) -> float:
        """Evaluate on a dataloader"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for x, age in loader:
                x = x.to(self.device)
                age = age.to(self.device)
                
                with torch.cuda.amp.autocast():
                    pred = self.model(x)
                    loss = self.criterion(pred, age)
                
                total_loss += loss.item() * x.size(0)
        
        avg_loss = total_loss / len(loader.dataset)
        return avg_loss
    
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
            }
        }
        
        # Save last checkpoint
        last_path = self.checkpoint_dir / "last_checkpoint.pt"
        torch.save(checkpoint, last_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
            logging.info(f"New best model saved (Val MAE={self.val_losses[-1]:.3f})")
    
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
        
        # Training loop
        for epoch in range(self.start_epoch, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.evaluate(self.val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log
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
                logging.info(f"Best validation MAE: {self.best_val_loss:.3f}")
                logging.info("=" * 60)
                break
        
        # Save training history
        self._save_training_history()
    
    def test(self) -> float:
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
        test_loss = self.evaluate(self.test_loader)
        logging.info(f"Test MAE: {test_loss:.3f} years")
        
        # Save test results
        results = {
            'best_epoch': checkpoint['epoch'],
            'best_val_loss': checkpoint['best_val_loss'],
            'test_loss': test_loss,
        }
        
        with open(self.output_dir / "test_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"Test results saved to {self.output_dir / 'test_results.json'}")
        
        return test_loss
    
    def _save_training_history(self) -> None:
        """Save training history to CSV"""
        history_df = pd.DataFrame({
            'epoch': range(1, len(self.train_losses) + 1),
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
        })
        
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