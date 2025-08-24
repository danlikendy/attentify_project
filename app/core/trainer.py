"""
Transformer Training Module

This module implements the training loop for the Transformer model,
including the learning rate schedule and optimization strategy
described in the "Attention Is All You Need" paper.
"""

import math
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm_
import numpy as np
from tqdm import tqdm

from .transformer import Transformer
from .text_processor import Vocabulary, TextProcessor


class TransformerDataset(Dataset):
    """
    Dataset class for Transformer training.
    """
    
    def __init__(self, src_texts: List[str], tgt_texts: List[str],
                 src_vocab: Vocabulary, tgt_vocab: Vocabulary,
                 max_src_len: int = 512, max_tgt_len: int = 512):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        
        # Preprocess all texts
        self.processor = TextProcessor(src_vocab, tgt_vocab)
        self.processor.max_src_length = max_src_len
        self.processor.max_tgt_length = max_tgt_len
    
    def __len__(self) -> int:
        return len(self.src_texts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]
        
        # Process single example
        src_tensor, tgt_tensor = self.processor.prepare_batch(
            [src_text], [tgt_text], 'translation')
        
        return src_tensor.squeeze(0), tgt_tensor.squeeze(0)


class WarmupCosineScheduler:
    """
    Learning rate scheduler with warmup and cosine annealing.
    
    As described in the paper:
    lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
    """
    
    def __init__(self, optimizer: optim.Optimizer, d_model: int, warmup_steps: int,
                 max_steps: int, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.current_step = 0
        
        # Base learning rate
        self.base_lr = d_model ** (-0.5)
    
    def step(self):
        """Update learning rate."""
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Warmup phase
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine annealing phase
            progress = (self.current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            lr = self.base_lr * (0.5 * (1 + math.cos(math.pi * progress)))
        
        # Apply minimum learning rate
        lr = max(lr, self.min_lr)
        
        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_last_lr(self) -> List[float]:
        """Get current learning rate."""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


class TransformerTrainer:
    """
    Main training class for the Transformer model.
    """
    
    def __init__(self, model: Transformer, src_vocab: Vocabulary, tgt_vocab: Vocabulary,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 max_grad_norm: float = 1.0):
        self.model = model.to(device)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = device
        self.max_grad_norm = max_grad_norm
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=tgt_vocab.token2idx[tgt_vocab.pad_token],
            label_smoothing=0.1  # As mentioned in the paper
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            betas=(0.9, 0.98),  # As in the paper
            eps=1e-9,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = None
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_loss = float('inf')
        
        # Logging
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_steps = len(dataloader)
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        for batch_idx, (src, tgt) in enumerate(progress_bar):
            # Move to device
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Create target for loss calculation (shifted by 1)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # Forward pass
            output = self.model(src, tgt_input)
            
            # Calculate loss
            loss = self.criterion(
                output.contiguous().view(-1, output.size(-1)),
                tgt_output.contiguous().view(-1)
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Optimizer step
            self.optimizer.step()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            self.current_step += 1
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{current_lr:.2e}',
                'Step': self.current_step
            })
        
        avg_loss = total_loss / total_steps
        return {'train_loss': avg_loss}
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            dataloader: Validation data loader
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_steps = len(dataloader)
        
        with torch.no_grad():
            for src, tgt in dataloader:
                # Move to device
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                
                # Create target for loss calculation
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                # Forward pass
                output = self.model(src, tgt_input)
                
                # Calculate loss
                loss = self.criterion(
                    output.contiguous().view(-1, output.size(-1)),
                    tgt_output.contiguous().view(-1)
                )
                
                total_loss += loss.item()
        
        avg_loss = total_loss / total_steps
        return {'val_loss': avg_loss}
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader,
              num_epochs: int, warmup_steps: int = 4000, save_dir: str = './checkpoints',
              save_every: int = 1, early_stopping_patience: int = 5) -> Dict[str, List[float]]:
        """
        Main training loop.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            num_epochs: Number of training epochs
            warmup_steps: Number of warmup steps for learning rate
            save_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs
            early_stopping_patience: Early stopping patience
        
        Returns:
            Dictionary with training history
        """
        # Setup learning rate scheduler
        total_steps = len(train_dataloader) * num_epochs
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            self.model.d_model,
            warmup_steps,
            total_steps
        )
        
        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_dataloader, epoch)
            
            # Validation
            val_metrics = self.validate(val_dataloader)
            
            # Update history
            history['train_loss'].append(train_metrics['train_loss'])
            history['val_loss'].append(val_metrics['val_loss'])
            history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Log metrics
            self.logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.4f} - "
                f"Val Loss: {val_metrics['val_loss']:.4f} - "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
            )
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(save_path / f'checkpoint_epoch_{epoch + 1}.pt')
            
            # Early stopping check
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                patience_counter = 0
                # Save best model
                self.save_checkpoint(save_path / 'best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        self.logger.info("Training completed!")
        return history
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_loss': self.best_loss,
            'src_vocab': self.src_vocab,
            'tgt_vocab': self.tgt_vocab,
            'model_config': {
                'd_model': self.model.d_model,
                'n_layers': self.model.n_layers,
                'n_heads': self.model.n_heads,
                'd_ff': self.model.encoder_layers[0].feed_forward.linear1.out_features
            }
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.current_step = checkpoint['step']
        self.best_loss = checkpoint['best_loss']
        
        self.logger.info(f"Checkpoint loaded from {filepath}")
        self.logger.info(f"Resuming from epoch {self.current_epoch}, step {self.current_step}")
    
    def generate_text(self, src_text: str, max_length: int = 100,
                     beam_size: int = 4, length_penalty: float = 0.6) -> str:
        """
        Generate text using the trained model.
        
        Args:
            src_text: Source text
            max_length: Maximum generation length
            beam_size: Beam search size
            length_penalty: Length penalty for beam search
        
        Returns:
            Generated text
        """
        self.model.eval()
        
        # Preprocess source text
        processor = TextProcessor(self.src_vocab, self.tgt_vocab)
        src_tensor, _ = processor.prepare_batch([src_text], task='inference')
        src_tensor = src_tensor.to(self.device)
        
        # Initialize target sequence
        batch_size = src_tensor.size(0)
        tgt = torch.full((batch_size, 1), 
                        self.tgt_vocab.token2idx[self.tgt_vocab.sos_token],
                        dtype=torch.long, device=self.device)
        
        # Beam search
        with torch.no_grad():
            for _ in range(max_length - 1):
                # Forward pass
                output = self.model(src_tensor, tgt)
                next_token_logits = output[:, -1, :]
                
                # Get top-k tokens
                top_k_logits, top_k_indices = torch.topk(next_token_logits, beam_size, dim=-1)
                
                # Select next token (greedy for simplicity, can be enhanced with beam search)
                next_token = top_k_logits.argmax(dim=-1, keepdim=True)
                
                # Append to target sequence
                tgt = torch.cat([tgt, next_token], dim=1)
                
                # Check if EOS token is generated
                if (tgt == self.tgt_vocab.token2idx[self.tgt_vocab.eos_token]).any():
                    break
        
        # Decode generated sequence
        generated_text = self.tgt_vocab.decode(tgt[0].tolist())
        return generated_text
    
    def get_attention_weights(self, src_text: str, tgt_text: str) -> Dict[str, torch.Tensor]:
        """
        Get attention weights for visualization.
        
        Args:
            src_text: Source text
            tgt_text: Target text
        
        Returns:
            Dictionary with attention weights from all layers
        """
        self.model.eval()
        
        # Preprocess texts
        processor = TextProcessor(self.src_vocab, self.tgt_vocab)
        src_tensor, _ = processor.prepare_batch([src_text], task='inference')
        tgt_tensor, _ = processor.prepare_batch([tgt_text], task='inference')
        src_tensor = src_tensor.to(self.device)
        tgt_tensor = tgt_tensor.to(self.device)
        
        # Get attention weights
        with torch.no_grad():
            attention_weights = self.model.get_attention_weights(src_tensor, tgt_tensor)
        
        return attention_weights
