"""
Text Processing and Tokenization Module

This module handles text preprocessing, tokenization, and data preparation
for the Transformer model, including support for multiple languages and tasks.
"""

import re
import json
import pickle
from typing import List, Dict, Tuple, Optional, Union
from collections import Counter
import torch
import numpy as np


class Vocabulary:
    """
    Vocabulary class for managing token-to-index and index-to-token mappings.
    """
    
    def __init__(self, min_freq: int = 2, max_size: int = 50000):
        self.min_freq = min_freq
        self.max_size = max_size
        self.token2idx = {}
        self.idx2token = {}
        self.token_freq = Counter()
        
        # Special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.sos_token = '<SOS>'
        self.eos_token = '<EOS>'
        
        # Initialize with special tokens
        self._add_special_tokens()
    
    def _add_special_tokens(self):
        """Add special tokens to vocabulary."""
        special_tokens = [self.pad_token, self.unk_token, self.sos_token, self.eos_token]
        for token in special_tokens:
            self.token2idx[token] = len(self.token2idx)
            self.idx2token[len(self.idx2token)] = token
    
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from list of texts."""
        # Count token frequencies
        for text in texts:
            tokens = self._tokenize(text)
            self.token_freq.update(tokens)
        
        # Add tokens that meet frequency threshold
        sorted_tokens = sorted(self.token_freq.items(), key=lambda x: x[1], reverse=True)
        
        for token, freq in sorted_tokens:
            if freq >= self.min_freq and len(self.token2idx) < self.max_size:
                if token not in self.token2idx:
                    self.token2idx[token] = len(self.token2idx)
                    self.idx2token[len(self.idx2token)] = token
    
    def _tokenize(self, text: str) -> List[str]:
        """Basic tokenization - split on whitespace and punctuation."""
        # Simple tokenization - can be enhanced with more sophisticated methods
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """Encode text to sequence of indices."""
        tokens = self._tokenize(text)
        
        # Add start and end tokens
        tokens = [self.sos_token] + tokens + [self.eos_token]
        
        # Convert to indices
        indices = []
        for token in tokens:
            if token in self.token2idx:
                indices.append(self.token2idx[token])
            else:
                indices.append(self.token2idx[self.unk_token])
        
        # Pad or truncate if max_length is specified
        if max_length is not None:
            if len(indices) < max_length:
                indices.extend([self.token2idx[self.pad_token]] * (max_length - len(indices)))
            else:
                indices = indices[:max_length]
        
        return indices
    
    def decode(self, indices: List[int]) -> str:
        """Decode sequence of indices back to text."""
        tokens = []
        for idx in indices:
            if idx in self.idx2token:
                token = self.idx2token[idx]
                if token not in [self.pad_token, self.sos_token, self.eos_token]:
                    tokens.append(token)
            else:
                tokens.append(self.unk_token)
        
        return ' '.join(tokens)
    
    def __len__(self) -> int:
        return len(self.token2idx)
    
    def save(self, filepath: str):
        """Save vocabulary to file."""
        vocab_data = {
            'token2idx': self.token2idx,
            'idx2token': self.idx2token,
            'token_freq': dict(self.token_freq),
            'min_freq': self.min_freq,
            'max_size': self.max_size
        }
        with open(filepath, 'wb') as f:
            pickle.dump(vocab_data, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'Vocabulary':
        """Load vocabulary from file."""
        with open(filepath, 'rb') as f:
            vocab_data = pickle.load(f)
        
        vocab = cls(vocab_data['min_freq'], vocab_data['max_size'])
        vocab.token2idx = vocab_data['token2idx']
        vocab.idx2token = vocab_data['idx2token']
        vocab.token_freq = Counter(vocab_data['token_freq'])
        
        return vocab


class TextProcessor:
    """
    Main text processing class for handling various NLP tasks.
    """
    
    def __init__(self, src_vocab: Vocabulary, tgt_vocab: Vocabulary):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_src_length = 512
        self.max_tgt_length = 512
    
    def preprocess_text(self, text: str, task: str = 'translation') -> str:
        """
        Preprocess text based on the specific task.
        
        Args:
            text: Input text
            task: Task type ('translation', 'summarization', 'simplification')
        """
        # Basic preprocessing
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        if task == 'summarization':
            # For summarization, keep sentences intact
            text = re.sub(r'[^\w\s\.\!\?]', '', text)
        elif task == 'simplification':
            # For simplification, remove complex punctuation
            text = re.sub(r'[^\w\s\.]', '', text)
        
        return text
    
    def prepare_batch(self, src_texts: List[str], tgt_texts: Optional[List[str]] = None,
                     task: str = 'translation') -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Prepare batch of texts for model input.
        
        Args:
            src_texts: List of source texts
            tgt_texts: Optional list of target texts (for training)
            task: Task type
        
        Returns:
            Tuple of (source_tensor, target_tensor)
        """
        # Preprocess source texts
        processed_src = [self.preprocess_text(text, task) for text in src_texts]
        
        # Encode source texts
        src_encoded = [self.src_vocab.encode(text, self.max_src_length) 
                      for text in processed_src]
        
        # Pad source sequences
        src_padded = self._pad_sequences(src_encoded, self.max_src_length)
        src_tensor = torch.tensor(src_padded, dtype=torch.long)
        
        if tgt_texts is not None:
            # Preprocess target texts
            processed_tgt = [self.preprocess_text(text, task) for text in tgt_texts]
            
            # Encode target texts
            tgt_encoded = [self.tgt_vocab.encode(text, self.max_tgt_length) 
                          for text in processed_tgt]
            
            # Pad target sequences
            tgt_padded = self._pad_sequences(tgt_encoded, self.max_tgt_length)
            tgt_tensor = torch.tensor(tgt_padded, dtype=torch.long)
            
            return src_tensor, tgt_tensor
        else:
            return src_tensor, None
    
    def _pad_sequences(self, sequences: List[List[int]], max_length: int) -> List[List[int]]:
        """Pad sequences to the same length."""
        padded = []
        for seq in sequences:
            if len(seq) < max_length:
                seq.extend([self.src_vocab.token2idx[self.src_vocab.pad_token]] * 
                          (max_length - len(seq)))
            else:
                seq = seq[:max_length]
            padded.append(seq)
        return padded
    
    def create_masks(self, src_tensor: torch.Tensor, tgt_tensor: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Create attention masks for source and target sequences.
        
        Args:
            src_tensor: Source tensor [batch_size, seq_len]
            tgt_tensor: Target tensor [batch_size, seq_len]
        
        Returns:
            Tuple of (src_mask, tgt_mask)
        """
        # Source mask (padding mask)
        src_mask = (src_tensor != self.src_vocab.token2idx[self.src_vocab.pad_token])
        src_mask = src_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
        
        if tgt_tensor is not None:
            # Target mask (padding + causal mask)
            tgt_padding_mask = (tgt_tensor != self.tgt_vocab.token2idx[self.tgt_vocab.pad_token])
            tgt_padding_mask = tgt_padding_mask.unsqueeze(1).unsqueeze(2)
            
            # Causal mask
            seq_len = tgt_tensor.size(1)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            
            # Combine masks
            tgt_mask = tgt_padding_mask & causal_mask
            tgt_mask = tgt_mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            
            return src_mask, tgt_mask
        else:
            return src_mask, None


class TaskSpecificProcessor:
    """
    Task-specific text processing for different NLP tasks.
    """
    
    @staticmethod
    def prepare_translation_data(src_texts: List[str], tgt_texts: List[str],
                               src_vocab: Vocabulary, tgt_vocab: Vocabulary) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data for machine translation task."""
        processor = TextProcessor(src_vocab, tgt_vocab)
        return processor.prepare_batch(src_texts, tgt_texts, 'translation')
    
    @staticmethod
    def prepare_summarization_data(texts: List[str], summaries: List[str],
                                 vocab: Vocabulary) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data for text summarization task."""
        processor = TextProcessor(vocab, vocab)
        return processor.prepare_batch(texts, summaries, 'summarization')
    
    @staticmethod
    def prepare_simplification_data(complex_texts: List[str], simple_texts: List[str],
                                  vocab: Vocabulary) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data for text simplification task."""
        processor = TextProcessor(vocab, vocab)
        return processor.prepare_batch(complex_texts, simple_texts, 'simplification')
    
    @staticmethod
    def create_inference_batch(texts: List[str], vocab: Vocabulary) -> torch.Tensor:
        """Create batch for inference (no target needed)."""
        processor = TextProcessor(vocab, vocab)
        src_tensor, _ = processor.prepare_batch(texts, task='inference')
        return src_tensor


class TextAugmentation:
    """
    Text augmentation techniques for improving model robustness.
    """
    
    @staticmethod
    def random_dropout(text: str, dropout_prob: float = 0.1) -> str:
        """Randomly drop words from text."""
        words = text.split()
        if len(words) <= 1:
            return text
        
        # Randomly drop words
        keep_mask = torch.rand(len(words)) > dropout_prob
        kept_words = [word for i, word in enumerate(words) if keep_mask[i]]
        
        return ' '.join(kept_words) if kept_words else text
    
    @staticmethod
    def random_swap(text: str, swap_prob: float = 0.1) -> str:
        """Randomly swap adjacent words."""
        words = text.split()
        if len(words) <= 1:
            return text
        
        # Randomly swap adjacent words
        for i in range(len(words) - 1):
            if torch.rand(1) < swap_prob:
                words[i], words[i + 1] = words[i + 1], words[i]
        
        return ' '.join(words)
    
    @staticmethod
    def back_translation(text: str, forward_model, backward_model,
                        src_vocab: Vocabulary, tgt_vocab: Vocabulary) -> str:
        """
        Perform back-translation for data augmentation.
        
        Args:
            text: Source text
            forward_model: Model for forward translation
            backward_model: Model for backward translation
            src_vocab: Source vocabulary
            tgt_vocab: Target vocabulary
        
        Returns:
            Augmented text
        """
        # Forward translation
        src_tensor = TaskSpecificProcessor.create_inference_batch([text], src_vocab)
        
        with torch.no_grad():
            forward_output = forward_model(src_tensor, torch.zeros(1, 1, dtype=torch.long))
            forward_tokens = torch.argmax(forward_output, dim=-1)
            forward_text = tgt_vocab.decode(forward_tokens[0].tolist())
        
        # Backward translation
        tgt_tensor = TaskSpecificProcessor.create_inference_batch([forward_text], tgt_vocab)
        
        with torch.no_grad():
            backward_output = backward_model(tgt_tensor, torch.zeros(1, 1, dtype=torch.long))
            backward_tokens = torch.argmax(backward_output, dim=-1)
            backward_text = src_vocab.decode(backward_tokens[0].tolist())
        
        return backward_text
