"""
Transformer Architecture Implementation
Based on "Attention Is All You Need" (Vaswani et al., 2017)

This module implements the complete Transformer architecture from scratch,
including Multi-Head Attention, Position-wise Feed-Forward Networks,
Positional Encoding, and the full Encoder-Decoder stack.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any


class PositionalEncoding(nn.Module):
    """
    Positional Encoding as described in the paper.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [seq_len, batch_size, d_model]
        """
        return x + self.pe[:x.size(0)]


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """
    
    def __init__(self, d_k: int, dropout: float = 0.1):
        super().__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            Q: Query tensor [batch_size, n_heads, seq_len, d_k]
            K: Key tensor [batch_size, n_heads, seq_len, d_k]
            V: Value tensor [batch_size, n_heads, seq_len, d_k]
            mask: Optional mask tensor
        Returns:
            output: Attention output
            attention_weights: Attention weights for visualization
        """
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            # Ensure mask has the right shape for broadcasting
            # scores shape: [batch_size, n_heads, seq_len, seq_len]
            # mask should be broadcastable to this shape
            if mask.dim() == 2:
                # If mask is [seq_len, seq_len], expand to [1, 1, seq_len, seq_len]
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                # If mask is [batch_size, seq_len, seq_len], expand to [batch_size, 1, seq_len, seq_len]
                mask = mask.unsqueeze(1)
            
            # Apply mask
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """
    
    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int,
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        
        # Linear projections for Q, K, V
        self.W_Q = nn.Linear(d_model, n_heads * d_k)
        self.W_K = nn.Linear(d_model, n_heads * d_k)
        self.W_V = nn.Linear(d_model, n_heads * d_v)
        self.W_O = nn.Linear(n_heads * d_v, d_model)
        
        self.attention = ScaledDotProductAttention(d_k, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            Q: Query tensor [batch_size, seq_len, d_model]
            K: Key tensor [batch_size, seq_len, d_model]
            V: Value tensor [batch_size, seq_len, d_model]
            mask: Optional mask tensor
        Returns:
            output: Multi-head attention output
            attention_weights: Attention weights for visualization
        """
        batch_size, seq_len, _ = Q.size()
        
        # Save original input for residual connection
        residual = Q
        
        # Linear projections and reshape for multi-head
        Q_proj = self.W_Q(Q)
        K_proj = self.W_K(K)
        V_proj = self.W_V(V)
        
        # Check if reshape is possible for Q
        expected_size = batch_size * seq_len * self.n_heads * self.d_k
        if Q_proj.numel() != expected_size:
            raise ValueError(f"Q size mismatch: expected {expected_size}, got {Q_proj.numel()}")
        
        # For K and V, we need to handle different sequence lengths
        K_batch_size, K_seq_len, _ = K.size()
        V_batch_size, V_seq_len, _ = V.size()
        
        # Reshape Q, K, V to multi-head format
        Q_proj = Q_proj.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K_proj = K_proj.view(K_batch_size, K_seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V_proj = V_proj.view(V_batch_size, V_seq_len, self.n_heads, self.d_v).transpose(1, 2)
        
        # Apply attention
        output, attention_weights = self.attention(Q_proj, K_proj, V_proj, mask)
        
        # Reshape and apply final linear projection
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.n_heads * self.d_v)
        output = self.W_O(output)
        
        # Residual connection and layer normalization
        output = self.layer_norm(residual + output)
        
        return output, attention_weights


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        """
        residual = x
        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = self.layer_norm(residual + x)
        return x


class EncoderLayer(nn.Module):
    """
    Single encoder layer with multi-head attention and feed-forward network.
    """
    
    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int,
                 d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, d_k, d_v, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional mask tensor
        Returns:
            output: Encoder layer output
            attention_weights: Attention weights for visualization
        """
        # Self-attention
        attn_output, attention_weights = self.attention(x, x, x, mask)
        
        # Feed-forward network
        output = self.feed_forward(attn_output)
        
        return output, attention_weights


class DecoderLayer(nn.Module):
    """
    Single decoder layer with masked multi-head attention, encoder-decoder attention,
    and feed-forward network.
    """
    
    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int,
                 d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.masked_attention = MultiHeadAttention(d_model, n_heads, d_k, d_v, dropout)
        self.encoder_attention = MultiHeadAttention(d_model, n_heads, d_k, d_v, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            encoder_output: Output from encoder
            src_mask: Source mask for encoder-decoder attention
            tgt_mask: Target mask for masked self-attention
        Returns:
            output: Decoder layer output
            masked_attn_weights: Masked attention weights
            encoder_attn_weights: Encoder-decoder attention weights
        """
        # Masked self-attention
        masked_output, masked_attn_weights = self.masked_attention(x, x, x, tgt_mask)
        
        # Encoder-decoder attention
        enc_output, encoder_attn_weights = self.encoder_attention(
            masked_output, encoder_output, encoder_output, src_mask)
        
        # Feed-forward network
        output = self.feed_forward(enc_output)
        
        return output, masked_attn_weights, encoder_attn_weights


class Transformer(nn.Module):
    """
    Complete Transformer model with encoder-decoder architecture.
    
    Based on the paper: N=6 layers, d_model=512, d_ff=2048, h=8 heads
    """
    
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 d_model: int = 512, n_layers: int = 6, n_heads: int = 8,
                 d_k: int = None, d_v: int = None, d_ff: int = 2048,
                 max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        
        # Auto-calculate d_k and d_v if not provided
        if d_k is None:
            d_k = d_model // n_heads
        if d_v is None:
            d_v = d_model // n_heads
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional encoding
        self.src_pos_encoding = PositionalEncoding(d_model, max_len)
        self.tgt_pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_k, d_v, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_k, d_v, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights as described in the paper."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Generate source mask for padding tokens."""
        src_mask = (src != 0).unsqueeze(-2)
        return src_mask
    
    def generate_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """Generate target mask for causal attention."""
        tgt_len = tgt.size(1)
        tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len)).bool()
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)
        return tgt_mask
    
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode source sequence.
        
        Args:
            src: Source tensor [batch_size, src_len]
            src_mask: Source mask
        Returns:
            Encoded source sequence
        """
        if src_mask is None:
            src_mask = self.generate_src_mask(src)
        
        # Embedding + positional encoding
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.src_pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through encoder layers
        for encoder_layer in self.encoder_layers:
            x, _ = encoder_layer(x, src_mask)
        
        return x
    
    def decode(self, tgt: torch.Tensor, encoder_output: torch.Tensor,
               src_mask: Optional[torch.Tensor] = None,
               tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode target sequence.
        
        Args:
            tgt: Target tensor [batch_size, tgt_len]
            encoder_output: Output from encoder
            src_mask: Source mask
            tgt_mask: Target mask
        Returns:
            Decoded target sequence
        """
        if src_mask is None:
            # We need to create a src_mask that matches the encoder_output sequence length
            # Since we don't have access to src here, we'll create a mask based on encoder_output
            src_len = encoder_output.size(1)
            tgt_len = tgt.size(1)
            # Create a mask that allows attention from target to source
            # Shape: [1, 1, tgt_len, src_len] - allows each target position to attend to all source positions
            src_mask = torch.ones(1, 1, tgt_len, src_len, device=encoder_output.device)
        
        if tgt_mask is None:
            tgt_mask = self.generate_tgt_mask(tgt)
        
        # Embedding + positional encoding
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.tgt_pos_encoding(x)  # Use tgt_pos_encoding for target sequence
        x = self.dropout(x)
        
        # Pass through decoder layers
        for decoder_layer in self.decoder_layers:
            x, _, _ = decoder_layer(x, encoder_output, src_mask, tgt_mask)
        
        return x
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the complete transformer.
        
        Args:
            src: Source tensor [batch_size, src_len]
            tgt: Target tensor [batch_size, tgt_len]
        Returns:
            Output logits [batch_size, tgt_len, tgt_vocab_size]
        """
        # Encode source
        encoder_output = self.encode(src)
        
        # Decode target
        decoder_output = self.decode(tgt, encoder_output)
        
        # Project to vocabulary
        output = self.output_projection(decoder_output)
        
        return output
    
    def get_attention_weights(self, src: torch.Tensor, tgt: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get attention weights for visualization.
        
        Args:
            src: Source tensor
            tgt: Target tensor
        Returns:
            Dictionary containing attention weights from all layers
        """
        attention_weights = {}
        
        # Encode source
        src_mask = self.generate_src_mask(src)
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.src_pos_encoding(x)
        x = self.dropout(x)
        
        # Collect encoder attention weights
        for i, encoder_layer in enumerate(self.encoder_layers):
            x, attn_weights = encoder_layer(x, src_mask)
            attention_weights[f'encoder_layer_{i}'] = attn_weights
        
        # Decode target
        tgt_mask = self.generate_tgt_mask(tgt)
        y = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        y = self.tgt_pos_encoding(y)
        y = self.dropout(y)
        
        # Collect decoder attention weights
        for i, decoder_layer in enumerate(self.decoder_layers):
            y, masked_attn, enc_attn = decoder_layer(y, x, src_mask, tgt_mask)
            attention_weights[f'decoder_masked_layer_{i}'] = masked_attn
            attention_weights[f'decoder_encoder_layer_{i}'] = enc_attn
        
        return attention_weights


def create_transformer_model(src_vocab_size: int, tgt_vocab_size: int,
                           d_model: int = 512, n_layers: int = 6,
                           n_heads: int = 8, d_ff: int = 2048,
                           dropout: float = 0.1) -> Transformer:
    """
    Factory function to create a transformer model with specified parameters.
    
    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        d_model: Model dimension
        n_layers: Number of layers
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout rate
    
    Returns:
        Configured Transformer model
    """
    d_k = d_v = d_model // n_heads
    
    return Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_k=d_k,
        d_v=d_v,
        d_ff=d_ff,
        dropout=dropout
    )
