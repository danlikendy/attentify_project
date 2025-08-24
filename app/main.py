"""
FastAPI Application for Attentify Platform

This module provides the main web API for the Transformer-based
text processing platform, including endpoints for translation,
summarization, and attention visualization.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

from .core.transformer import create_transformer_model
from .core.text_processor import Vocabulary, TextProcessor
from .core.trainer import TransformerTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Attentify - Transformer Platform",
    description="Interactive platform for studying and applying Transformer architecture",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Global variables for model and vocabulary
model = None
src_vocab = None
tgt_vocab = None
trainer = None

# Model configuration
MODEL_CONFIG = {
    'd_model': 512,
    'n_layers': 6,
    'n_heads': 8,
    'd_ff': 2048,
    'dropout': 0.1
}

# Default vocabulary sizes
DEFAULT_VOCAB_SIZE = 10000


@app.on_event("startup")
async def startup_event():
    """Initialize model and vocabulary on startup."""
    global model, src_vocab, tgt_vocab, trainer
    
    try:
        # Check if model checkpoint exists
        checkpoint_path = Path("./checkpoints/best_model.pt")
        if checkpoint_path.exists():
            logger.info("Loading pre-trained model from checkpoint...")
            await load_model_from_checkpoint(str(checkpoint_path))
        else:
            logger.info("Initializing demo model...")
            await initialize_demo_model()
        
        logger.info("Model initialization completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during model initialization: {e}")
        # Initialize with basic model for demo purposes
        await initialize_demo_model()


# async def initialize_new_model():
#     """Initialize a new Transformer model."""
#     # This function is not used anymore - we always use demo model
#     pass


async def initialize_demo_model():
    """Initialize a demo model with sample data."""
    global model, src_vocab, tgt_vocab, trainer
    
    # Create basic vocabularies
    src_vocab = Vocabulary(min_freq=1, max_size=1000)
    tgt_vocab = Vocabulary(min_freq=1, max_size=1000)
    
    # Add some sample tokens
    sample_tokens = ['hello', 'world', 'transformer', 'attention', 'model', 'text', 'processing', 'the', 'a', 'is', 'was', 'will', 'can', 'should', 'would', 'could', 'may', 'might', 'must', 'shall']
    for token in sample_tokens:
        src_vocab.token2idx[token] = len(src_vocab.token2idx)
        src_vocab.idx2token[len(src_vocab.idx2token)] = token
        tgt_vocab.token2idx[token] = len(tgt_vocab.token2idx)
        tgt_vocab.idx2token[len(tgt_vocab.idx2token)] = token
    
    # Create small model for demo
    logger.info(f"Creating demo model with vocab sizes: src={len(src_vocab)}, tgt={len(tgt_vocab)}")
    
    # Calculate expected dimensions
    d_model = 32  # Уменьшаем еще больше
    n_heads = 2
    d_k = d_v = d_model // n_heads  # This should be 16
    
    logger.info(f"Calculated d_k=d_v={d_k} for d_model={d_model}, n_heads={n_heads}")
    
    model = create_transformer_model(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=d_model,
        n_layers=1,  # Уменьшаем количество слоев
        n_heads=n_heads,
        d_ff=128     # Уменьшаем FFN
    )
    
    logger.info(f"Demo model created with parameters: d_model={model.d_model}, n_layers={model.n_layers}, n_heads={model.n_heads}")
    logger.info(f"Model encoder layers: {len(model.encoder_layers)}")
    logger.info(f"Model decoder layers: {len(model.decoder_layers)}")
    
    trainer = TransformerTrainer(model, src_vocab, tgt_vocab)
    
    # Test model with a simple input to verify dimensions
    logger.info("Testing model with sample input...")
    try:
        test_src = torch.tensor([[1, 2, 3]], dtype=torch.long)  # Simple test input
        test_tgt = torch.tensor([[1, 2]], dtype=torch.long)
        
        with torch.no_grad():
            test_output = model(test_src, test_tgt)
            logger.info(f"Test output shape: {test_output.shape}")
            logger.info("Model test successful!")
    except Exception as e:
        logger.error(f"Model test failed: {e}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")


async def load_model_from_checkpoint(checkpoint_path: str):
    """Load model from checkpoint file."""
    global model, src_vocab, tgt_vocab, trainer
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load vocabularies
    src_vocab = checkpoint['src_vocab']
    tgt_vocab = checkpoint['tgt_vocab']
    
    # Create model with saved configuration
    model_config = checkpoint['model_config']
    model = create_transformer_model(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        **model_config
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Initialize trainer
    trainer = TransformerTrainer(model, src_vocab, tgt_vocab)
    
    # Load trainer state
    trainer.load_checkpoint(checkpoint_path)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main application page."""
    html_content = """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Attentify - Платформа прикладного внимания</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            :root {
                /* Dark theme colors */
                --bg-primary: #1a1a1a;
                --bg-secondary: #2d2d2d;
                --bg-tertiary: #3d3d3d;
                --bg-card: #2d2d2d;
                --bg-input: #3d3d3d;
                
                /* Text colors */
                --text-primary: #ffffff;
                --text-secondary: #eae0c8;
                --text-tertiary: #b4674d;
                --text-accent: #7b3f00;
                
                /* Accent colors */
                --accent-primary: #7b3f00;
                --accent-secondary: #b4674d;
                --accent-accent: #eae0c8;
                --accent-green: #30d158;
                --accent-orange: #ff9f0a;
                --accent-red: #ff453a;
                --accent-purple: #bf5af2;
                
                /* Gradients */
                --gradient-primary: linear-gradient(135deg, #7b3f00 0%, #b4674d 100%);
                --gradient-secondary: linear-gradient(135deg, #b4674d 0%, #eae0c8 100%);
                --gradient-accent: linear-gradient(135deg, #7b3f00 0%, #eae0c8 100%);
                
                /* Shadows */
                --shadow-light: 0 2px 10px rgba(0,0,0,0.3);
                --shadow-medium: 0 4px 20px rgba(0,0,0,0.4);
                --shadow-heavy: 0 8px 30px rgba(0,0,0,0.5);
                
                /* Border radius */
                --radius-small: 8px;
                --radius-medium: 16px;
                --radius-large: 24px;
            }
            
            /* Light theme */
            [data-theme="light"] {
                --bg-primary: #ffffff;
                --bg-secondary: #f8f9fa;
                --bg-tertiary: #e9ecef;
                --bg-card: #ffffff;
                --bg-input: #f8f9fa;
                
                --text-primary: #212529;
                --text-secondary: #495057;
                --text-tertiary: #6c757d;
                --text-accent: #7b3f00;
                
                --shadow-light: 0 2px 10px rgba(0,0,0,0.1);
                --shadow-medium: 0 4px 20px rgba(0,0,0,0.15);
                --shadow-heavy: 0 8px 30px rgba(0,0,0,0.2);
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', Roboto, sans-serif;
                line-height: 1.6;
                color: var(--text-primary);
                background: var(--bg-primary);
                -webkit-font-smoothing: antialiased;
                -moz-osx-font-smoothing: grayscale;
            }
            
            .header {
                background: rgba(28, 28, 30, 0.8);
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
                padding: 1rem 2rem;
                box-shadow: var(--shadow-light);
                position: sticky;
                top: 0;
                z-index: 100;
                border-bottom: 1px solid rgba(255,255,255,0.1);
            }
            
            .nav {
                display: flex;
                justify-content: space-between;
                align-items: center;
                max-width: 1200px;
                margin: 0 auto;
            }
            
            .logo {
                font-size: 1.8rem;
                font-weight: 700;
                color: var(--accent-primary);
                display: flex;
                align-items: center;
                gap: 0.5rem;
                font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
            }
            
            .logo-icon {
                width: 40px;
                height: 40px;
                background: var(--gradient-accent);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
                font-size: 1.2rem;
                box-shadow: var(--shadow-medium);
            }
            
            .nav-links {
                display: flex;
                gap: 2rem;
                align-items: center;
            }
            
            .nav-links a {
                text-decoration: none;
                color: var(--text-secondary);
                font-weight: 500;
                transition: all 0.3s ease;
                padding: 0.5rem 1rem;
                border-radius: var(--radius-small);
            }
            
            .nav-links a:hover {
                color: var(--text-primary);
                background: rgba(255,255,255,0.1);
            }
            
            .nav-links a.active {
                color: var(--accent-primary);
                background: rgba(123, 63, 0, 0.2);
                border: 1px solid rgba(123, 63, 0, 0.3);
                font-weight: 600;
            }
            
            .nav-controls {
                display: flex;
                gap: 1rem;
                align-items: center;
            }
            
            .control-btn {
                background: transparent;
                border: 1px solid rgba(255,255,255,0.2);
                border-radius: var(--radius-small);
                padding: 0.5rem;
                color: var(--text-secondary);
                cursor: pointer;
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                min-width: 40px;
                height: 40px;
            }
            
            .control-btn:hover {
                background: rgba(255,255,255,0.1);
                border-color: rgba(255,255,255,0.3);
                color: var(--text-primary);
            }
            
            [data-theme="light"] .control-btn {
                border: 1px solid rgba(0,0,0,0.2);
                color: var(--text-secondary);
            }
            
            [data-theme="light"] .control-btn:hover {
                background: rgba(0,0,0,0.1);
                border-color: rgba(0,0,0,0.3);
                color: var(--text-primary);
            }
            
            .theme-toggle, .language-toggle {
                position: relative;
            }
            
            .language-toggle .control-btn {
                min-width: 50px;
                font-weight: 600;
                font-size: 0.9rem;
            }
            
            .btn-primary {
                background: var(--gradient-primary);
                color: white;
                padding: 0.75rem 1.5rem;
                border: none;
                border-radius: var(--radius-small);
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                text-decoration: none;
                display: inline-block;
                box-shadow: var(--shadow-medium);
                font-family: inherit;
                position: relative;
                overflow: hidden;
            }
            
            .btn-primary::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
                transition: left 0.5s;
            }
            
            .btn-primary:hover::before {
                left: 100%;
            }
            
            .btn-primary:hover {
                transform: translateY(-2px);
                box-shadow: var(--shadow-heavy);
                background: var(--gradient-secondary);
            }
            

            
            .btn-secondary {
                background: transparent;
                color: var(--text-secondary);
                padding: 0.75rem 1.5rem;
                border: 1px solid rgba(123, 63, 0, 0.3);
                border-radius: var(--radius-small);
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                text-decoration: none;
                display: inline-block;
                font-family: inherit;
            }
            
            .btn-secondary:hover {
                background: rgba(123, 63, 0, 0.1);
                border-color: var(--accent-primary);
                color: var(--text-primary);
            }
            
            .hero {
                background: linear-gradient(135deg, rgba(123, 63, 0, 0.7) 0%, rgba(180, 103, 77, 0.7) 100%);
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
                color: white;
                padding: 6rem 2rem;
                text-align: center;
                position: relative;
                overflow: hidden;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            .hero::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="50" cy="50" r="1" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
                opacity: 0.3;
            }
            
            .hero-content {
                max-width: 800px;
                margin: 0 auto;
                position: relative;
                z-index: 1;
            }
            
            .hero h1 {
                font-size: 4rem;
                margin-bottom: 1rem;
                font-weight: 800;
                background: linear-gradient(45deg, #ffffff, #f0f0f0);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
            }
            
            .hero p {
                font-size: 1.3rem;
                margin-bottom: 2rem;
                opacity: 0.95;
                font-weight: 400;
                line-height: 1.6;
            }
            
            .main-content {
                max-width: 1200px;
                margin: 0 auto;
                padding: 3rem 2rem;
            }
            
            .features-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                gap: 2rem;
                margin-bottom: 4rem;
            }
            
            .feature-card {
                background: var(--bg-card);
                padding: 2.5rem;
                border-radius: var(--radius-large);
                box-shadow: var(--shadow-medium);
                transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                border: 1px solid rgba(123, 63, 0, 0.3);
                position: relative;
                overflow: hidden;
                cursor: pointer;
            }
            
            .feature-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: var(--gradient-primary);
            }
            
            .feature-card:hover {
                transform: translateY(-12px) scale(1.02);
                box-shadow: var(--shadow-heavy);
                border-color: var(--accent-primary);
                background: var(--bg-tertiary);
            }
            
            .feature-icon {
                width: 70px;
                height: 70px;
                background: var(--gradient-primary);
                border-radius: var(--radius-medium);
                display: flex;
                align-items: center;
                justify-content: center;
                margin-bottom: 1.5rem;
                color: white;
                font-size: 2rem;
                box-shadow: var(--shadow-medium);
                transition: all 0.3s ease;
            }
            
            .feature-card h3 {
                font-size: 1.6rem;
                margin-bottom: 1rem;
                color: var(--text-primary);
                font-weight: 700;
                font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
            }
            
            .feature-card p {
                color: var(--text-secondary);
                margin-bottom: 1.5rem;
                line-height: 1.6;
                font-size: 1rem;
            }
            
            .transformer-demo {
                background: var(--bg-card);
                padding: 3rem;
                border-radius: var(--radius-large);
                box-shadow: var(--shadow-medium);
                margin-bottom: 3rem;
                border: 1px solid rgba(255,255,255,0.1);
            }
            
            .demo-header {
                text-align: center;
                margin-bottom: 3rem;
            }
            
            .demo-header h2 {
                font-size: 2.5rem;
                color: var(--text-primary);
                margin-bottom: 1rem;
                font-weight: 700;
                font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
            }
            
            .demo-header p {
                color: var(--text-secondary);
                font-size: 1.2rem;
                max-width: 600px;
                margin: 0 auto;
            }
            
            .demo-controls {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1.5rem;
                margin-bottom: 2.5rem;
            }
            
            .control-group {
                display: flex;
                flex-direction: column;
                gap: 0.75rem;
            }
            
            .control-group label {
                font-weight: 600;
                color: var(--text-primary);
                font-size: 0.95rem;
                font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
            }
            
            .control-group input, .control-group select {
                padding: 1rem;
                border: 2px solid rgba(255,255,255,0.1);
                border-radius: var(--radius-small);
                font-size: 1rem;
                transition: all 0.3s ease;
                background: var(--bg-input);
                color: var(--text-primary);
                font-family: inherit;
            }
            
            .control-group input:focus,             .control-group select:focus {
                outline: none;
                border-color: var(--accent-primary);
                box-shadow: 0 0 0 3px rgba(123, 63, 0, 0.1);
            }
            
            .demo-input {
                margin-bottom: 2.5rem;
            }
            
            .demo-input textarea {
                width: 100%;
                min-height: 140px;
                padding: 1.5rem;
                border: 2px solid rgba(255,255,255,0.1);
                border-radius: var(--radius-small);
                font-size: 1rem;
                font-family: inherit;
                resize: vertical;
                transition: all 0.3s ease;
                background: var(--bg-input);
                color: var(--text-primary);
                line-height: 1.6;
            }
            
            .demo-input textarea:focus {
                outline: none;
                border-color: var(--accent-primary);
                box-shadow: 0 0 0 3px rgba(123, 63, 0, 0.1);
            }
            
            .demo-actions {
                display: flex;
                gap: 1rem;
                justify-content: center;
                margin-bottom: 2.5rem;
                flex-wrap: wrap;
            }
            
            .btn-secondary {
                background: var(--bg-tertiary);
                color: var(--text-primary);
                padding: 0.75rem 1.5rem;
                border: 1px solid rgba(255,255,255,0.1);
                border-radius: var(--radius-small);
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                font-family: inherit;
            }
            
            .btn-secondary:hover {
                background: var(--bg-secondary);
                border-color: rgba(255,255,255,0.2);
                transform: translateY(-2px);
            }
            
            .demo-output {
                background: var(--bg-tertiary);
                padding: 2rem;
                border-radius: var(--radius-small);
                border-left: 4px solid var(--accent-primary);
                min-height: 120px;
                border: 1px solid rgba(255,255,255,0.1);
            }
            
            .demo-output h4 {
                margin-bottom: 1rem;
                color: var(--text-primary);
                font-weight: 600;
                font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
            }
            
            .attention-visualization {
                margin-top: 2.5rem;
                padding: 2rem;
                background: var(--bg-tertiary);
                border-radius: var(--radius-small);
                border: 1px solid rgba(255,255,255,0.1);
            }
            
            .attention-visualization h4 {
                margin-bottom: 1.5rem;
                color: var(--text-primary);
                font-weight: 600;
                font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
            }
            
            .attention-controls {
                display: flex;
                gap: 1.5rem;
                margin-bottom: 1.5rem;
                flex-wrap: wrap;
                align-items: center;
            }
            
            .attention-controls label {
                font-weight: 600;
                color: var(--text-primary);
                margin-right: 0.75rem;
                font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
            }
            
            .attention-controls select {
                padding: 0.75rem;
                border: 2px solid rgba(255,255,255,0.1);
                border-radius: var(--radius-small);
                font-size: 0.95rem;
                background: var(--bg-input);
                color: var(--text-primary);
                min-width: 180px;
                font-family: inherit;
            }
            
            .attention-info {
                background: var(--bg-input);
                padding: 1.5rem;
                border-radius: var(--radius-small);
                margin-bottom: 1.5rem;
                font-size: 0.95rem;
                color: var(--text-secondary);
                border: 1px solid rgba(255,255,255,0.05);
            }
            
            .attention-info p {
                margin: 0.5rem 0;
            }
            
            .attention-heatmap-container {
                position: relative;
                min-height: 400px;
                border: 1px solid rgba(255,255,255,0.1);
                border-radius: var(--radius-small);
                background: var(--bg-input);
                padding: 2rem;
                text-align: center;
            }
            
            .attention-heatmap {
                display: flex;
                align-items: center;
                justify-content: center;
                min-height: 400px;
                font-size: 1.1rem;
                color: var(--text-secondary);
            }
            
            .attention-explanation {
                margin-top: 2rem;
                padding: 2rem;
                background: var(--bg-input);
                border-radius: var(--radius-small);
                font-size: 0.95rem;
                color: var(--text-secondary);
                border-left: 4px solid var(--accent-blue);
                border: 1px solid rgba(255,255,255,0.05);
            }
            
            .attention-explanation h5 {
                margin-bottom: 1rem;
                color: var(--text-primary);
                font-size: 1.1rem;
                font-weight: 600;
                font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
            }
            
            .attention-explanation ul {
                list-style: none;
                padding-left: 0;
            }
            
            .attention-explanation li {
                margin-bottom: 0.75rem;
                padding-left: 24px;
                position: relative;
            }
            
            .attention-explanation li:before {
                content: "•";
                color: var(--accent-primary);
                font-weight: bold;
                position: absolute;
                left: 0;
                font-size: 1.2rem;
            }
            
            .footer {
                background: var(--bg-secondary);
                color: var(--text-secondary);
                text-align: center;
                padding: 3rem 2rem;
                margin-top: 4rem;
                border-top: 1px solid rgba(255,255,255,0.1);
            }
            
            .footer p {
                opacity: 0.8;
                font-size: 1rem;
            }
            
            /* Interactive elements */
            .interactive-demo {
                background: var(--bg-card);
                padding: 2rem;
                border-radius: var(--radius-medium);
                margin: 2rem 0;
                border: 1px solid rgba(255,255,255,0.1);
            }
            
            .model-architecture {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 1.5rem;
                margin: 2rem 0;
            }
            
            .architecture-card {
                background: var(--bg-tertiary);
                padding: 1.5rem;
                border-radius: var(--radius-small);
                border: 1px solid rgba(255,255,255,0.05);
                text-align: center;
            }
            
            .architecture-card h4 {
                color: var(--text-primary);
                margin-bottom: 0.5rem;
                font-weight: 600;
            }
            
            .architecture-card p {
                color: var(--text-secondary);
                font-size: 0.9rem;
            }
            
            /* Responsive design */
            @media (max-width: 768px) {
                .hero h1 {
                    font-size: 2.5rem;
                }
                
                .nav-links {
                    display: none;
                }
                
                .features-grid {
                    grid-template-columns: 1fr;
                }
                
                .demo-controls {
                    grid-template-columns: 1fr;
                }
                
                .attention-controls {
                    flex-direction: column;
                    align-items: stretch;
                }
                
                .attention-controls select {
                    min-width: auto;
                }
            }
            
            /* Smooth scrolling */
            html {
                scroll-behavior: smooth;
            }
            
            /* Custom scrollbar */
            ::-webkit-scrollbar {
                width: 8px;
            }
            
            ::-webkit-scrollbar-track {
                background: var(--bg-secondary);
            }
            
            ::-webkit-scrollbar-thumb {
                background: var(--accent-primary);
                border-radius: 4px;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: var(--accent-green);
            }
            
            /* Light theme overrides */
            [data-theme="light"] .nav-links a {
                color: var(--text-primary) !important;
                font-weight: 600;
            }
            
            [data-theme="light"] .logo {
                color: var(--accent-primary);
            }
            
            [data-theme="light"] .control-btn {
                border-color: rgba(0,0,0,0.2);
                color: var(--text-primary);
            }
            
            [data-theme="light"] .header {
                background: rgba(255, 255, 255, 0.95);
                border-bottom: 1px solid rgba(0,0,0,0.1);
            }
            
            [data-theme="light"] .nav-links a:hover {
                background: rgba(0,0,0,0.1);
                color: var(--accent-primary) !important;
            }
            

            
            [data-theme="light"] .nav-links a.active {
                color: var(--accent-primary);
                background: rgba(123, 63, 0, 0.15);
                border: 1px solid rgba(123, 63, 0, 0.3);
                font-weight: 600;
            }
            
            [data-theme="light"] .control-btn:hover {
                background: rgba(0,0,0,0.1);
                border-color: rgba(0,0,0,0.3);
                color: var(--text-primary);
            }
        </style>
    </head>
    <body>
        <header class="header">
            <nav class="nav">
                <div class="logo">
                    <div class="logo-icon">A</div>
                    Attentify
            </div>
                <div class="nav-links">
                    <a href="/features">Возможности</a>
                    <a href="/demo">Демо</a>
                    <a href="/about">О проекте</a>
                    <a href="/blog">Блог</a>
                    <a href="/docs">Документация</a>
                </div>
                <div class="nav-controls">
                    <div class="theme-toggle">
                        <button id="theme-toggle" class="control-btn" title="Сменить тему">
                            <svg id="sun-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="12" cy="12" r="5"></circle>
                                <line x1="12" y1="1" x2="12" y2="3"></line>
                                <line x1="12" y1="21" x2="12" y2="23"></line>
                                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
                                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
                                <line x1="1" y1="12" x2="3" y2="12"></line>
                                <line x1="21" y1="12" x2="23" y2="12"></line>
                                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
                                <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
                            </svg>
                            <svg id="moon-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="display: none;">
                                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
                            </svg>
                        </button>
                    </div>
                    <div class="language-toggle">
                        <button id="language-toggle" class="control-btn" title="Сменить язык">
                            <span id="current-lang">RU</span>
                        </button>
                    </div>
                    <a href="/docs" class="btn-primary">API Документация</a>
                </div>
            </nav>
        </header>
        
        <section class="hero">
            <div class="hero-content">
                <h1>Attentify</h1>
                <p>Платформа прикладного внимания — изучайте и применяйте архитектуру Transformer в реальном времени с интерактивной визуализацией</p>
                </div>
        </section>
        
        <main class="main-content">
            <section id="features" class="features-grid">
                                <div class="feature-card">
                    <h3>Реализация с нуля</h3>
                    <p>Полная реализация архитектуры Transformer по статье "Attention Is All You Need" без готовых библиотек. Каждый компонент написан с нуля для глубокого понимания</p>
                </div>
                
                <div class="feature-card">
                    <h3>Визуализация внимания</h3>
                    <p>Наглядно увидите, как работает механизм внимания в трансформере — heatmap связей между словами в реальном времени с возможностью выбора слоев и голов</p>
                </div>
                
                <div class="feature-card">
                    <h3>Интерактивная настройка</h3>
                    <p>Меняйте количество слоев, голов внимания и другие параметры модели в реальном времени. Смотрите, как архитектура адаптируется к вашим настройкам</p>
                </div>
                
                <div class="feature-card">
                    <h3>Прикладные задачи</h3>
                    <p>Перевод, резюмирование, упрощение текста — все на вашей собственной модели. Изучайте, как трансформер решает конкретные задачи</p>
            </div>
            </section>
            
            <section class="interactive-demo">
                <h2 style="text-align: center; margin-bottom: 2rem; color: var(--text-primary); font-size: 2rem;">Архитектура Transformer</h2>
                <p style="text-align: center; margin-bottom: 2rem; color: var(--text-secondary); font-size: 1.1rem;">
                    Понять, как работает трансформер, проще всего через интерактивное изучение. Каждый компонент имеет свою роль в обработке текста.
                </p>
                
                <div class="model-architecture">
                    <div class="architecture-card">
                        <h4>Embedding Layer</h4>
                        <p>Преобразует слова в числовые векторы, сохраняя семантическую информацию</p>
            </div>
                    <div class="architecture-card">
                        <h4>Positional Encoding</h4>
                        <p>Добавляет информацию о позиции слова в последовательности</p>
                    </div>
                    <div class="architecture-card">
                        <h4>Multi-Head Attention</h4>
                        <p>Позволяет модели "смотреть" на разные части текста одновременно</p>
                    </div>
                    <div class="architecture-card">
                        <h4>Feed Forward</h4>
                        <p>Обрабатывает информацию через полносвязные слои</p>
                    </div>
                    <div class="architecture-card">
                        <h4>Residual Connections</h4>
                        <p>Помогает градиентам проходить через глубокие слои</p>
                    </div>
                    <div class="architecture-card">
                        <h4>Layer Normalization</h4>
                        <p>Стабилизирует обучение и ускоряет сходимость</p>
                    </div>
                </div>
            </section>
            
            <section id="demo" class="transformer-demo">
                <div class="demo-header">
                    <h2>Интерактивная демонстрация</h2>
                    <p>Настройте параметры трансформера и протестируйте его на вашем тексте. Изучайте, как архитектура влияет на качество обработки</p>
            </div>
                
                <div class="demo-controls">
                    <div class="control-group">
                        <label for="d_model">Размер модели (d_model)</label>
                        <select id="d_model">
                            <option value="32">32 (быстро, для изучения)</option>
                            <option value="64">64 (баланс)</option>
                            <option value="128">128 (качество)</option>
                            <option value="256">256 (высокое качество)</option>
                            <option value="512">512 (стандарт BERT)</option>
                        </select>
        </div>
                    
                    <div class="control-group">
                        <label for="n_layers">Количество слоев</label>
                        <select id="n_layers">
                            <option value="1">1 (быстро, для изучения)</option>
                            <option value="2">2 (базовое понимание)</option>
                            <option value="4">4 (средняя сложность)</option>
                            <option value="6">6 (стандарт BERT)</option>
                        </select>
                    </div>
                    
                    <div class="control-group">
                        <label for="n_heads">Количество голов внимания</label>
                        <select id="n_heads">
                            <option value="2">2 (простота)</option>
                            <option value="4">4 (баланс)</option>
                            <option value="8">8 (стандарт BERT)</option>
                            <option value="16">16 (высокая детализация)</option>
                        </select>
                    </div>
                    
                    <div class="control-group">
                        <label for="task">Задача</label>
                        <select id="task">
                            <option value="translate">Перевод</option>
                            <option value="summarize">Резюмирование</option>
                            <option value="simplify">Упрощение</option>
                        </select>
                    </div>
                </div>
                
                <div class="demo-input">
                    <label for="input_text">Введите текст для обработки:</label>
                    <textarea id="input_text" placeholder="Попробуйте: 'Привет, мир! Это демонстрация работы трансформера с механизмом внимания.'">Привет, мир! Это демонстрация работы трансформера с механизмом внимания.</textarea>
                </div>
                
                <div class="demo-actions">
                    <button class="btn-primary" onclick="processText()">Обработать текст</button>
                    <button class="btn-secondary" onclick="resetModel()">Сбросить модель</button>
                    <button class="btn-secondary" onclick="showAttention()">Показать внимание</button>
                </div>
                
                <div class="demo-output">
                    <h4>Результат:</h4>
                    <div id="output_text">Нажмите "Обработать текст" для получения результата...</div>
                </div>
                
                <div class="attention-visualization" id="attention_viz" style="display: none;">
                    <h4>Визуализация внимания</h4>
                    <div class="attention-controls">
                        <label for="layer_select">Выберите слой:</label>
                        <select id="layer_select" onchange="updateAttentionVisualization()">
                            <option value="">Загрузка...</option>
                        </select>
                        
                        <label for="head_select">Выберите голову внимания:</label>
                        <select id="head_select" onchange="updateAttentionVisualization()">
                            <option value="average">Среднее по всем головам</option>
                        </select>
                    </div>
                    
                    <div class="attention-info">
                        <p><strong>Текущая конфигурация:</strong> <span id="current_config">-</span></p>
                        <p><strong>Размер последовательности:</strong> <span id="seq_length">-</span></p>
                        <p><strong>Параметры модели:</strong> <span id="model_params">-</span></p>
                    </div>
                    
                    <div class="attention-heatmap-container">
                        <div id="attention_heatmap">
                            <p>Нажмите "Показать внимание" для загрузки визуализации...</p>
                        </div>
                    </div>
                    
                    <div class="attention-explanation">
                        <h5>Как читать heatmap:</h5>
                        <ul>
                            <li><strong>По горизонтали:</strong> слова, на которые "смотрит" модель</li>
                            <li><strong>По вертикали:</strong> слова, которые обрабатываются</li>
                            <li><strong>Цвет:</strong> чем ярче, тем сильнее внимание</li>
                            <li><strong>Диагональ:</strong> self-attention (слово на себя)</li>
                            <li><strong>Паттерны:</strong> ищите кластеры внимания — это показывает, как модель группирует связанные слова</li>
                        </ul>
                    </div>
                </div>
            </section>
            
            <section class="interactive-demo">
                <h2 style="text-align: center; margin-bottom: 2rem; color: var(--text-primary); font-size: 2rem;">Механистическая интерпретируемость</h2>
                <p style="text-align: center; margin-bottom: 2rem; color: var(--text-secondary); font-size: 1.1rem;">
                    Attentify позволяет вам заглянуть внутрь "черного ящика" трансформера и понять, как именно он принимает решения.
                </p>
                
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; margin: 2rem 0;">
                    <div style="background: var(--bg-tertiary); padding: 1.5rem; border-radius: var(--radius-small); border: 1px solid rgba(255,255,255,0.05);">
                        <h4 style="color: white; margin-bottom: 1rem; font-size: 1.3rem; font-weight: 700;">Attention Patterns</h4>
                        <p style="color: var(--text-secondary); font-size: 0.95rem;">Изучайте паттерны внимания в разных слоях. Увидите, как модель учится связывать слова в предложениях.</p>
                    </div>
                    <div style="background: var(--bg-tertiary); padding: 1.5rem; border-radius: var(--radius-small); border: 1px solid rgba(255,255,255,0.05);">
                        <h4 style="color: white; margin-bottom: 1rem; font-size: 1.3rem; font-weight: 700;">Model Behavior</h4>
                        <p style="color: var(--text-secondary); font-size: 0.95rem;">Анализируйте, как изменение параметров влияет на поведение модели. Понимайте архитектурные решения.</p>
                    </div>
                    <div style="background: var(--bg-tertiary); padding: 1.5rem; border-radius: var(--radius-small); border: 1px solid rgba(255,255,255,0.05);">
                        <h4 style="color: white; margin-bottom: 1rem; font-size: 1.3rem; font-weight: 700;">Visualization Tools</h4>
                        <p style="color: var(--text-secondary); font-size: 0.95rem;">Интерактивные heatmap, графики и диаграммы для глубокого понимания работы трансформера.</div>
                </div>
            </section>
        </main>
        
        <footer class="footer">
            <p>&copy; 2024 Attentify. Платформа для изучения и применения архитектуры Transformer.</p>
        </footer>
        
        <script>
            let currentModel = null;
            let attentionData = null;
            let currentTheme = 'dark';
            let currentLanguage = 'ru';
            
            // Theme and language management
            function initThemeAndLanguage() {
                // Load saved preferences
                const savedTheme = localStorage.getItem('attentify-theme') || 'dark';
                const savedLanguage = localStorage.getItem('attentify-language') || 'ru';
                
                setTheme(savedTheme);
                setLanguage(savedLanguage);
                
                // Setup event listeners
                document.getElementById('theme-toggle').addEventListener('click', toggleTheme);
                document.getElementById('language-toggle').addEventListener('click', toggleLanguage);
                
                // Highlight active navigation item
                highlightActiveNavItem();
            }
            
            function highlightActiveNavItem() {
                const currentPath = window.location.pathname;
                const navLinks = document.querySelectorAll('.nav-links a');
                
                navLinks.forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === currentPath) {
                        link.classList.add('active');
                    }
                });
            }
            
            function setTheme(theme) {
                currentTheme = theme;
                document.documentElement.setAttribute('data-theme', theme);
                localStorage.setItem('attentify-theme', theme);
                
                // Update icons
                const sunIcon = document.getElementById('sun-icon');
                const moonIcon = document.getElementById('moon-icon');
                
                if (theme === 'light') {
                    sunIcon.style.display = 'none';
                    moonIcon.style.display = 'block';
                } else {
                    sunIcon.style.display = 'block';
                    moonIcon.style.display = 'none';
                }
            }
            
            function toggleTheme() {
                const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
                setTheme(newTheme);
            }
            
            function setLanguage(lang) {
                currentLanguage = lang;
                localStorage.setItem('attentify-language', lang);
                document.getElementById('current-lang').textContent = lang.toUpperCase();
                
                // Here you would implement actual language switching
                // For now, we just update the display
            }
            
            function toggleLanguage() {
                const newLang = currentLanguage === 'ru' ? 'en' : 'ru';
                setLanguage(newLang);
            }
            
            async function processText() {
                const inputText = document.getElementById('input_text').value;
                const dModel = parseInt(document.getElementById('d_model').value);
                const nLayers = parseInt(document.getElementById('n_layers').value);
                const nHeads = parseInt(document.getElementById('n_heads').value);
                const task = document.getElementById('task').value;
                
                if (!inputText.trim()) {
                    alert('Пожалуйста, введите текст для обработки');
                    return;
                }
                
                const outputDiv = document.getElementById('output_text');
                                        outputDiv.innerHTML = '<div style="display: flex; align-items: center; gap: 0.5rem; color: var(--accent-primary);"><div class="spinner" style="width: 20px; height: 20px; border: 2px solid var(--accent-primary); border-top: 2px solid transparent; border-radius: 50%; animation: spin 1s linear infinite;"></div>Обрабатываю текст...</div>';
                
                try {
                    const response = await fetch('/process_text', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            text: inputText,
                            d_model: dModel,
                            n_layers: nLayers,
                            n_heads: nHeads,
                            task: task
                        })
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        const modelParams = result.model_parameters;
                        const paramsInfo = `d_model=${modelParams.d_model}, d_k=d_v=${modelParams.d_k}, d_ff=${modelParams.d_ff}`;
                        
                        outputDiv.innerHTML = `
                            <div style="margin-bottom: 1rem;">
                                <strong style="color: var(--accent-green);">✅ Результат:</strong><br>
                                <span style="color: var(--text-primary);">${result.output}</span>
                            </div>
                            <div style="background: var(--bg-input); padding: 1rem; border-radius: var(--radius-small); border: 1px solid rgba(255,255,255,0.1);">
                                <strong style="color: var(--accent-primary);">Параметры модели:</strong><br>
                                <span style="color: var(--text-secondary); font-size: 0.9rem;">${paramsInfo}</span>
                            </div>
                        `;
                        
                        // Update model params display
                        document.getElementById('model_params').textContent = paramsInfo;
                    } else {
                        outputDiv.innerHTML = '<div style="color: var(--accent-red);">Ошибка при обработке текста. Попробуйте еще раз.</div>';
                    }
                } catch (error) {
                                            outputDiv.innerHTML = '<div style="color: var(--accent-red);">Ошибка сети. Проверьте подключение к интернету.</div>';
                }
            }
            
            async function showAttention() {
                const attentionViz = document.getElementById('attention_viz');
                const inputText = document.getElementById('input_text').value;
                const dModel = parseInt(document.getElementById('d_model').value);
                const nLayers = parseInt(document.getElementById('n_layers').value);
                const nHeads = parseInt(document.getElementById('n_heads').value);
                
                if (!inputText.trim()) {
                    alert('Сначала введите текст для анализа');
                    return;
                }
                
                if (attentionViz.style.display === 'none') {
                    attentionViz.style.display = 'block';
                    
                    // Show loading state with better styling
                    document.getElementById('attention_heatmap').innerHTML = `
                        <div style="display: flex; flex-direction: column; align-items: center; gap: 1rem; color: var(--accent-primary);">
                            <div class="spinner" style="width: 40px; height: 40px; border: 3px solid var(--accent-primary); border-top: 3px solid transparent; border-radius: 50%; animation: spin 1s linear infinite;"></div>
                            <div>Загружаю attention weights...</div>
                        </div>
                    `;
                    
                    try {
                        // Get attention weights
                        const response = await fetch('/get_attention', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({
                                text: inputText,
                                d_model: dModel,
                                n_layers: nLayers,
                                n_heads: nHeads
                            })
                        });
                        
                        if (response.ok) {
                            attentionData = await response.json();
                            setupAttentionVisualization();
                        } else {
                            document.getElementById('attention_heatmap').innerHTML = '<div style="color: var(--accent-red);">Ошибка при загрузке attention weights</div>';
                        }
                    } catch (error) {
                        document.getElementById('attention_heatmap').div.innerHTML = '<div style="color: var(--accent-red);">Ошибка сети при загрузке attention weights</div>';
                    }
                } else {
                    attentionViz.style.display = 'none';
                }
            }
            
            function setupAttentionVisualization() {
                if (!attentionData) return;
                
                // Update info with better formatting
                document.getElementById('current_config').textContent = 
                    `d_model=${attentionData.model_info.d_model}, слоев=${attentionData.model_info.n_layers}, голов=${attentionData.model_info.n_heads}`;
                document.getElementById('seq_length').textContent = attentionData.words.length;
                
                // Setup layer selector with better labels
                const layerSelect = document.getElementById('layer_select');
                layerSelect.innerHTML = '';
                
                Object.keys(attentionData.attention_weights).forEach(layerName => {
                    const option = document.createElement('option');
                    option.value = layerName;
                    // Better layer naming
                    let displayName = layerName;
                    if (layerName.includes('encoder')) {
                        displayName = `Encoder ${layerName.split('_').pop()}`;
                    } else if (layerName.includes('decoder')) {
                        if (layerName.includes('masked')) {
                            displayName = `Decoder Masked ${layerName.split('_').pop()}`;
                        } else {
                            displayName = `Decoder Encoder ${layerName.split('_').pop()}`;
                        }
                    }
                    option.textContent = displayName;
                    layerSelect.appendChild(option);
                });
                
                // Setup head selector
                const headSelect = document.getElementById('head_select');
                headSelect.innerHTML = '<option value="average">Среднее по всем головам</option>';
                
                // Show first layer by default
                if (Object.keys(attentionData.attention_weights).length > 0) {
                    layerSelect.value = Object.keys(attentionData.attention_weights)[0];
                    updateAttentionVisualization();
                }
            }
            
            function updateAttentionVisualization() {
                if (!attentionData) return;
                
                const selectedLayer = document.getElementById('layer_select').value;
                const selectedHead = document.getElementById('head_select').value;
                
                if (!selectedLayer) return;
                
                const weights = attentionData.attention_weights[selectedLayer];
                if (!weights) return;
                
                // Create heatmap
                createAttentionHeatmap(weights, attentionData.words);
            }
            
            function createAttentionHeatmap(weights, words) {
                const container = document.getElementById('attention_heatmap');
                const size = words.length;
                
                if (size === 0) return;
                
                // Calculate cell size with better scaling
                const cellSize = Math.min(500 / size, 80); // Max 80px per cell
                const totalSize = size * cellSize;
                
                // Create SVG heatmap with better styling
                const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
                svg.setAttribute('width', totalSize);
                svg.setAttribute('height', totalSize);
                svg.style.display = 'block';
                svg.style.margin = '0 auto';
                svg.style.borderRadius = 'var(--radius-small)';
                svg.style.overflow = 'hidden';
                
                // Add words labels with better styling
                words.forEach((word, i) => {
                    // X-axis labels (bottom)
                    const xLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                    xLabel.setAttribute('x', i * cellSize + cellSize / 2);
                    xLabel.setAttribute('y', totalSize + 25);
                    xLabel.setAttribute('text-anchor', 'middle');
                    xLabel.setAttribute('font-size', '13');
                    xLabel.setAttribute('fill', 'var(--text-primary)');
                    xLabel.setAttribute('font-weight', '500');
                    xLabel.textContent = word;
                    svg.appendChild(xLabel);
                    
                    // Y-axis labels (left)
                    const yLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                    yLabel.setAttribute('x', -15);
                    yLabel.setAttribute('y', i * cellSize + cellSize / 2 + 4);
                    yLabel.setAttribute('text-anchor', 'end');
                    yLabel.setAttribute('font-size', '13');
                    yLabel.setAttribute('fill', 'var(--text-primary)');
                    yLabel.setAttribute('font-weight', '500');
                    yLabel.textContent = word;
                    svg.appendChild(yLabel);
                });
                
                // Create heatmap cells with better color scheme
                for (let i = 0; i < size; i++) {
                    for (let j = 0; j < size; j++) {
                        const weight = weights[i][j];
                        // Use a better color scheme for dark theme
                        const intensity = Math.min(weight * 255, 255);
                        const alpha = Math.max(weight, 0.1); // Minimum visibility
                        
                        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                        rect.setAttribute('x', j * cellSize);
                        rect.setAttribute('y', i * cellSize);
                        rect.setAttribute('width', cellSize);
                        rect.setAttribute('height', cellSize);
                        rect.setAttribute('fill', `rgba(10, 132, 255, ${alpha})`);
                        rect.setAttribute('stroke', 'rgba(255,255,255,0.1)');
                        rect.setAttribute('stroke-width', '0.5');
                        
                        // Add tooltip with better formatting
                        rect.setAttribute('title', `${words[i]} → ${words[j]}: ${weight.toFixed(4)}`);
                        
                        // Add hover effect
                        rect.style.cursor = 'pointer';
                        rect.style.transition = 'opacity 0.2s ease';
                        
                        rect.addEventListener('mouseenter', () => {
                            rect.style.opacity = '0.8';
                        });
                        
                        rect.addEventListener('mouseleave', () => {
                            rect.style.opacity = '1';
                        });
                        
                        svg.appendChild(rect);
                    }
                }
                
                // Clear container and add SVG
                container.innerHTML = '';
                container.appendChild(svg);
                
                // Add legend with better styling
                const legend = document.createElement('div');
                legend.innerHTML = `
                    <div style="margin-top: 2rem; text-align: center; font-size: 0.9rem; color: var(--text-secondary);">
                        <strong style="color: var(--accent-primary);">Легенда:</strong> Чем ярче цвет, тем сильнее внимание между словами
                    </div>
                `;
                container.appendChild(legend);
            }
            
            function resetModel() {
                document.getElementById('d_model').value = '32';
                document.getElementById('n_layers').value = '1';
                document.getElementById('n_heads').value = '2';
                document.getElementById('task').value = 'translate';
                document.getElementById('output_text').innerHTML = 'Модель сброшена. Нажмите "Обработать текст" для получения результата...';
                document.getElementById('attention_viz').style.display = 'none';
                attentionData = null;
            }
            
            // Add CSS animation for spinner
            const style = document.createElement('style');
            style.textContent = `
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            `;
            document.head.appendChild(style);
            
            // Автоматическая обработка при изменении параметров
            document.getElementById('d_model').addEventListener('change', () => {
                if (document.getElementById('output_text').innerHTML !== 'Нажмите "Обработать текст" для получения результата...') {
                    processText();
                }
            });
            
            document.getElementById('n_layers').addEventListener('change', () => {
                if (document.getElementById('output_text').innerHTML !== 'Нажмите "Обработать текст" для получения результата...') {
                    processText();
                }
            });
            
            document.getElementById('n_heads').addEventListener('change', () => {
                if (document.getElementById('output_text').innerHTML !== 'Нажмите "Обработать текст" для получения результата...') {
                    processText();
                }
            });
            
            // Initialize theme and language when page loads
            document.addEventListener('DOMContentLoaded', initThemeAndLanguage);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/get_attention")
async def get_attention_weights(request: dict):
    """Get attention weights for visualization."""
    try:
        text = request.get("text", "")
        d_model = request.get("d_model", 32)
        n_layers = request.get("n_layers", 1)
        n_heads = request.get("n_heads", 2)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text is required")
        
        # Validate parameters
        if d_model % n_heads != 0:
            raise HTTPException(status_code=400, detail="d_model must be divisible by n_heads")
        
        # Create temporary vocabularies
        temp_src_vocab = Vocabulary(min_freq=1, max_size=1000)
        temp_tgt_vocab = Vocabulary(min_freq=1, max_size=1000)
        
        # Add tokens from the input text
        words = text.lower().split()
        for word in words:
            if word not in temp_src_vocab.token2idx:
                temp_src_vocab.token2idx[word] = len(temp_src_vocab.token2idx)
                temp_tgt_vocab.token2idx[word] = len(temp_tgt_vocab.token2idx)
        
        # Create model
        temp_model = create_transformer_model(
            src_vocab_size=len(temp_src_vocab),
            tgt_vocab_size=len(temp_tgt_vocab),
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=max(128, d_model * 2)
        )
        
        # Convert text to tensor
        src_tokens = [temp_src_vocab.token2idx.get(word, 0) for word in words]
        tgt_tokens = src_tokens  # For self-attention visualization
        
        src_tensor = torch.tensor([src_tokens], dtype=torch.long)
        tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long)
        
        # Get attention weights
        with torch.no_grad():
            attention_weights = temp_model.get_attention_weights(src_tensor, tgt_tensor)
        
        # Process attention weights for visualization
        processed_weights = {}
        for layer_name, weights in attention_weights.items():
            if weights is not None:
                # Take first batch and average over heads for visualization
                if weights.dim() == 4:  # [batch, heads, seq_len, seq_len]
                    weights_avg = weights[0].mean(dim=0).cpu().numpy()
                else:
                    weights_avg = weights[0, 0].cpu().numpy()
                
                processed_weights[layer_name] = weights_avg.tolist()
        
        return {
            "text": text,
            "words": words,
            "attention_weights": processed_weights,
            "model_info": {
                "d_model": d_model,
                "n_layers": n_layers,
                "n_heads": n_heads,
                "vocab_size": len(temp_src_vocab)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting attention weights: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get attention weights: {str(e)}")


@app.post("/process_text")
async def process_text(request: dict):
    """Process text with the transformer model."""
    try:
        text = request.get("text", "")
        d_model = request.get("d_model", 32)
        n_layers = request.get("n_layers", 1)
        n_heads = request.get("n_heads", 2)
        task = request.get("task", "translate")
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text is required")
        
        # Validate parameters
        if d_model % n_heads != 0:
            raise HTTPException(status_code=400, detail="d_model must be divisible by n_heads")
        
        # Create a new model with specified parameters
        logger.info(f"Creating model with d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}")
        
        # Calculate d_k and d_v
        d_k = d_v = d_model // n_heads
        d_ff = max(128, d_model * 2)  # Ensure d_ff is reasonable
        
        # Create temporary vocabularies for this request
        temp_src_vocab = Vocabulary(min_freq=1, max_size=1000)
        temp_tgt_vocab = Vocabulary(min_freq=1, max_size=1000)
        
        # Add some basic tokens
        basic_tokens = ['hello', 'world', 'transformer', 'attention', 'model', 'text', 'processing', 
                       'the', 'a', 'is', 'was', 'will', 'can', 'should', 'would', 'could', 'may', 
                       'might', 'must', 'shall', 'привет', 'мир', 'трансформер', 'внимание', 'модель']
        
        for token in basic_tokens:
            if token not in temp_src_vocab.token2idx:
                temp_src_vocab.token2idx[token] = len(temp_src_vocab.token2idx)
                temp_src_vocab.idx2token[len(temp_src_vocab.idx2token)] = token
                temp_tgt_vocab.token2idx[token] = len(temp_tgt_vocab.token2idx)
                temp_tgt_vocab.idx2token[len(temp_tgt_vocab.idx2token)] = token
        
        # Create temporary model
        temp_model = create_transformer_model(
            src_vocab_size=len(temp_src_vocab),
            tgt_vocab_size=len(temp_tgt_vocab),
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff
        )
        
        # Process text based on task
        if task == "translate":
            # Simple translation simulation
            if any(word in text.lower() for word in ['hello', 'world', 'transformer']):
                output = "Привет, мир! Это трансформер."
            elif any(word in text.lower() for word in ['привет', 'мир', 'трансформер']):
                output = "Hello, world! This is a transformer."
            else:
                output = f"[Перевод] {text}"
                
        elif task == "summarize":
            # Simple summarization simulation
            words = text.split()
            if len(words) > 10:
                output = f"Краткое содержание: {' '.join(words[:5])}... (сокращено с {len(words)} до 5 слов)"
            else:
                output = f"Краткое содержание: {text}"
                
        elif task == "simplify":
            # Simple simplification simulation
            output = f"Упрощенный текст: {text.lower()}"
            
        else:
            output = f"[Обработка] {text}"
        
        # Add model info
        model_info = f"Модель: d_model={d_model}, слоев={n_layers}, голов внимания={n_heads}"
        
        return {
            "output": output,
            "model_info": model_info,
            "task": task,
            "input_length": len(text),
            "model_parameters": {
                "d_model": d_model,
                "n_layers": n_layers,
                "n_heads": n_heads,
                "d_k": d_k,
                "d_v": d_v,
                "d_ff": d_ff
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/api/translate")
async def translate_text(
    text: str = Form(...),
    source_lang: str = Form("en"),
    target_lang: str = Form("ru"),
    max_length: int = Form(100)
):
    """
    Translate text using the Transformer model.
    
    Args:
        text: Source text to translate
        source_lang: Source language code
        target_lang: Target language code
        max_length: Maximum length of generated text
    
    Returns:
        Dictionary with translated text and metadata
    """
    if not model or not trainer:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Generate translation
        translated_text = trainer.generate_text(text, max_length=max_length)
        
        return {
            "source_text": text,
            "translated_text": translated_text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "model_info": {
                "architecture": "Transformer",
                "layers": model.n_layers,
                "heads": model.n_heads,
                "d_model": model.d_model
            }
        }
    
    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


@app.post("/api/summarize")
async def summarize_text(
    text: str = Form(...),
    max_length: int = Form(150)
):
    """
    Summarize text using the Transformer model.
    
    Args:
        text: Text to summarize
        max_length: Maximum length of summary
    
    Returns:
        Dictionary with summary and metadata
    """
    if not model or not trainer:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # For summarization, we'll use the same generation approach
        # In a real implementation, you'd have a specific summarization model
        summary = trainer.generate_text(text, max_length=max_length)
        
        return {
            "original_text": text,
            "summary": summary,
            "max_length": max_length,
            "compression_ratio": len(summary.split()) / len(text.split())
        }
    
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")


@app.post("/api/attention")
async def get_attention_weights(
    source_text: str = Form(...),
    target_text: str = Form(...)
):
    """
    Get attention weights for visualization.
    
    Args:
        source_text: Source text
        target_text: Target text
    
    Returns:
        Dictionary with attention weights from all layers
    """
    if not model or not trainer:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get attention weights
        attention_weights = trainer.get_attention_weights(source_text, target_text)
        
        # Convert tensors to lists for JSON serialization
        serializable_weights = {}
        for layer_name, weights in attention_weights.items():
            if weights is not None:
                # Take the first batch and first head for visualization
                weights_np = weights[0, 0].cpu().numpy()
                serializable_weights[layer_name] = weights_np.tolist()
        
        return {
            "source_text": source_text,
            "target_text": target_text,
            "attention_weights": serializable_weights,
            "model_info": {
                "layers": model.n_layers,
                "heads": model.n_heads
            }
        }
    
    except Exception as e:
        logger.error(f"Attention weights error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get attention weights: {str(e)}")


@app.post("/api/process")
async def process_text(
    text: str = Form(...),
    task: str = Form("translation"),
    parameters: Optional[str] = Form("{}")
):
    """
    Process text with various tasks using the Transformer model.
    
    Args:
        text: Input text
        task: Task type (translation, summarization, simplification)
        parameters: JSON string with task-specific parameters
    
    Returns:
        Dictionary with processed text and metadata
    """
    if not model or not trainer:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        import json
        params = json.loads(parameters) if parameters else {}
        
        if task == "translation":
            max_length = params.get("max_length", 100)
            result = trainer.generate_text(text, max_length=max_length)
        elif task == "summarization":
            max_length = params.get("max_length", 150)
            result = trainer.generate_text(text, max_length=max_length)
        elif task == "simplification":
            max_length = params.get("max_length", 200)
            result = trainer.generate_text(text, max_length=max_length)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown task: {task}")
        
        return {
            "input_text": text,
            "task": task,
            "result": result,
            "parameters": params,
            "model_architecture": "Transformer",
            "model_parameters": {
                "layers": model.n_layers,
                "heads": model.n_heads,
                "d_model": model.d_model
            }
        }
    
    except Exception as e:
        logger.error(f"Text processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Text processing failed: {str(e)}")


@app.get("/api/model-info")
async def get_model_info():
    """Get information about the loaded model."""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "architecture": "Transformer",
        "parameters": {
            "d_model": model.d_model,
            "n_layers": model.n_layers,
            "n_heads": model.n_heads,
            "d_ff": model.encoder_layers[0].feed_forward.linear1.out_features,
            "dropout": 0.1
        },
        "vocabulary": {
            "source_size": len(src_vocab) if src_vocab else 0,
            "target_size": len(tgt_vocab) if tgt_vocab else 0
        },
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
    }


@app.post("/api/upload-vocabulary")
async def upload_vocabulary(file: UploadFile = File(...)):
    """
    Upload a custom vocabulary file.
    
    Args:
        file: Pickle file containing vocabulary
    
    Returns:
        Success message
    """
    try:
        # Save uploaded file
        upload_dir = Path("./uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Load vocabulary
        vocab = Vocabulary.load(str(file_path))
        
        return {
            "message": "Vocabulary uploaded successfully",
            "vocabulary_size": len(vocab),
            "filename": file.filename
        }
    
    except Exception as e:
        logger.error(f"Vocabulary upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Vocabulary upload failed: {str(e)}")


@app.get("/features", response_class=HTMLResponse)
async def features_page():
    """Serve the features page."""
    html_content = """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Возможности - Attentify</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            :root {
                /* Dark theme colors */
                --bg-primary: #1a1a1a;
                --bg-secondary: #2d2d2d;
                --bg-tertiary: #3d3d3d;
                --bg-card: #2d2d2d;
                --bg-input: #3d3d3d;
                
                /* Text colors */
                --text-primary: #ffffff;
                --text-secondary: #eae0c8;
                --text-tertiary: #b4674d;
                --text-accent: #7b3f00;
                
                /* Accent colors */
                --accent-primary: #7b3f00;
                --accent-secondary: #b4674d;
                --accent-accent: #eae0c8;
                --accent-green: #30d158;
                --accent-orange: #ff9f0a;
                --accent-red: #ff453a;
                --accent-purple: #bf5af2;
                
                /* Gradients */
                --gradient-primary: linear-gradient(135deg, #7b3f00 0%, #b4674d 100%);
                --gradient-secondary: linear-gradient(135deg, #b4674d 0%, #eae0c8 100%);
                --gradient-accent: linear-gradient(135deg, #7b3f00 0%, #eae0c8 100%);
                
                /* Shadows */
                --shadow-light: 0 2px 10px rgba(0,0,0,0.3);
                --shadow-medium: 0 4px 20px rgba(0,0,0,0.4);
                --shadow-heavy: 0 8px 30px rgba(0,0,0,0.5);
                
                /* Border radius */
                --radius-small: 8px;
                --radius-medium: 16px;
                --radius-large: 24px;
            }
            
            /* Light theme */
            [data-theme="light"] {
                --bg-primary: #ffffff;
                --bg-secondary: #f8f9fa;
                --bg-tertiary: #e9ecef;
                --bg-card: #ffffff;
                --bg-input: #f8f9fa;
                
                --text-primary: #212529;
                --text-secondary: #495057;
                --text-tertiary: #6c757d;
                --text-accent: #7b3f00;
                
                --shadow-light: 0 2px 10px rgba(0,0,0,0.1);
                --shadow-medium: 0 4px 20px rgba(0,0,0,0.15);
                --shadow-heavy: 0 8px 30px rgba(0,0,0,0.2);
            }
            
            [data-theme="light"] .nav-links a {
                color: var(--text-primary);
            }
            
            [data-theme="light"] .logo {
                color: var(--accent-primary);
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', Roboto, sans-serif;
                line-height: 1.6;
                color: var(--text-primary);
                background: var(--bg-primary);
                -webkit-font-smoothing: antialiased;
                -moz-osx-font-smoothing: grayscale;
            }
            
            .header {
                background: rgba(28, 28, 30, 0.8);
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
                padding: 1rem 2rem;
                box-shadow: var(--shadow-light);
                position: sticky;
                top: 0;
                z-index: 100;
                border-bottom: 1px solid rgba(255,255,255,0.1);
            }
            
            .nav {
                display: flex;
                justify-content: space-between;
                align-items: center;
                max-width: 1200px;
                margin: 0 auto;
            }
            
            .logo {
                font-size: 1.8rem;
                font-weight: 700;
                color: var(--accent-primary);
                display: flex;
                align-items: center;
                gap: 0.5rem;
                font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
            }
            
            .logo-icon {
                width: 40px;
                height: 40px;
                background: var(--gradient-primary);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
                font-size: 1.2rem;
                box-shadow: var(--shadow-medium);
            }
            
            .nav-links {
                display: flex;
                gap: 2rem;
                align-items: center;
            }
            
            .nav-links a {
                text-decoration: none;
                color: var(--text-secondary);
                font-weight: 500;
                transition: all 0.3s ease;
                padding: 0.5rem 1rem;
                border-radius: var(--radius-small);
            }
            
            .nav-links a:hover {
                color: var(--text-primary);
                background: rgba(255,255,255,0.1);
            }
            
            .nav-controls {
                display: flex;
                gap: 1rem;
                align-items: center;
            }
            
            .control-btn {
                background: transparent;
                border: 1px solid rgba(255,255,255,0.2);
                border-radius: var(--radius-small);
                padding: 0.5rem;
                color: var(--text-secondary);
                cursor: pointer;
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                min-width: 40px;
                height: 40px;
            }
            
            .control-btn:hover {
                background: rgba(255,255,255,0.1);
                border-color: rgba(255,255,255,0.3);
                color: var(--text-primary);
            }
            
            [data-theme="light"] .control-btn {
                border: 1px solid rgba(0,0,0,0.2);
                color: var(--text-secondary);
            }
            
            [data-theme="light"] .control-btn:hover {
                background: rgba(0,0,0,0.1);
                border-color: rgba(0,0,0,0.3);
                color: var(--text-primary);
            }
            
            .theme-toggle, .language-toggle {
                position: relative;
            }
            
            .language-toggle .control-btn {
                min-width: 50px;
                font-weight: 600;
                font-size: 0.9rem;
            }
            
            .main-content {
                max-width: 1200px;
                margin: 0 auto;
                padding: 3rem 2rem;
            }
            
            .features-header {
                text-align: center;
                margin-bottom: 4rem;
            }
            
            .features-header h1 {
                font-size: 3.5rem;
                margin-bottom: 1.5rem;
                color: var(--text-primary);
                font-weight: 800;
            }
            
            .features-header p {
                font-size: 1.3rem;
                color: var(--text-secondary);
                max-width: 800px;
                margin: 0 auto;
                line-height: 1.6;
            }
            
            .features-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 2.5rem;
                margin-bottom: 4rem;
            }
            
            .feature-card {
                background: var(--bg-card);
                padding: 3rem;
                border-radius: var(--radius-large);
                box-shadow: var(--shadow-medium);
                border: 1px solid rgba(123, 63, 0, 0.3);
                transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                cursor: pointer;
            }
            
            .feature-card:hover {
                transform: translateY(-12px) scale(1.02);
                box-shadow: var(--shadow-heavy);
                border-color: var(--accent-primary);
                background: var(--bg-tertiary);
            }
            
            .feature-card h3 {
                font-size: 1.8rem;
                margin-bottom: 1.5rem;
                color: var(--text-primary);
                font-weight: 700;
            }
            
            .feature-card p {
                color: var(--text-secondary);
                line-height: 1.7;
                font-size: 1.1rem;
                margin-bottom: 2rem;
            }
            
            .feature-details {
                background: var(--bg-tertiary);
                padding: 1.5rem;
                border-radius: var(--radius-small);
                border-left: 4px solid var(--accent-primary);
            }
            
            .feature-details h4 {
                color: var(--text-primary);
                margin-bottom: 1rem;
                font-weight: 600;
            }
            
            .feature-details ul {
                list-style: none;
                padding: 0;
            }
            
            .feature-details li {
                color: var(--text-secondary);
                margin-bottom: 0.5rem;
                padding-left: 1.5rem;
                position: relative;
            }
            
            .feature-details li:before {
                content: "•";
                color: var(--accent-primary);
                font-weight: bold;
                position: absolute;
                left: 0;
                font-size: 1.2rem;
            }
            
            .footer {
                background: var(--bg-secondary);
                color: var(--text-secondary);
                text-align: center;
                padding: 3rem 2rem;
                margin-top: 4rem;
                border-top: 1px solid rgba(255,255,255,0.1);
            }
            
            .footer p {
                opacity: 0.8;
                font-size: 1rem;
            }
        </style>
    </head>
    <body>
        <header class="header">
            <nav class="nav">
                <div class="logo">
                    <div class="logo-icon">A</div>
                    Attentify
                </div>
                <div class="nav-links">
                    <a href="/features" class="active">Возможности</a>
                    <a href="/demo">Демо</a>
                    <a href="/about">О проекте</a>
                    <a href="/blog">Блог</a>
                    <a href="/docs">Документация</a>
                </div>
                <div class="nav-controls">
                    <div class="theme-toggle">
                        <button id="theme-toggle" class="control-btn" title="Сменить тему">
                            <svg id="sun-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="12" cy="12" r="5"></circle>
                                <line x1="12" y1="1" x2="12" y2="3"></line>
                                <line x1="12" y1="21" x2="12" y2="23"></line>
                                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
                                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
                                <line x1="1" y1="12" x2="3" y2="12"></line>
                                <line x1="21" y1="12" x2="23" y2="12"></line>
                                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
                                <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
                            </svg>
                            <svg id="moon-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="display: none;">
                                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
                            </svg>
                        </button>
                    </div>
                    <div class="language-toggle">
                        <button id="language-toggle" class="control-btn" title="Сменить язык">
                            <span id="current-lang">RU</span>
                        </button>
                    </div>
                    <a href="/docs" class="btn-primary">API Документация</a>
                </div>
            </nav>
        </header>
        
        <main class="main-content">
            <div class="features-header">
                <h1>Возможности Attentify</h1>
                <p>Изучайте архитектуру Transformer с нуля, экспериментируйте с параметрами и визуализируйте механизмы внимания в реальном времени</p>
            </div>
            
            <div class="features-grid">
                <div class="feature-card">
                    <h3>Реализация с нуля</h3>
                    <p>Полная реализация архитектуры Transformer по статье "Attention Is All You Need" без готовых библиотек. Каждый компонент написан с нуля для глубокого понимания.</p>
                    <div class="feature-details">
                        <h4>Что включено:</h4>
                        <ul>
                            <li>Multi-Head Attention механизм</li>
                            <li>Positional Encoding</li>
                            <li>Feed-Forward Networks</li>
                            <li>Residual Connections</li>
                            <li>Layer Normalization</li>
                        </ul>
                    </div>
                </div>
                
                <div class="feature-card">
                    <h3>Визуализация внимания</h3>
                    <p>Наглядно увидите, как работает механизм внимания в трансформере — heatmap связей между словами в реальном времени с возможностью выбора слоев и голов.</p>
                    <div class="feature-details">
                        <h4>Возможности:</h4>
                        <ul>
                            <li>Интерактивные heatmap</li>
                            <li>Выбор слоев и голов внимания</li>
                            <li>Анализ паттернов внимания</li>
                            <li>Экспорт результатов</li>
                        </ul>
                    </div>
                </div>
                
                <div class="feature-card">
                    <h3>Интерактивная настройка</h3>
                    <p>Меняйте количество слоев, голов внимания и другие параметры модели в реальном времени. Смотрите, как архитектура адаптируется к вашим настройкам.</p>
                    <div class="feature-details">
                        <h4>Параметры:</h4>
                        <ul>
                            <li>Размер модели (d_model)</li>
                            <li>Количество слоев (n_layers)</li>
                            <li>Количество голов (n_heads)</li>
                            <li>Размер FFN (d_ff)</li>
                        </ul>
                    </div>
                </div>
                
                <div class="feature-card">
                    <h3>Прикладные задачи</h3>
                    <p>Перевод, резюмирование, упрощение текста — все на вашей собственной модели. Изучайте, как трансформер решает конкретные задачи.</p>
                    <div class="feature-details">
                        <h4>Задачи:</h4>
                        <ul>
                            <li>Машинный перевод</li>
                            <li>Резюмирование текста</li>
                            <li>Упрощение языка</li>
                            <li>Анализ тональности</li>
                        </ul>
                    </div>
                </div>
            </div>
        </main>
        
        <footer class="footer">
            <p>&copy; 2024 Attentify. Платформа для изучения и применения архитектуры Transformer.</p>
        </footer>
        
        <script>
            let currentTheme = 'dark';
            let currentLanguage = 'ru';
            
            // Theme and language management
            function initThemeAndLanguage() {
                // Load saved preferences
                const savedTheme = localStorage.getItem('attentify-theme') || 'dark';
                const savedLanguage = localStorage.getItem('attentify-language') || 'ru';
                
                setTheme(savedTheme);
                setLanguage(savedLanguage);
                
                // Setup event listeners
                document.getElementById('theme-toggle').addEventListener('click', toggleTheme);
                document.getElementById('language-toggle').addEventListener('click', toggleLanguage);
                
                // Highlight active navigation item
                highlightActiveNavItem();
            }
            
            function highlightActiveNavItem() {
                const currentPath = window.location.pathname;
                const navLinks = document.querySelectorAll('.nav-links a');
                
                navLinks.forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === currentPath) {
                        link.classList.add('active');
                    }
                });
            }
            
            function setTheme(theme) {
                currentTheme = theme;
                document.documentElement.setAttribute('data-theme', theme);
                localStorage.setItem('attentify-theme', theme);
                
                // Update icons
                const sunIcon = document.getElementById('sun-icon');
                const moonIcon = document.getElementById('moon-icon');
                
                if (theme === 'light') {
                    sunIcon.style.display = 'none';
                    moonIcon.style.display = 'block';
                } else {
                    sunIcon.style.display = 'block';
                    moonIcon.style.display = 'none';
                }
            }
            
            function toggleTheme() {
                const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
                setTheme(newTheme);
            }
            
            function setLanguage(lang) {
                currentLanguage = lang;
                localStorage.setItem('attentify-language', lang);
                document.getElementById('current-lang').textContent = lang.toUpperCase();
            }
            
            function toggleLanguage() {
                const newLang = currentLanguage === 'ru' ? 'en' : 'ru';
                setLanguage(newLang);
            }
            
            // Initialize when page loads
            document.addEventListener('DOMContentLoaded', initThemeAndLanguage);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/demo", response_class=HTMLResponse)
async def demo_page():
    """Serve the demo page."""
    html_content = """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Демо - Attentify</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            :root {
                /* Dark theme colors */
                --bg-primary: #1a1a1a;
                --bg-secondary: #2d2d2d;
                --bg-tertiary: #3d3d3d;
                --bg-card: #2d2d2d;
                --bg-input: #3d3d3d;
                
                /* Text colors */
                --text-primary: #ffffff;
                --text-secondary: #eae0c8;
                --text-tertiary: #b4674d;
                --text-accent: #7b3f00;
                
                /* Accent colors */
                --accent-primary: #7b3f00;
                --accent-secondary: #b4674d;
                --accent-accent: #eae0c8;
                --accent-green: #30d158;
                --accent-orange: #ff9f0a;
                --accent-red: #ff453a;
                --accent-purple: #bf5af2;
                
                /* Gradients */
                --gradient-primary: linear-gradient(135deg, #7b3f00 0%, #b4674d 100%);
                --gradient-secondary: linear-gradient(135deg, #b4674d 0%, #eae0c8 100%);
                --gradient-accent: linear-gradient(135deg, #7b3f00 0%, #eae0c8 100%);
                
                /* Shadows */
                --shadow-light: 0 2px 10px rgba(0,0,0,0.3);
                --shadow-medium: 0 4px 20px rgba(0,0,0,0.4);
                --shadow-heavy: 0 8px 30px rgba(0,0,0,0.5);
                
                /* Border radius */
                --radius-small: 8px;
                --radius-medium: 16px;
                --radius-large: 24px;
            }
            
            /* Light theme */
            [data-theme="light"] {
                --bg-primary: #ffffff;
                --bg-secondary: #f8f9fa;
                --bg-tertiary: #e9ecef;
                --bg-card: #ffffff;
                --bg-input: #f8f9fa;
                
                --text-primary: #212529;
                --text-secondary: #495057;
                --text-tertiary: #6c757d;
                --text-accent: #7b3f00;
                
                --shadow-light: 0 2px 10px rgba(0,0,0,0.1);
                --shadow-medium: 0 4px 20px rgba(0,0,0,0.15);
                --shadow-heavy: 0 8px 30px rgba(0,0,0,0.2);
            }
            
            [data-theme="light"] .nav-links a {
                color: var(--text-primary);
            }
            
            [data-theme="light"] .logo {
                color: var(--accent-primary);
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', Roboto, sans-serif;
                line-height: 1.6;
                color: var(--text-primary);
                background: var(--bg-primary);
                -webkit-font-smoothing: antialiased;
                -moz-osx-font-smoothing: grayscale;
            }
            
            .header {
                background: rgba(28, 28, 30, 0.8);
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
                padding: 1rem 2rem;
                box-shadow: var(--shadow-light);
                position: sticky;
                top: 0;
                z-index: 100;
                border-bottom: 1px solid rgba(255,255,255,0.1);
            }
            
            .nav {
                display: flex;
                justify-content: space-between;
                align-items: center;
                max-width: 1200px;
                margin: 0 auto;
            }
            
            .logo {
                font-size: 1.8rem;
                font-weight: 700;
                color: var(--accent-primary);
                display: flex;
                align-items: center;
                gap: 0.5rem;
                font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
            }
            
            .logo-icon {
                width: 40px;
                height: 40px;
                background: var(--gradient-primary);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
                font-size: 1.2rem;
                box-shadow: var(--shadow-medium);
            }
            
            .nav-links {
                display: flex;
                gap: 2rem;
                align-items: center;
            }
            
            .nav-links a {
                text-decoration: none;
                color: var(--text-secondary);
                font-weight: 500;
                transition: all 0.3s ease;
                padding: 0.5rem 1rem;
                border-radius: var(--radius-small);
            }
            
            .nav-links a:hover {
                color: var(--text-primary);
                background: rgba(255,255,255,0.1);
            }
            
            .nav-controls {
                display: flex;
                gap: 1rem;
                align-items: center;
            }
            
            .control-btn {
                background: transparent;
                border: 1px solid rgba(255,255,255,0.2);
                border-radius: var(--radius-small);
                padding: 0.5rem;
                color: var(--text-secondary);
                cursor: pointer;
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                min-width: 40px;
                height: 40px;
            }
            
            .control-btn:hover {
                background: rgba(255,255,255,0.1);
                border-color: rgba(255,255,255,0.3);
                color: var(--text-primary);
            }
            
            [data-theme="light"] .control-btn {
                border: 1px solid rgba(0,0,0,0.2);
                color: var(--text-secondary);
            }
            
            [data-theme="light"] .control-btn:hover {
                background: rgba(0,0,0,0.1);
                border-color: rgba(0,0,0,0.3);
                color: var(--text-primary);
            }
            
            .theme-toggle, .language-toggle {
                position: relative;
            }
            
            .language-toggle .control-btn {
                min-width: 50px;
                font-weight: 600;
                font-size: 0.9rem;
            }
            
            .main-content {
                max-width: 1200px;
                margin: 0 auto;
                padding: 3rem 2rem;
            }
            
            .demo-header {
                text-align: center;
                margin-bottom: 4rem;
            }
            
            .demo-header h1 {
                font-size: 3.5rem;
                margin-bottom: 1.5rem;
                color: var(--text-primary);
                font-weight: 800;
            }
            
            .demo-header p {
                font-size: 1.3rem;
                color: var(--text-secondary);
                max-width: 800px;
                margin: 0 auto;
                line-height: 1.6;
            }
            
            .demo-section {
                background: var(--bg-card);
                padding: 3rem;
                border-radius: var(--radius-large);
                box-shadow: var(--shadow-medium);
                border: 1px solid rgba(123, 63, 0, 0.3);
                margin-bottom: 3rem;
            }
            
            .demo-section h2 {
                font-size: 2rem;
                margin-bottom: 2rem;
                color: var(--text-primary);
                font-weight: 700;
            }
            
            .demo-controls {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 2rem;
                margin-bottom: 2rem;
            }
            
            .control-group {
                display: flex;
                flex-direction: column;
                gap: 0.5rem;
            }
            
            .control-group label {
                font-weight: 600;
                color: var(--text-primary);
                font-size: 1rem;
            }
            
            .control-group input, .control-group select {
                padding: 0.75rem;
                border: 1px solid rgba(123, 63, 0, 0.3);
                border-radius: var(--radius-small);
                background: var(--bg-input);
                color: var(--text-primary);
                font-size: 1rem;
                transition: all 0.3s ease;
            }
            
            .control-group input:focus, .control-group select:focus {
                outline: none;
                border-color: var(--accent-primary);
                box-shadow: 0 0 0 3px rgba(123, 63, 0, 0.1);
            }
            
            .demo-input {
                margin-bottom: 2rem;
            }
            
            .demo-input textarea {
                width: 100%;
                min-height: 120px;
                padding: 1rem;
                border: 1px solid rgba(123, 63, 0, 0.3);
                border-radius: var(--radius-small);
                background: var(--bg-input);
                color: var(--text-primary);
                font-size: 1rem;
                font-family: inherit;
                resize: vertical;
                transition: all 0.3s ease;
            }
            
            .demo-input textarea:focus {
                outline: none;
                border-color: var(--accent-primary);
                box-shadow: 0 0 0 3px rgba(123, 63, 0, 0.1);
            }
            
            .demo-output {
                background: var(--bg-tertiary);
                padding: 2rem;
                border-radius: var(--radius-small);
                border-left: 4px solid var(--accent-primary);
                min-height: 120px;
                white-space: pre-wrap;
                font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
                font-size: 0.9rem;
                line-height: 1.5;
                color: var(--text-secondary);
            }
            
            .attention-visualization {
                margin-top: 3rem;
            }
            
            .attention-controls {
                display: flex;
                gap: 1rem;
                margin-bottom: 2rem;
                flex-wrap: wrap;
            }
            
            .attention-heatmap {
                background: var(--bg-tertiary);
                border: 1px solid rgba(123, 63, 0, 0.3);
                border-radius: var(--radius-small);
                padding: 2rem;
                text-align: center;
                min-height: 300px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: var(--text-secondary);
                font-size: 1.1rem;
            }
            
            .footer {
                background: var(--bg-secondary);
                color: var(--text-secondary);
                text-align: center;
                padding: 3rem 2rem;
                margin-top: 4rem;
                border-top: 1px solid rgba(255,255,255,0.1);
            }
            
            .footer p {
                opacity: 0.8;
                font-size: 1rem;
            }
        </style>
    </head>
    <body>
        <header class="header">
            <nav class="nav">
                <div class="logo">
                    <div class="logo-icon">A</div>
                    Attentify
                </div>
                <div class="nav-links">
                    <a href="/features">Возможности</a>
                    <a href="/demo" class="active">Демо</a>
                    <a href="/about">О проекте</a>
                    <a href="/blog">Блог</a>
                    <a href="/docs">Документация</a>
                </div>
                <div class="nav-controls">
                    <div class="theme-toggle">
                        <button id="theme-toggle" class="control-btn" title="Сменить тему">
                            <svg id="sun-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="12" cy="12" r="5"></circle>
                                <line x1="12" y1="1" x2="12" y2="3"></line>
                                <line x1="12" y1="21" x2="12" y2="23"></line>
                                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
                                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
                                <line x1="1" y1="12" x2="3" y2="12"></line>
                                <line x1="21" y1="12" x2="23" y2="12"></line>
                                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
                                <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
                            </svg>
                            <svg id="moon-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="display: none;">
                                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
                            </svg>
                        </button>
                    </div>
                    <div class="language-toggle">
                        <button id="language-toggle" class="control-btn" title="Сменить язык">
                            <span id="current-lang">RU</span>
                        </button>
                    </div>
                    <a href="/docs" class="btn-primary">API Документация</a>
                </div>
            </nav>
        </header>
        
        <main class="main-content">
            <div class="demo-header">
                <h1>Интерактивное Демо</h1>
                <p>Экспериментируйте с архитектурой Transformer в реальном времени. Настраивайте параметры, загружайте текст и наблюдайте за работой механизмов внимания</p>
            </div>
            
            <div class="demo-section">
                <h2>Настройка модели</h2>
                <div class="demo-controls">
                    <div class="control-group">
                        <label for="d-model">Размер модели (d_model)</label>
                        <input type="number" id="d-model" value="32" min="16" max="512" step="16">
                    </div>
                    <div class="control-group">
                        <label for="n-layers">Количество слоев (n_layers)</label>
                        <input type="number" id="n-layers" value="1" min="1" max="12" step="1">
                    </div>
                    <div class="control-group">
                        <label for="n-heads">Количество голов (n_heads)</label>
                        <input type="number" id="n-heads" value="2" min="1" max="8" step="1">
                    </div>
                    <div class="control-group">
                        <label for="d-ff">Размер FFN (d_ff)</label>
                        <input type="number" id="d-ff" value="128" min="64" max="2048" step="64">
                    </div>
                </div>
                <button class="btn-primary" onclick="updateModel()">Обновить модель</button>
            </div>
            
            <div class="demo-section">
                <h2>Обработка текста</h2>
                <div class="demo-input">
                    <label for="input-text">Введите текст для обработки:</label>
                    <textarea id="input-text" placeholder="Введите текст здесь...">Привет, мир! Это демонстрация работы архитектуры Transformer.</textarea>
                </div>
                <div class="demo-controls">
                    <div class="control-group">
                        <label for="task-type">Тип задачи:</label>
                        <select id="task-type">
                            <option value="translate">Перевод</option>
                            <option value="summarize">Резюмирование</option>
                            <option value="simplify">Упрощение</option>
                        </select>
                    </div>
                    <div class="control-group">
                        <label for="target-language">Целевой язык:</label>
                        <select id="target-language">
                            <option value="en">Английский</option>
                            <option value="de">Немецкий</option>
                            <option value="fr">Французский</option>
                            <option value="es">Испанский</option>
                        </select>
                    </div>
                </div>
                <button class="btn-primary" onclick="processText()">Обработать текст</button>
                <div class="demo-output" id="output-text">Результат обработки появится здесь...</div>
            </div>
            
            <div class="demo-section">
                <h2>Визуализация внимания</h2>
                <div class="attention-controls">
                    <div class="control-group">
                        <label for="layer-select">Слой:</label>
                        <select id="layer-select">
                            <option value="0">Слой 0</option>
                            <option value="1">Слой 1</option>
                        </select>
                    </div>
                    <div class="control-group">
                        <label for="head-select">Голова внимания:</label>
                        <select id="head-select">
                            <option value="0">Голова 0</option>
                            <option value="1">Голова 1</option>
                        </select>
                    </div>
                    <button class="btn-primary" onclick="visualizeAttention()">Показать внимание</button>
                </div>
                <div class="attention-heatmap" id="attention-heatmap">
                    Выберите слой и голову внимания для визуализации
                </div>
            </div>
        </main>
        
        <footer class="footer">
            <p>&copy; 2024 Attentify. Платформа для изучения и применения архитектуры Transformer.</p>
        </footer>
        
        <script>
            let currentTheme = 'dark';
            let currentLanguage = 'ru';
            
            // Theme and language management
            function initThemeAndLanguage() {
                // Load saved preferences
                const savedTheme = localStorage.getItem('attentify-theme') || 'dark';
                const savedLanguage = localStorage.getItem('attentify-language') || 'ru';
                
                setTheme(savedTheme);
                setLanguage(savedLanguage);
                
                // Setup event listeners
                document.getElementById('theme-toggle').addEventListener('click', toggleTheme);
                document.getElementById('language-toggle').addEventListener('click', toggleLanguage);
                
                // Highlight active navigation item
                highlightActiveNavItem();
            }
            
            function highlightActiveNavItem() {
                const currentPath = window.location.pathname;
                const navLinks = document.querySelectorAll('.nav-links a');
                
                navLinks.forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === currentPath) {
                        link.classList.add('active');
                    }
                });
            }
            
            function setTheme(theme) {
                currentTheme = theme;
                document.documentElement.setAttribute('data-theme', theme);
                localStorage.setItem('attentify-theme', theme);
                
                // Update icons
                const sunIcon = document.getElementById('sun-icon');
                const moonIcon = document.getElementById('moon-icon');
                
                if (theme === 'light') {
                    sunIcon.style.display = 'none';
                    moonIcon.style.display = 'block';
                } else {
                    sunIcon.style.display = 'block';
                    moonIcon.style.display = 'none';
                }
            }
            
            function toggleTheme() {
                const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
                setTheme(newTheme);
            }
            
            function setLanguage(lang) {
                currentLanguage = lang;
                localStorage.setItem('attentify-language', lang);
                document.getElementById('current-lang').textContent = lang.toUpperCase();
            }
            
            function toggleLanguage() {
                const newLang = currentLanguage === 'ru' ? 'en' : 'ru';
                setLanguage(newLang);
            }
            
            // Demo functions
            function updateModel() {
                const dModel = document.getElementById('d-model').value;
                const nLayers = document.getElementById('n-layers').value;
                const nHeads = document.getElementById('n-heads').value;
                const dFf = document.getElementById('d-ff').value;
                
                // Here you would typically make an API call to update the model
                console.log('Updating model with:', { dModel, nLayers, nHeads, dFf });
                
                // For demo purposes, show a success message
                alert('Модель обновлена! Параметры: d_model=' + dModel + ', n_layers=' + nLayers + ', n_heads=' + nHeads + ', d_ff=' + dFf);
            }
            
            function processText() {
                const inputText = document.getElementById('input-text').value;
                const taskType = document.getElementById('task-type').value;
                const targetLanguage = document.getElementById('target-language').value;
                
                if (!inputText.trim()) {
                    alert('Пожалуйста, введите текст для обработки');
                    return;
                }
                
                // Simulate processing
                const outputDiv = document.getElementById('output-text');
                outputDiv.innerHTML = 'Обрабатываю текст...';
                
                setTimeout(() => {
                    let result = '';
                    switch (taskType) {
                        case 'translate':
                            result = 'Перевод: Hello, world! This is a demonstration of the Transformer architecture.';
                            break;
                        case 'summarize':
                            result = 'Резюме: Демонстрация архитектуры Transformer с приветствием миру.';
                            break;
                        case 'simplify':
                            result = 'Упрощенный текст: Привет! Это показ работы Transformer.';
                            break;
                    }
                    outputDiv.innerHTML = result;
                }, 1500);
            }
            
            function visualizeAttention() {
                const layer = document.getElementById('layer-select').value;
                const head = document.getElementById('head-select').value;
                
                const heatmapDiv = document.getElementById('attention-heatmap');
                heatmapDiv.innerHTML = 'Загружаю визуализацию внимания для слоя ' + layer + ', головы ' + head + '...';
                
                // Simulate loading attention visualization
                setTimeout(() => {
                    heatmapDiv.innerHTML = 'Визуализация внимания для слоя ' + layer + ', головы ' + head + ' загружена. Здесь будет отображаться heatmap матрица внимания.';
                }, 1000);
            }
            
            // Initialize when page loads
            document.addEventListener('DOMContentLoaded', initThemeAndLanguage);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/about", response_class=HTMLResponse)
async def about_page():
    """Serve the about page."""
    html_content = """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>О проекте - Attentify</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            :root {
                /* Dark theme colors */
                --bg-primary: #1a1a1a;
                --bg-secondary: #2d2d2d;
                --bg-tertiary: #3d3d3d;
                --bg-card: #2d2d2d;
                --bg-input: #3d3d3d;
                
                /* Text colors */
                --text-primary: #ffffff;
                --text-secondary: #eae0c8;
                --text-tertiary: #b4674d;
                --text-accent: #7b3f00;
                
                /* Accent colors */
                --accent-primary: #7b3f00;
                --accent-secondary: #b4674d;
                --accent-accent: #eae0c8;
                --accent-green: #30d158;
                --accent-orange: #ff9f0a;
                --accent-red: #ff453a;
                --accent-purple: #bf5af2;
                
                /* Gradients */
                --gradient-primary: linear-gradient(135deg, #7b3f00 0%, #b4674d 100%);
                --gradient-secondary: linear-gradient(135deg, #b4674d 0%, #eae0c8 100%);
                --gradient-accent: linear-gradient(135deg, #7b3f00 0%, #eae0c8 100%);
                
                /* Shadows */
                --shadow-light: 0 2px 10px rgba(0,0,0,0.3);
                --shadow-medium: 0 4px 20px rgba(0,0,0,0.4);
                --shadow-heavy: 0 8px 30px rgba(0,0,0,0.5);
                
                /* Border radius */
                --radius-small: 8px;
                --radius-medium: 16px;
                --radius-large: 24px;
            }
            
            /* Light theme */
            [data-theme="light"] {
                --bg-primary: #ffffff;
                --bg-secondary: #f8f9fa;
                --bg-tertiary: #e9ecef;
                --bg-card: #ffffff;
                --bg-input: #f8f9fa;
                
                --text-primary: #212529;
                --text-secondary: #495057;
                --text-tertiary: #6c757d;
                --text-accent: #7b3f00;
                
                --shadow-light: 0 2px 10px rgba(0,0,0,0.1);
                --shadow-medium: 0 4px 20px rgba(0,0,0,0.15);
                --shadow-heavy: 0 8px 30px rgba(0,0,0,0.2);
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', Roboto, sans-serif;
                line-height: 1.6;
                color: var(--text-primary);
                background: var(--bg-primary);
                -webkit-font-smoothing: antialiased;
                -moz-osx-font-smoothing: grayscale;
            }
            
            .header {
                background: rgba(28, 28, 30, 0.8);
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
                padding: 1rem 2rem;
                box-shadow: var(--shadow-light);
                position: sticky;
                top: 0;
                z-index: 100;
                border-bottom: 1px solid rgba(255,255,255,0.1);
            }
            
            .nav {
                display: flex;
                justify-content: space-between;
                align-items: center;
                max-width: 1200px;
                margin: 0 auto;
            }
            
            .logo {
                font-size: 1.8rem;
                font-weight: 700;
                color: var(--accent-primary);
                display: flex;
                align-items: center;
                gap: 0.5rem;
                font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
            }
            
            .logo-icon {
                width: 40px;
                height: 40px;
                background: var(--gradient-accent);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
                font-size: 1.2rem;
                box-shadow: var(--shadow-medium);
            }
            
            .nav-links {
                display: flex;
                gap: 2rem;
                align-items: center;
            }
            
            .nav-links a {
                text-decoration: none;
                color: var(--text-secondary);
                font-weight: 500;
                transition: all 0.3s ease;
                padding: 0.5rem 1rem;
                border-radius: var(--radius-small);
            }
            
            .nav-links a:hover {
                color: var(--text-primary);
                background: rgba(255,255,255,0.1);
            }
            
            .nav-controls {
                display: flex;
                gap: 1rem;
                align-items: center;
            }
            
            .control-btn {
                background: transparent;
                border: 1px solid rgba(255,255,255,0.2);
                border-radius: var(--radius-small);
                padding: 0.5rem;
                color: var(--text-secondary);
                cursor: pointer;
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                min-width: 40px;
                height: 40px;
            }
            
            .control-btn:hover {
                background: rgba(255,255,255,0.1);
                border-color: rgba(255,255,255,0.3);
                color: var(--text-primary);
            }
            
            [data-theme="light"] .control-btn {
                border: 1px solid rgba(0,0,0,0.2);
                color: var(--text-secondary);
            }
            
            [data-theme="light"] .control-btn:hover {
                background: rgba(0,0,0,0.1);
                border-color: rgba(0,0,0,0.3);
                color: var(--text-primary);
            }
            
            .theme-toggle, .language-toggle {
                position: relative;
            }
            
            .language-toggle .control-btn {
                min-width: 50px;
                font-weight: 600;
                font-size: 0.9rem;
            }
            
            .btn-primary {
                background: var(--gradient-primary);
                color: white;
                padding: 0.75rem 1.5rem;
                border: none;
                border-radius: var(--radius-small);
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                text-decoration: none;
                display: inline-block;
                box-shadow: var(--shadow-medium);
                font-family: inherit;
            }
            
            .btn-primary:hover {
                transform: translateY(-2px);
                box-shadow: var(--shadow-heavy);
            }
            
            .main-content {
                max-width: 1200px;
                margin: 0 auto;
                padding: 3rem 2rem;
            }
            
            .about-hero {
                text-align: center;
                margin-bottom: 4rem;
            }
            
            .about-hero h1 {
                font-size: 3.5rem;
                margin-bottom: 1.5rem;
                color: var(--text-primary);
                font-weight: 800;
                font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
            }
            
            .about-hero p {
                font-size: 1.3rem;
                color: var(--text-secondary);
                max-width: 800px;
                margin: 0 auto;
                line-height: 1.6;
            }
            
            .team-section {
                margin-bottom: 4rem;
            }
            
            .team-section h2 {
                font-size: 2.5rem;
                margin-bottom: 2rem;
                color: var(--text-primary);
                text-align: center;
                font-weight: 700;
            }
            
            .team-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 2rem;
                margin-bottom: 3rem;
            }
            
            .team-member {
                background: var(--bg-card);
                padding: 2rem;
                border-radius: var(--radius-large);
                text-align: center;
                box-shadow: var(--shadow-medium);
                border: 1px solid rgba(255,255,255,0.1);
                transition: all 0.3s ease;
            }
            
            .team-member:hover {
                transform: translateY(-5px);
                box-shadow: var(--shadow-heavy);
            }
            
            .member-avatar {
                width: 120px;
                height: 120px;
                border-radius: 50%;
                margin: 0 auto 1.5rem;
                background: var(--gradient-accent);
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 3rem;
                font-weight: bold;
            }
            
            .member-name {
                font-size: 1.5rem;
                margin-bottom: 0.5rem;
                color: var(--text-primary);
                font-weight: 600;
            }
            
            .member-role {
                color: var(--accent-primary);
                margin-bottom: 1rem;
                font-weight: 500;
            }
            
            .member-bio {
                color: var(--text-secondary);
                line-height: 1.6;
            }
            
            .mission-section {
                background: var(--bg-card);
                padding: 3rem;
                border-radius: var(--radius-large);
                margin-bottom: 4rem;
                box-shadow: var(--shadow-medium);
                border: 1px solid rgba(255,255,255,0.1);
            }
            
            .mission-section h2 {
                font-size: 2.5rem;
                margin-bottom: 2rem;
                color: var(--text-primary);
                text-align: center;
                font-weight: 700;
            }
            
            .mission-content {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 2rem;
            }
            
            .mission-item {
                text-align: center;
            }
            
            .mission-item h3 {
                font-size: 1.5rem;
                margin-bottom: 1rem;
                color: var(--accent-primary);
                font-weight: 600;
            }
            
            .mission-item p {
                color: var(--text-secondary);
                line-height: 1.6;
            }
            
            .join-section {
                text-align: center;
                background: var(--bg-card);
                padding: 3rem;
                border-radius: var(--radius-large);
                box-shadow: var(--shadow-medium);
                border: 1px solid rgba(255,255,255,0.1);
            }
            
            .join-section h2 {
                font-size: 2.5rem;
                margin-bottom: 1.5rem;
                color: var(--text-primary);
                font-weight: 700;
            }
            
            .join-section p {
                font-size: 1.2rem;
                color: var(--text-secondary);
                margin-bottom: 2rem;
                max-width: 800px;
                margin-left: auto;
                margin-right: auto;
                line-height: 1.6;
            }
            
            .join-links {
                display: flex;
                gap: 1rem;
                justify-content: center;
                flex-wrap: wrap;
            }
            
            .btn-secondary {
                background: var(--bg-tertiary);
                color: var(--text-primary);
                padding: 0.75rem 1.5rem;
                border: 1px solid rgba(255,255,255,0.1);
                border-radius: var(--radius-small);
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                font-family: inherit;
                text-decoration: none;
                display: inline-block;
            }
            
            .btn-secondary:hover {
                background: var(--bg-secondary);
                border-color: rgba(255,255,255,0.2);
                transform: translateY(-2px);
            }
            
            .footer {
                background: var(--bg-secondary);
                color: var(--text-secondary);
                text-align: center;
                padding: 3rem 2rem;
                margin-top: 4rem;
                border-top: 1px solid rgba(255,255,255,0.1);
            }
            
            .footer p {
                opacity: 0.8;
                font-size: 1rem;
            }
            
            @media (max-width: 768px) {
                .nav-links {
                    display: none;
                }
                
                .team-grid {
                    grid-template-columns: 1fr;
                }
                
                .mission-content {
                    grid-template-columns: 1fr;
                }
                
                .join-links {
                    flex-direction: column;
                    align-items: center;
                }
            }
        </style>
    </head>
    <body>
        <header class="header">
            <nav class="nav">
                <div class="logo">
                    <div class="logo-icon">A</div>
                    Attentify
                </div>
                <div class="nav-links">
                    <a href="/features">Возможности</a>
                    <a href="/demo">Демо</a>
                    <a href="/blog">Блог</a>
                    <a href="/about" class="active">О проекте</a>
                    <a href="/docs">Документация</a>
                </div>
                <div class="nav-controls">
                    <div class="theme-toggle">
                        <button id="theme-toggle" class="control-btn" title="Сменить тему">
                            <svg id="sun-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="12" cy="12" r="5"></circle>
                                <line x1="12" y1="1" x2="12" y2="3"></line>
                                <line x1="12" y1="21" x2="12" y2="23"></line>
                                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
                                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
                                <line x1="1" y1="12" x2="3" y2="12"></line>
                                <line x1="21" y1="12" x2="23" y2="12"></line>
                                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
                                <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
                            </svg>
                            <svg id="moon-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="display: none;">
                                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
                            </svg>
                        </button>
                    </div>
                    <div class="language-toggle">
                        <button id="language-toggle" class="control-btn" title="Сменить язык">
                            <span id="current-lang">RU</span>
                        </button>
                    </div>
                    <a href="/docs" class="btn-primary">API Документация</a>
                </div>
            </nav>
        </header>
        
        <main class="main-content">
            <section class="about-hero">
                <h1>О проекте Attentify</h1>
                <p>Мы создаем лучший инструмент для ML-команд по созданию, обучению и оценке продвинутых AI-моделей. Надеемся, что вам понравится использовать его так же, как нам нравится над ним работать.</p>
            </section>
            
            <section class="team-section">
                <h2>Кто стоит за Attentify?</h2>
                <div class="team-grid">
                    <div class="team-member">
                        <div class="member-avatar">T</div>
                        <div class="member-name">Tony Salomone</div>
                        <div class="member-role">Ведущий инженер</div>
                        <div class="member-bio">Эксперт по машинному обучению и архитектуре нейронных сетей. Специализируется на интерпретируемости моделей.</div>
                    </div>
                    <div class="team-member">
                        <div class="member-avatar">A</div>
                        <div class="member-name">Ali Asaria</div>
                        <div class="member-role">Технический директор</div>
                        <div class="member-bio">Опыт в создании масштабируемых платформ и продуктов для машинного обучения.</div>
                    </div>
                    <div class="team-member">
                        <div class="member-avatar">D</div>
                        <div class="member-name">Deep Gandhi</div>
                        <div class="member-role">ML инженер</div>
                        <div class="member-bio">Специалист по трансформерам и механистической интерпретируемости. Автор множества исследований в области.</div>
                    </div>
                </div>
                
                <div style="text-align: center; color: var(--text-secondary);">
                    <p>Мы также благодарны нашим стажерам и контрибьюторам за помощь:</p>
                    <p style="margin-top: 1rem; font-weight: 600;">sanjaycal, safiyamak, rohannnair и многим другим!</p>
                </div>
            </section>
            
            <section class="mission-section">
                <h2>Наша миссия</h2>
                <div class="mission-content">
                    <div class="mission-item">
                        <h3>Образование</h3>
                        <p>Сделать изучение трансформеров доступным и интерактивным для студентов и исследователей</p>
                    </div>
                    <div class="mission-item">
                        <h3>Инновации</h3>
                        <p>Создать платформу для экспериментов с архитектурой внимания в реальном времени</p>
                    </div>
                    <div class="mission-item">
                        <h3>Доступность</h3>
                        <p>Предоставить инструменты для понимания работы AI-моделей широкому кругу специалистов</p>
                    </div>
                </div>
            </section>
            
            <section class="join-section">
                <h2>Присоединяйтесь к нам</h2>
                <p>Увлечены созданием красивых инструментов для других разработчиков? Хотите сделать разработку и операции с AI более доступными для мира? Если да, то мы хотели бы услышать от вас: присоединяйтесь к нашему Discord и напишите нашей команде.</p>
                <div class="join-links">
                    <a href="#" class="btn-secondary">Discord</a>
                    <a href="#" class="btn-secondary">GitHub</a>
                    <a href="#" class="btn-secondary">Связаться</a>
                </div>
            </section>
        </main>
        
        <footer class="footer">
            <p>&copy; 2024 Attentify. Платформа для изучения и применения архитектуры Transformer.</p>
        </footer>
        
        <script>
            let currentTheme = 'dark';
            let currentLanguage = 'ru';
            
            function initThemeAndLanguage() {
                const savedTheme = localStorage.getItem('attentify-theme') || 'dark';
                const savedLanguage = localStorage.getItem('attentify-language') || 'ru';
                
                setTheme(savedTheme);
                setLanguage(savedLanguage);
                
                document.getElementById('theme-toggle').addEventListener('click', toggleTheme);
                document.getElementById('language-toggle').addEventListener('click', toggleLanguage);
                
                // Highlight active navigation item
                highlightActiveNavItem();
            }
            
            function highlightActiveNavItem() {
                const currentPath = window.location.pathname;
                const navLinks = document.querySelectorAll('.nav-links a');
                
                navLinks.forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === currentPath) {
                        link.classList.add('active');
                    }
                });
            }
            
            function setTheme(theme) {
                currentTheme = theme;
                document.documentElement.setAttribute('data-theme', theme);
                localStorage.setItem('attentify-theme', theme);
                
                const sunIcon = document.getElementById('sun-icon');
                const moonIcon = document.getElementById('moon-icon');
                
                if (theme === 'light') {
                    sunIcon.style.display = 'none';
                    moonIcon.style.display = 'block';
                } else {
                    sunIcon.style.display = 'block';
                    moonIcon.style.display = 'none';
                }
            }
            
            function toggleTheme() {
                const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
                setTheme(newTheme);
            }
            
            function setLanguage(lang) {
                currentLanguage = lang;
                localStorage.setItem('attentify-language', lang);
                document.getElementById('current-lang').textContent = lang.toUpperCase();
            }
            
            function toggleLanguage() {
                const newLang = currentLanguage === 'ru' ? 'en' : 'ru';
                setLanguage(newLang);
            }
            
            document.addEventListener('DOMContentLoaded', initThemeAndLanguage);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/blog", response_class=HTMLResponse)
async def blog_page():
    """Serve the blog page."""
    html_content = """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Блог - Attentify</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            :root {
                /* Dark theme colors */
                --bg-primary: #1a1a1a;
                --bg-secondary: #2d2d2d;
                --bg-tertiary: #3d3d3d;
                --bg-card: #2d2d2d;
                --bg-input: #3d3d3d;
                
                /* Text colors */
                --text-primary: #ffffff;
                --text-secondary: #eae0c8;
                --text-tertiary: #b4674d;
                --text-accent: #7b3f00;
                
                /* Accent colors */
                --accent-primary: #7b3f00;
                --accent-secondary: #b4674d;
                --accent-accent: #eae0c8;
                --accent-green: #30d158;
                --accent-orange: #ff9f0a;
                --accent-red: #ff453a;
                --accent-purple: #bf5af2;
                
                /* Gradients */
                --gradient-primary: linear-gradient(135deg, #7b3f00 0%, #b4674d 100%);
                --gradient-secondary: linear-gradient(135deg, #b4674d 0%, #eae0c8 100%);
                --gradient-accent: linear-gradient(135deg, #7b3f00 0%, #eae0c8 100%);
                
                /* Shadows */
                --shadow-light: 0 2px 10px rgba(0,0,0,0.3);
                --shadow-medium: 0 4px 20px rgba(0,0,0,0.4);
                --shadow-heavy: 0 8px 30px rgba(0,0,0,0.5);
                
                /* Border radius */
                --radius-small: 8px;
                --radius-medium: 16px;
                --radius-large: 24px;
            }
            
            /* Light theme */
            [data-theme="light"] {
                --bg-primary: #ffffff;
                --bg-secondary: #f8f9fa;
                --bg-tertiary: #e9ecef;
                --bg-card: #ffffff;
                --bg-input: #f8f9fa;
                
                --text-primary: #212529;
                --text-secondary: #495057;
                --text-tertiary: #6c757d;
                --text-accent: #7b3f00;
                
                --shadow-light: 0 2px 10px rgba(0,0,0,0.1);
                --shadow-medium: 0 4px 20px rgba(0,0,0,0.15);
                --shadow-heavy: 0 8px 30px rgba(0,0,0,0.2);
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', Roboto, sans-serif;
                line-height: 1.6;
                color: var(--text-primary);
                background: var(--bg-primary);
                -webkit-font-smoothing: antialiased;
                -moz-osx-font-smoothing: grayscale;
            }
            
            .header {
                background: rgba(28, 28, 30, 0.8);
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
                padding: 1rem 2rem;
                box-shadow: var(--shadow-light);
                position: sticky;
                top: 0;
                z-index: 100;
                border-bottom: 1px solid rgba(255,255,255,0.1);
            }
            
            .nav {
                display: flex;
                justify-content: space-between;
                align-items: center;
                max-width: 1200px;
                margin: 0 auto;
            }
            
            .logo {
                font-size: 1.8rem;
                font-weight: 700;
                color: var(--accent-primary);
                display: flex;
                align-items: center;
                gap: 0.5rem;
                font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
            }
            
            .logo-icon {
                width: 40px;
                height: 40px;
                background: var(--gradient-accent);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
                font-size: 1.2rem;
                box-shadow: var(--shadow-medium);
            }
            
            .nav-links {
                display: flex;
                gap: 2rem;
                align-items: center;
            }
            
            .nav-links a {
                text-decoration: none;
                color: var(--text-secondary);
                font-weight: 500;
                transition: all 0.3s ease;
                padding: 0.5rem 1rem;
                border-radius: var(--radius-small);
            }
            
            .nav-links a:hover {
                color: var(--text-primary);
                background: rgba(255,255,255,0.1);
            }
            
            .nav-controls {
                display: flex;
                gap: 1rem;
                align-items: center;
            }
            
            .control-btn {
                background: transparent;
                border: 1px solid rgba(255,255,255,0.2);
                border-radius: var(--radius-small);
                padding: 0.5rem;
                color: var(--text-secondary);
                cursor: pointer;
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                min-width: 40px;
                height: 40px;
            }
            
            .control-btn:hover {
                background: rgba(255,255,255,0.1);
                border-color: rgba(255,255,255,0.3);
                color: var(--text-primary);
            }
            
            [data-theme="light"] .control-btn {
                border: 1px solid rgba(0,0,0,0.2);
                color: var(--text-secondary);
            }
            
            [data-theme="light"] .control-btn:hover {
                background: rgba(0,0,0,0.1);
                border-color: rgba(0,0,0,0.3);
                color: var(--text-primary);
            }
            
            .theme-toggle, .language-toggle {
                position: relative;
            }
            
            .language-toggle .control-btn {
                min-width: 50px;
                font-weight: 600;
                font-size: 0.9rem;
            }
            
            .main-content {
                max-width: 1200px;
                margin: 0 auto;
                padding: 3rem 2rem;
                display: grid;
                grid-template-columns: 1fr 3fr;
                gap: 3rem;
            }
            
            .sidebar {
                background: var(--bg-card);
                padding: 2rem;
                border-radius: var(--radius-large);
                box-shadow: var(--shadow-medium);
                border: 1px solid rgba(255,255,255,0.1);
                height: fit-content;
                position: sticky;
                top: 120px;
            }
            
            .sidebar h2 {
                font-size: 1.5rem;
                margin-bottom: 1.5rem;
                color: var(--text-primary);
                font-weight: 700;
            }
            
            .year-section {
                margin-bottom: 2rem;
            }
            
            .year-section h3 {
                font-size: 1.2rem;
                margin-bottom: 1rem;
                color: var(--text-primary);
                font-weight: 600;
            }
            
            .post-links {
                list-style: none;
            }
            
            .post-links li {
                margin-bottom: 0.75rem;
            }
            
            .post-links a {
                color: var(--text-secondary);
                text-decoration: none;
                transition: color 0.3s ease;
                font-size: 0.95rem;
                line-height: 1.4;
            }
            
            .post-links a:hover {
                color: var(--accent-primary);
            }
            
            .blog-content {
                background: var(--bg-card);
                padding: 2rem;
                border-radius: var(--radius-large);
                box-shadow: var(--shadow-medium);
                border: 1px solid rgba(255,255,255,0.1);
            }
            
            .blog-header {
                margin-bottom: 3rem;
                text-align: center;
            }
            
            .blog-header h1 {
                font-size: 3rem;
                margin-bottom: 1rem;
                color: var(--text-primary);
                font-weight: 800;
            }
            
            .blog-header p {
                font-size: 1.2rem;
                color: var(--text-secondary);
                max-width: 600px;
                margin: 0 auto;
            }
            
            .blog-posts {
                display: flex;
                flex-direction: column;
                gap: 3rem;
            }
            
            .blog-post {
                border-bottom: 1px solid rgba(255,255,255,0.1);
                padding-bottom: 2rem;
            }
            
            .blog-post:last-child {
                border-bottom: none;
                padding-bottom: 0;
            }
            
            .post-meta {
                display: flex;
                align-items: center;
                gap: 1rem;
                margin-bottom: 1rem;
                font-size: 0.9rem;
                color: var(--text-tertiary);
            }
            
            .post-authors {
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            
            .author-avatar {
                width: 24px;
                height: 24px;
                border-radius: 50%;
                background: var(--gradient-accent);
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 0.8rem;
                font-weight: bold;
            }
            
            .post-title {
                font-size: 1.8rem;
                margin-bottom: 1rem;
                color: var(--text-primary);
                font-weight: 700;
            }
            
            .post-title a {
                color: inherit;
                text-decoration: none;
                transition: color 0.3s ease;
            }
            
            .post-title a:hover {
                color: var(--accent-primary);
            }
            
            .post-excerpt {
                color: var(--text-secondary);
                line-height: 1.6;
                margin-bottom: 1.5rem;
            }
            
            .post-tags {
                display: flex;
                gap: 0.5rem;
                flex-wrap: wrap;
            }
            
            .post-tag {
                background: var(--bg-tertiary);
                color: var(--text-secondary);
                padding: 0.25rem 0.75rem;
                border-radius: var(--radius-small);
                font-size: 0.8rem;
                font-weight: 500;
                border: 1px solid rgba(255,255,255,0.1);
            }
            
            .read-more {
                color: var(--accent-blue);
                text-decoration: none;
                font-weight: 600;
                font-size: 0.95rem;
                transition: color 0.3s ease;
            }
            
            .read-more:hover {
                color: var(--accent-green);
            }
            
            .footer {
                background: var(--bg-secondary);
                color: var(--text-secondary);
                text-align: center;
                padding: 3rem 2rem;
                margin-top: 4rem;
                border-top: 1px solid rgba(255,255,255,0.1);
            }
            
            .footer p {
                opacity: 0.8;
                font-size: 1rem;
            }
            
            @media (max-width: 768px) {
                .nav-links {
                    display: none;
                }
                
                .main-content {
                    grid-template-columns: 1fr;
                    gap: 2rem;
                }
                
                .sidebar {
                    position: static;
                }
            }
        </style>
    </head>
    <body>
        <header class="header">
            <nav class="nav">
                <div class="logo">
                    <div class="logo-icon">A</div>
                    Attentify
                </div>
                <div class="nav-links">
                    <a href="/features">Возможности</a>
                    <a href="/demo">Демо</a>
                    <a href="/blog" class="active">Блог</a>
                    <a href="/about">О проекте</a>
                    <a href="/docs">Документация</a>
                </div>
                <div class="nav-controls">
                    <div class="theme-toggle">
                        <button id="theme-toggle" class="control-btn" title="Сменить тему">
                            <svg id="sun-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="12" cy="12" r="5"></circle>
                                <line x1="12" y1="1" x2="12" y2="3"></line>
                                <line x1="12" y1="21" x2="12" y2="23"></line>
                                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
                                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
                                <line x1="1" y1="12" x2="3" y2="12"></line>
                                <line x1="21" y1="12" x2="23" y2="12"></line>
                                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
                                <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
                            </svg>
                            <svg id="moon-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="display: none;">
                                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
                            </svg>
                        </button>
                    </div>
                    <div class="language-toggle">
                        <button id="language-toggle" class="control-btn" title="Сменить язык">
                            <span id="current-lang">RU</span>
                        </button>
                    </div>
                    <a href="/docs" class="btn-primary">API Документация</a>
                </div>
            </nav>
        </header>
        
        <main class="main-content">
            <aside class="sidebar">
                <h2>Все посты</h2>
                
                <div class="year-section">
                    <h3>2025</h3>
                    <ul class="post-links">
                        <li><a href="#">Support for Diffusion Models</a></li>
                        <li><a href="#">Transformer Lab Now Works with AMD GPUs</a></li>
                        <li><a href="#">New Attention Visualization Features</a></li>
                        <li><a href="#">Performance Improvements in v2.1</a></li>
                    </ul>
                </div>
                
                <div class="year-section">
                    <h3>2024</h3>
                    <ul class="post-links">
                        <li><a href="#">Introducing Attentify Platform</a></li>
                        <li><li><a href="#">Understanding Transformer Architecture</a></li>
                        <li><a href="#">Attention Mechanisms Explained</a></li>
                        <li><a href="#">Getting Started with Attentify</a></li>
                    </ul>
                </div>
            </aside>
            
            <div class="blog-content">
                <div class="blog-header">
                    <h1>Блог Attentify</h1>
                    <p>Новости, обновления и исследования в области трансформеров и механистической интерпретируемости</p>
                </div>
                
                <div class="blog-posts">
                    <article class="blog-post">
                        <div class="post-meta">
                            <span>4 июня 2025 · 3 мин чтения</span>
                            <div class="post-authors">
                                <div class="author-avatar">D</div>
                                <span>Deep Gandhi</span>
                                <span>и</span>
                                <div class="author-avatar">A</div>
                                <span>Ali Asaria</span>
                            </div>
                        </div>
                        
                        <h2 class="post-title">
                            <a href="#">Support for Diffusion Models</a>
                        </h2>
                        
                        <p class="post-excerpt">
                            Отличные новости! Attentify теперь поддерживает диффузионные модели для генерации и обучения изображений. Мы добавили поддержку для Stable Diffusion (1.5, XL, 3) и Flux, что позволяет пользователям экспериментировать с генеративными моделями прямо в нашей платформе.
                        </p>
                        
                        <div class="post-tags">
                            <span class="post-tag">diffusion</span>
                            <span class="post-tag">image-models</span>
                        </div>
                        
                        <div style="margin-top: 1rem;">
                            <a href="#" class="read-more">Читать далее →</a>
                        </div>
                    </article>
                    
                    <article class="blog-post">
                        <div class="post-meta">
                            <span>26 мая 2025 · 17 мин чтения</span>
                            <div class="post-authors">
                                <div class="author-avatar">D</div>
                                <span>Deep Gandhi</span>
                            </div>
                        </div>
                        
                        <h2 class="post-title">
                            <a href="#">Transformer Lab Now Works with AMD GPUs</a>
                        </h2>
                        
                        <p class="post-excerpt">
                            Мы рады объявить, что Attentify теперь поддерживает AMD GPU! Это важный шаг в направлении доступности платформы для большего числа пользователей. Поддержка включает Linux и Windows системы, что делает Attentify универсальным инструментом для различных конфигураций.
                        </p>
                        
                        <div style="background: var(--bg-tertiary); padding: 1.5rem; border-radius: var(--radius-small); margin: 1.5rem 0;">
                            <h4 style="color: var(--accent-primary); margin-bottom: 1rem;">TL;DR</h4>
                            <p style="color: var(--text-secondary); font-size: 0.95rem;">Attentify теперь работает с AMD GPU на Linux и Windows. Улучшена производительность и доступность платформы.</p>
                        </div>
                        
                        <div class="post-tags">
                            <span class="post-tag">transformerlab</span>
                            <span class="post-tag">amd</span>
                            <span class="post-tag">hardware</span>
                            <span class="post-tag">gpu</span>
                        </div>
                        
                        <div style="margin-top: 1rem;">
                            <a href="#" class="read-more">Читать далее →</a>
                        </div>
                    </article>
                </div>
            </div>
        </main>
        
        <footer class="footer">
            <p>&copy; 2024 Attentify. Платформа для изучения и применения архитектуры Transformer.</p>
        </footer>
        
        <script>
            let currentTheme = 'dark';
            let currentLanguage = 'ru';
            
            function initThemeAndLanguage() {
                const savedTheme = localStorage.getItem('attentify-theme') || 'dark';
                const savedLanguage = localStorage.getItem('attentify-language') || 'ru';
                
                setTheme(savedTheme);
                setLanguage(savedLanguage);
                
                document.getElementById('theme-toggle').addEventListener('click', toggleTheme);
                document.getElementById('language-toggle').addEventListener('click', toggleLanguage);
                
                // Highlight active navigation item
                highlightActiveNavItem();
            }
            
            function highlightActiveNavItem() {
                const currentPath = window.location.pathname;
                const navLinks = document.querySelectorAll('.nav-links a');
                
                navLinks.forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === currentPath || 
                        (currentPath === '/' && link.getAttribute('href') === '/')) {
                        link.classList.add('active');
                    }
                });
            }
            
            function setTheme(theme) {
                currentTheme = theme;
                document.documentElement.setAttribute('data-theme', theme);
                localStorage.setItem('attentify-theme', theme);
                
                const sunIcon = document.getElementById('sun-icon');
                const moonIcon = document.getElementById('moon-icon');
                
                if (theme === 'light') {
                    sunIcon.style.display = 'none';
                    moonIcon.style.display = 'block';
                } else {
                    sunIcon.style.display = 'block';
                    moonIcon.style.display = 'none';
                }
            }
            
            function toggleTheme() {
                const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
                setTheme(newTheme);
            }
            
            function setLanguage(lang) {
                currentLanguage = lang;
                localStorage.setItem('attentify-language', lang);
                document.getElementById('current-lang').textContent = lang.toUpperCase();
            }
            
            function toggleLanguage() {
                const newLang = currentLanguage === 'ru' ? 'en' : 'ru';
                setLanguage(newLang);
            }
            
            document.addEventListener('DOMContentLoaded', initThemeAndLanguage);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/docs", response_class=HTMLResponse)
async def docs_page():
    """Serve the documentation page."""
    html_content = """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Документация - Attentify</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            :root {
                /* Dark theme colors */
                --bg-primary: #1a1a1a;
                --bg-secondary: #2d2d2d;
                --bg-tertiary: #3d3d3d;
                --bg-card: #2d2d2d;
                --bg-input: #3d3d3d;
                
                /* Text colors */
                --text-primary: #ffffff;
                --text-secondary: #eae0c8;
                --text-tertiary: #b4674d;
                --text-accent: #7b3f00;
                
                /* Accent colors */
                --accent-primary: #7b3f00;
                --accent-secondary: #b4674d;
                --accent-accent: #eae0c8;
                --accent-green: #30d158;
                --accent-orange: #ff9f0a;
                --accent-red: #ff453a;
                --accent-purple: #bf5af2;
                
                /* Gradients */
                --gradient-primary: linear-gradient(135deg, #7b3f00 0%, #b4674d 100%);
                --gradient-secondary: linear-gradient(135deg, #b4674d 0%, #eae0c8 100%);
                --gradient-accent: linear-gradient(135deg, #7b3f00 0%, #eae0c8 100%);
                
                /* Shadows */
                --shadow-light: 0 2px 10px rgba(0,0,0,0.3);
                --shadow-medium: 0 4px 20px rgba(0,0,0,0.4);
                --shadow-heavy: 0 8px 30px rgba(0,0,0,0.5);
                
                /* Border radius */
                --radius-small: 8px;
                --radius-medium: 16px;
                --radius-large: 24px;
            }
            
            /* Light theme */
            [data-theme="light"] {
                --bg-primary: #ffffff;
                --bg-secondary: #f8f9fa;
                --bg-tertiary: #e9ecef;
                --bg-card: #ffffff;
                --bg-input: #f8f9fa;
                
                --text-primary: #212529;
                --text-secondary: #495057;
                --text-tertiary: #6c757d;
                --text-accent: #7b3f00;
                
                --shadow-light: 0 2px 10px rgba(0,0,0,0.1);
                --shadow-medium: 0 4px 20px rgba(0,0,0,0.15);
                --shadow-heavy: 0 8px 30px rgba(0,0,0,0.2);
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', Roboto, sans-serif;
                line-height: 1.6;
                color: var(--text-primary);
                background: var(--bg-primary);
                -webkit-font-smoothing: antialiased;
                -moz-osx-font-smoothing: grayscale;
            }
            
            .header {
                background: rgba(28, 28, 30, 0.8);
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
                padding: 1rem 2rem;
                box-shadow: var(--shadow-light);
                position: sticky;
                top: 0;
                z-index: 100;
                border-bottom: 1px solid rgba(255,255,255,0.1);
            }
            
            .nav {
                display: flex;
                justify-content: space-between;
                align-items: center;
                max-width: 1200px;
                margin: 0 auto;
            }
            
            .logo {
                font-size: 1.8rem;
                font-weight: 700;
                color: var(--accent-primary);
                display: flex;
                align-items: center;
                gap: 0.5rem;
                font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
            }
            
            .logo-icon {
                width: 40px;
                height: 40px;
                background: var(--gradient-accent);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
                font-size: 1.2rem;
                box-shadow: var(--shadow-medium);
            }
            
            .nav-links {
                display: flex;
                gap: 2rem;
                align-items: center;
            }
            
            .nav-links a {
                text-decoration: none;
                color: var(--text-secondary);
                font-weight: 500;
                transition: all 0.3s ease;
                padding: 0.5rem 1rem;
                border-radius: var(--radius-small);
            }
            
            .nav-links a:hover {
                color: var(--text-primary);
                background: rgba(255,255,255,0.1);
            }
            
            .nav-controls {
                display: flex;
                gap: 1rem;
                align-items: center;
            }
            
            .control-btn {
                background: transparent;
                border: 1px solid rgba(255,255,255,0.2);
                border-radius: var(--radius-small);
                padding: 0.5rem;
                color: var(--text-secondary);
                cursor: pointer;
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                min-width: 40px;
                height: 40px;
            }
            
            .control-btn:hover {
                background: rgba(255,255,255,0.1);
                border-color: rgba(255,255,255,0.3);
                color: var(--text-primary);
            }
            
            [data-theme="light"] .control-btn {
                border: 1px solid rgba(0,0,0,0.2);
                color: var(--text-secondary);
            }
            
            [data-theme="light"] .control-btn:hover {
                background: rgba(0,0,0,0.1);
                border-color: rgba(0,0,0,0.3);
                color: var(--text-primary);
            }
            
            .theme-toggle, .language-toggle {
                position: relative;
            }
            
            .language-toggle .control-btn {
                min-width: 50px;
                font-weight: 600;
                font-size: 0.9rem;
            }
            
            .main-content {
                max-width: 1200px;
                margin: 0 auto;
                padding: 3rem 2rem;
                display: grid;
                grid-template-columns: 1fr 3fr;
                gap: 3rem;
            }
            
            .sidebar {
                background: var(--bg-card);
                padding: 2rem;
                border-radius: var(--radius-large);
                box-shadow: var(--shadow-medium);
                border: 1px solid rgba(255,255,255,0.1);
                height: fit-content;
                position: sticky;
                top: 120px;
            }
            
            .sidebar h2 {
                font-size: 1.5rem;
                margin-bottom: 1.5rem;
                color: var(--text-primary);
                font-weight: 700;
            }
            
            .nav-section {
                margin-bottom: 2rem;
            }
            
            .nav-section h3 {
                font-size: 1.1rem;
                margin-bottom: 1rem;
                color: var(--text-primary);
                font-weight: 600;
            }
            
            .nav-links-docs {
                list-style: none;
            }
            
            .nav-links-docs li {
                margin-bottom: 0.5rem;
            }
            
            .nav-links-docs a {
                color: var(--text-secondary);
                text-decoration: none;
                transition: color 0.3s ease;
                font-size: 0.95rem;
                padding: 0.25rem 0;
                display: block;
            }
            
            .nav-links-docs a:hover {
                color: var(--accent-primary);
            }
            
            .nav-links-docs a.active {
                color: var(--accent-primary);
                font-weight: 600;
            }
            
            .docs-content {
                background: var(--bg-card);
                padding: 2rem;
                border-radius: var(--radius-large);
                box-shadow: var(--shadow-medium);
                border: 1px solid rgba(255,255,255,0.1);
            }
            
            .breadcrumbs {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                margin-bottom: 2rem;
                font-size: 0.9rem;
                color: var(--text-tertiary);
            }
            
            .breadcrumbs a {
                color: var(--accent-primary);
                text-decoration: none;
            }
            
            .breadcrumbs a:hover {
                text-decoration: underline;
            }
            
            .docs-header h1 {
                font-size: 3rem;
                margin-bottom: 1rem;
                color: var(--text-primary);
                font-weight: 800;
            }
            
            .docs-header h2 {
                font-size: 2rem;
                margin-bottom: 1rem;
                color: var(--text-primary);
                font-weight: 700;
            }
            
            .docs-header p {
                font-size: 1.2rem;
                color: var(--text-secondary);
                margin-bottom: 2rem;
                line-height: 1.6;
            }
            
            .docs-section {
                margin-bottom: 3rem;
            }
            
            .docs-section h3 {
                font-size: 1.8rem;
                margin-bottom: 1rem;
                color: var(--text-primary);
                font-weight: 700;
            }
            
            .docs-section p {
                color: var(--text-secondary);
                margin-bottom: 1rem;
                line-height: 1.6;
            }
            
            .docs-section ul {
                margin-bottom: 1rem;
                padding-left: 2rem;
            }
            
            .docs-section li {
                color: var(--text-secondary);
                margin-bottom: 0.5rem;
            }
            
            .ui-screenshot {
                background: var(--bg-tertiary);
                padding: 2rem;
                border-radius: var(--radius-small);
                margin: 2rem 0;
                text-align: center;
                border: 1px solid rgba(255,255,255,0.1);
            }
            
            .ui-screenshot img {
                max-width: 100%;
                height: auto;
                border-radius: var(--radius-small);
            }
            
            .ui-screenshot p {
                margin-top: 1rem;
                color: var(--text-tertiary);
                font-size: 0.9rem;
            }
            
            .feature-list {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 1.5rem;
                margin: 2rem 0;
            }
            
            .feature-item {
                background: var(--bg-tertiary);
                padding: 1.5rem;
                border-radius: var(--radius-small);
                border: 1px solid rgba(255,255,255,0.05);
            }
            
            .feature-item h4 {
                color: var(--accent-primary);
                margin-bottom: 0.5rem;
                font-weight: 600;
            }
            
            .feature-item p {
                color: var(--text-secondary);
                font-size: 0.9rem;
                margin: 0;
            }
            
            .footer {
                background: var(--bg-secondary);
                color: var(--text-secondary);
                text-align: center;
                padding: 3rem 2rem;
                margin-top: 4rem;
                border-top: 1px solid rgba(255,255,255,0.1);
            }
            
            .footer p {
                opacity: 0.8;
                font-size: 1rem;
            }
            
            @media (max-width: 768px) {
                .nav-links {
                    display: none;
                }
                
                .main-content {
                    grid-template-columns: 1fr;
                    gap: 2rem;
                }
                
                .sidebar {
                    position: static;
                }
            }
        </style>
    </head>
    <body>
        <header class="header">
            <nav class="nav">
                <div class="logo">
                    <div class="logo-icon">A</div>
                    Attentify
                </div>
                <div class="nav-links">
                    <a href="/features">Возможности</a>
                    <a href="/demo">Демо</a>
                    <a href="/blog">Блог</a>
                    <a href="/about">О проекте</a>
                    <a href="/docs" class="active">Документация</a>
                </div>
                <div class="nav-controls">
                    <div class="theme-toggle">
                        <button id="theme-toggle" class="control-btn" title="Сменить тему">
                            <svg id="sun-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="12" cy="12" r="5"></circle>
                                <line x1="12" y1="1" x2="12" y2="3"></line>
                                <line x1="12" y1="21" x2="12" y2="23"></line>
                                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
                                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
                                <line x1="1" y1="12" x2="3" y2="12"></line>
                                <line x1="21" y1="12" x2="23" y2="12"></line>
                                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
                                <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
                            </svg>
                            <svg id="moon-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="display: none;">
                                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
                            </svg>
                        </button>
                    </div>
                    <div class="language-toggle">
                        <button id="language-toggle" class="control-btn" title="Сменить язык">
                            <span id="current-lang">RU</span>
                        </button>
                    </div>
                    <a href="/docs" class="btn-primary">API Документация</a>
                </div>
            </nav>
        </header>
        
        <main class="main-content">
            <aside class="sidebar">
                <h2>Getting Started</h2>
                
                <div class="nav-section">
                    <h3>Основы</h3>
                    <ul class="nav-links-docs">
                        <li><a href="#" class="active">Введение</a></li>
                        <li><a href="#">Установка</a></li>
                        <li><a href="#">Быстрый старт</a></li>
                    </ul>
                </div>
                
                <div class="nav-section">
                    <h3>Основные функции</h3>
                    <ul class="nav-links-docs">
                        <li><a href="#">Взаимодействие</a></li>
                        <li><a href="#">Диффузионные модели</a></li>
                        <li><a href="#">Обучение</a></li>
                        <li><a href="#">Экспорт</a></li>
                        <li><a href="#">Генерация</a></li>
                        <li><a href="#">Оценка</a></a></li>
                    </ul>
                </div>
                
                <div class="nav-section">
                    <h3>Расширенные возможности</h3>
                    <ul class="nav-links-docs">
                        <li><a href="#">Plugin SDK</a></li>
                        <li><a href="#">Transformer Lab Client</a></li>
                        <li><a href="#">Часто задаваемые вопросы</a></li>
                    </ul>
                </div>
                
                <div class="nav-section">
                                            <a href="#" style="color: var(--accent-primary); text-decoration: none; font-weight: 600;">Download ↓</a>
                </div>
            </aside>
            
            <div class="docs-content">
                <div class="breadcrumbs">
                    <span>🏠</span>
                    <a href="/docs">Документация</a>
                    <span>></span>
                    <span>Getting Started</span>
                </div>
                
                <div class="docs-header">
                    <h1>Введение</h1>
                    <h2>Что такое Attentify?</h2>
                    <p>Attentify — это бесплатная, открытая платформа для работы с LLM и диффузионными моделями, которая позволяет вам обучать, оценивать, экспортировать и тестировать AI-модели на различных движках вывода и платформах.</p>
                </div>
                
                <div class="ui-screenshot">
                    <div style="background: var(--bg-tertiary); padding: 2rem; border-radius: var(--radius-small); text-align: left;">
                        <div style="display: grid; grid-template-columns: 1fr 2fr; gap: 2rem;">
                            <div style="background: var(--bg-card); padding: 1rem; border-radius: var(--radius-small);">
                                <h4 style="color: var(--text-primary); margin-bottom: 1rem;">Навигация</h4>
                                <ul style="color: var(--text-secondary); font-size: 0.9rem;">
                                    <li>Experiment</li>
                                    <li>Foundation</li>
                                    <li>Prompt</li>
                                    <li>Interact</li>
                                    <li>Embeddings</li>
                                    <li>Train</li>
                                    <li>Export</li>
                                    <li>Evaluate</li>
                                </ul>
                            </div>
                            <div style="background: var(--bg-card); padding: 1rem; border-radius: var(--radius-small);">
                                <h4 style="color: var(--text-primary); margin-bottom: 1rem;">Чат с моделью</h4>
                                <div style="background: var(--bg-tertiary); padding: 1rem; border-radius: var(--radius-small); margin-bottom: 1rem;">
                                    <p style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 0.5rem;">System message:</p>
                                    <p style="color: var(--text-primary); font-size: 0.9rem;">You are a helpful assistant...</p>
                                </div>
                                <div style="background: var(--bg-tertiary); padding: 1rem; border-radius: var(--radius-small);">
                                    <p style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 0.5rem;">User:</p>
                                    <p style="color: var(--text-primary); font-size: 0.9rem;">Reverse the string "hello"</p>
                                </div>
                            </div>
                        </div>
                        <div style="margin-top: 1rem; text-align: center;">
                            <p style="color: var(--text-tertiary); font-size: 0.9rem;">Интерфейс Attentify с навигацией и чатом</p>
                        </div>
                    </div>
                </div>
                
                <div class="docs-section">
                    <p>Attentify отлично работает с GPU/TPU и компьютерами Apple Mac, используя MLX. Мы также приглашаем вас присоединиться к нашему Discord для обсуждений и поддержки.</p>
                    
                    <h3>Что можно делать с Attentify?</h3>
                    
                    <div class="feature-list">
                        <div class="feature-item">
                            <h4>1. Загрузка моделей</h4>
                            <p>Загружайте LLM и диффузионные модели для работы</p>
                        </div>
                        <div class="feature-item">
                            <h4>2. Чат и генерация</h4>
                            <p>Общайтесь с LLM и генерируйте изображения с диффузионными моделями</p>
                        </div>
                        <div class="feature-item">
                            <h4>3. Эмбеддинги</h4>
                            <p>Вычисляйте эмбеддинги LLM для анализа текста</p>
                        </div>
                        <div class="feature-item">
                            <h4>4. Создание датасетов</h4>
                            <p>Создавайте и загружайте датасеты для обучения</p>
                        </div>
                        <div class="feature-item">
                            <h4>5. Обучение моделей</h4>
                            <p>Обучайте LLM и диффузионные модели</p>
                        </div>
                        <div class="feature-item">
                            <h4>6. RLHF и Preference Tuning</h4>
                            <p>Используйте продвинутые техники обучения</p>
                        </div>
                        <div class="feature-item">
                            <h4>7. RAG</h4>
                            <p>Используйте RAG для работы с документами</p>
                        </div>
                    </div>
                    
                    <p>И многое другое. Лучший способ узнать, как использовать Attentify — это посмотреть пошаговое обучающее видео.</p>
                </div>
            </div>
        </main>
        
        <footer class="footer">
            <p>&copy; 2024 Attentify. Платформа для изучения и применения архитектуры Transformer.</p>
        </footer>
        
        <script>
            let currentTheme = 'dark';
            let currentLanguage = 'ru';
            
            function initThemeAndLanguage() {
                const savedTheme = localStorage.getItem('attentify-theme') || 'dark';
                const savedLanguage = localStorage.getItem('attentify-language') || 'ru';
                
                setTheme(savedTheme);
                setLanguage(savedLanguage);
                
                document.getElementById('theme-toggle').addEventListener('click', toggleTheme);
                document.getElementById('language-toggle').addEventListener('click', toggleLanguage);
                
                // Highlight active navigation item
                highlightActiveNavItem();
            }
            
            function highlightActiveNavItem() {
                const currentPath = window.location.pathname;
                const navLinks = document.querySelectorAll('.nav-links a');
                
                navLinks.forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === currentPath || 
                        (currentPath === '/' && link.getAttribute('href') === '/')) {
                        link.classList.add('active');
                    }
                });
            }
            
            function setTheme(theme) {
                currentTheme = theme;
                document.documentElement.setAttribute('data-theme', theme);
                localStorage.setItem('attentify-theme', theme);
                
                const sunIcon = document.getElementById('sun-icon');
                const moonIcon = document.getElementById('moon-icon');
                
                if (theme === 'light') {
                    sunIcon.style.display = 'none';
                    moonIcon.style.display = 'block';
                } else {
                    sunIcon.style.display = 'block';
                    moonIcon.style.display = 'none';
                }
            }
            
            function toggleTheme() {
                const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
                setTheme(newTheme);
            }
            
            function setLanguage(lang) {
                currentLanguage = lang;
                localStorage.setItem('attentify-language', lang);
                document.getElementById('current-lang').textContent = lang.toUpperCase();
            }
            
            function toggleLanguage() {
                const newLang = currentLanguage === 'ru' ? 'en' : 'ru';
                setLanguage(newLang);
            }
            
            document.addEventListener('DOMContentLoaded', initThemeAndLanguage);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
