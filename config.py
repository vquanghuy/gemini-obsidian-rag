"""Shared configuration and utilities for Gemini Obsidian RAG.

This module provides centralized configuration management and shared
functions for both indexer.py and search.py.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import torch

# Load environment variables from .env file
load_dotenv()

# Configuration constants
VAULT_PATH = os.getenv("VAULT_PATH")
DB_PATH = os.getenv("DB_PATH", "./chroma_db")
HF_TOKEN = os.getenv("HF_TOKEN")  # Auto-loaded by load_dotenv
COLLECTION_NAME = "obsidian_notes"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "google/embeddinggemma-300m"
DEFAULT_TOP_K = 5


def validate_configuration() -> None:
    """Validate required environment variables are set.

    Raises:
        SystemExit: If required configuration is missing.
    """
    errors = []

    if not VAULT_PATH:
        errors.append("VAULT_PATH is not set in .env file")
    elif not Path(VAULT_PATH).exists():
        errors.append(f"VAULT_PATH does not exist: {VAULT_PATH}")
    elif not Path(VAULT_PATH).is_dir():
        errors.append(f"VAULT_PATH is not a directory: {VAULT_PATH}")

    if not HF_TOKEN:
        errors.append("HF_TOKEN is not set in .env file")
        errors.append("Get your token from: https://huggingface.co/settings/tokens")

    if errors:
        print("Configuration Error:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        print("\nPlease create a .env file based on .env.example", file=sys.stderr)
        sys.exit(1)


def get_device() -> str:
    """Detect and return appropriate device for Apple Silicon or CPU.

    Returns:
        Device string: "mps" for Apple Silicon, "cpu" otherwise.
    """
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def setup_embedding_model(verbose: bool = False) -> HuggingFaceEmbedding:
    """Initialize and return HuggingFaceEmbedding model.

    Args:
        verbose: If True, print device information.

    Returns:
        Configured HuggingFaceEmbedding instance.
    """
    device = get_device()

    if verbose:
        print(f"Using device: {device}", file=sys.stderr)
        print(f"Loading embedding model: {EMBEDDING_MODEL}", file=sys.stderr)

    # HF_TOKEN is automatically read from environment by HuggingFace libraries
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL, device=device)

    if verbose:
        print("Embedding model loaded successfully", file=sys.stderr)

    return embed_model
