# Agent Guidelines for Gemini Obsidian RAG

This document provides guidelines for AI coding agents working in this repository.

## Project Overview

A local, privacy-first semantic search tool for Obsidian vaults using RAG (Retrieval-Augmented Generation). The tool indexes markdown notes, converts them to embeddings using a local model, stores them in ChromaDB, and provides semantic search via a CLI interface with JSON output.

## Technology Stack

- **Language**: Python 3.8+
- **RAG Framework**: LlamaIndex
- **Vector Database**: ChromaDB (embedded mode)
- **Embedding Model**: google/embeddinggemma-300m (local, Apple Silicon optimized)
- **Output Format**: JSON (for LLM consumption)

## Build & Development Commands

### Environment Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Build initial index (full indexing)
python3 indexer.py --full

# Incremental update (process only changed files)
python3 indexer.py --incremental

# Search the vault
python3 search.py "your query here"
```

### Testing
```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_indexer.py

# Run a single test function
pytest tests/test_indexer.py::test_chunk_creation

# Run with coverage
pytest --cov=. --cov-report=html
```

### Linting & Formatting
```bash
black .              # Format code
isort .              # Sort imports
ruff check .         # Lint
mypy indexer.py      # Type checking
```

## Code Style Guidelines

### File Organization
```python
# Standard library imports
import os
from pathlib import Path
from typing import Dict, List, Optional

# Third-party imports
import chromadb
from llama_index.core import SimpleDirectoryReader

# Local imports (when applicable)
from .utils import load_config
```

### Naming Conventions
- **Files**: `snake_case.py` (e.g., `indexer.py`, `search.py`)
- **Classes**: `PascalCase` (e.g., `VaultIndexer`, `SearchEngine`)
- **Functions/Variables**: `snake_case` (e.g., `build_index`, `chunk_size`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `VAULT_PATH`, `DEFAULT_CHUNK_SIZE`)
- **Private members**: Prefix with `_` (e.g., `_load_embeddings`)

### Type Hints
Always use type hints for function signatures:
```python
def search_vault(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Search the vault and return JSON results."""
    pass
```

### Configuration
- Configuration values (VAULT_PATH, chunk_size, etc.) should be defined as module-level constants
- Document default values in docstrings
- Make parameters configurable via command-line arguments where appropriate

### Error Handling
Always return structured JSON responses, even for errors:
```python
try:
    results = query_database(user_query)
    return {"query": user_query, "results": results, "count": len(results)}
except DatabaseNotFoundError:
    return {
        "query": user_query,
        "error": "Index not found. Please run: python3 indexer.py --full",
        "results": [],
        "count": 0
    }
```

### Docstrings
Use Google-style docstrings:
```python
def chunk_document(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Split document text into overlapping chunks.
    
    Args:
        text: The document text to chunk.
        chunk_size: Maximum tokens per chunk (default: 512).
        overlap: Number of overlapping tokens (default: 50).
    
    Returns:
        List of text chunks with preserved sentence boundaries.
    
    Raises:
        ValueError: If chunk_size or overlap are negative.
    """
    pass
```

### Metadata Storage
When storing documents in ChromaDB, always include:
```python
metadata = {
    "filepath": str(file_path),
    "modified_time": os.path.getmtime(file_path),
    "chunk_index": chunk_idx  # Optional: for tracking chunk position
}
```

### JSON Output Format
All CLI scripts should output valid JSON to stdout:
```python
import json

output = {
    "query": query_string,
    "results": [
        {
            "source": "/path/to/file.md",
            "content": "chunk text...",
            "score": 0.89
        }
    ],
    "count": len(results)
}
print(json.dumps(output, indent=2, ensure_ascii=False))
```

## Architecture Patterns

### Indexer Pattern
- Use LlamaIndex's `SimpleDirectoryReader` for loading markdown files
- Use `SentenceSplitter` for intelligent chunking
- Store metadata alongside vectors for incremental updates
- Support both `--full` and `--incremental` modes

### Search Pattern
- Load ChromaDB in read-only mode
- Embed query using the same model as indexing
- Return top_k results with scores
- Always output JSON (never plain text)

## Privacy & Security
- **Never send data externally**: All embeddings must be generated locally
- **Use local models only**: Default to `google/embeddinggemma-300m`
- **No API keys required**: This is a fully offline tool
- **Respect .gitignore**: Never commit ChromaDB databases, virtual environments, or cache files

## Future Development Notes
See README.md "Future Development" section for planned features including:
- Incremental update implementation (file-level strategy)
- Configurable similarity thresholds
- Metadata filtering
- Multi-vault support

When implementing features, prioritize:
1. **Privacy**: Keep all data local
2. **Simplicity**: CLI-based, no background servers
3. **LLM-agnostic**: JSON output for any LLM to consume
4. **Incremental**: Support efficient updates, not just full re-indexing
