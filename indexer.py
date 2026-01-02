"""Indexer for Gemini Obsidian RAG.

This script builds a vector database from an Obsidian vault using LlamaIndex
and ChromaDB. It processes all markdown files, generates embeddings using a
local model, and stores them for semantic search.

Usage:
    python3 indexer.py --full
"""

import sys
import os
import argparse
from pathlib import Path
from typing import List, Tuple

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    Document,
    StorageContext,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from config import (
    VAULT_PATH,
    DB_PATH,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    validate_configuration,
    setup_embedding_model,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Build vector index from Obsidian vault"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Perform full index rebuild (required for now; incremental mode coming soon)",
    )

    args = parser.parse_args()

    # Validate that --full is provided
    if not args.full:
        parser.error("--full flag is required (incremental mode not yet implemented)")

    return args


def setup_vector_store(
    db_path: str, reset: bool = False
) -> Tuple[chromadb.ClientAPI, ChromaVectorStore]:
    """Setup ChromaDB vector store.

    Args:
        db_path: Path to ChromaDB database directory.
        reset: If True, delete existing collection and create new one.

    Returns:
        Tuple of (ChromaDB client, ChromaVectorStore instance).
    """
    # Create persistent ChromaDB client
    chroma_client = chromadb.PersistentClient(path=db_path)

    # If reset, delete existing collection
    if reset:
        try:
            chroma_client.delete_collection(name=COLLECTION_NAME)
            print(f"Deleted existing collection: {COLLECTION_NAME}", file=sys.stderr)
        except Exception:
            # Collection doesn't exist, which is fine
            pass

    # Get or create collection
    chroma_collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    # Create vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    return chroma_client, vector_store


def load_documents(vault_path: str) -> List[Document]:
    """Load documents from Obsidian vault with progress bar.

    Args:
        vault_path: Path to Obsidian vault directory.

    Returns:
        List of Document objects.
    """
    print(f"Loading documents from: {vault_path}", file=sys.stderr)

    # Create reader with filters for markdown files
    reader = SimpleDirectoryReader(
        input_dir=vault_path,
        recursive=True,
        required_exts=[".md"],
        exclude=["**/.obsidian/**"],  # Exclude Obsidian config directory
        filename_as_id=True,
    )

    # Load documents with built-in progress bar
    documents = reader.load_data(show_progress=True)

    # Add metadata (filepath and modification time)
    for doc in documents:
        if hasattr(doc, "metadata") and "file_path" in doc.metadata:
            file_path = doc.metadata["file_path"]
            if os.path.exists(file_path):
                doc.metadata["modified_time"] = os.path.getmtime(file_path)

    print(f"Loaded {len(documents)} documents", file=sys.stderr)
    return documents


def build_index_full(vault_path: str, db_path: str) -> None:
    """Build full index with progress tracking.

    Args:
        vault_path: Path to Obsidian vault.
        db_path: Path to ChromaDB database.
    """
    print("Starting full index build...", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Step 1: Validate configuration
    validate_configuration()

    # Step 2: Setup embedding model
    embed_model = setup_embedding_model(verbose=True)

    # Step 3: Configure LlamaIndex settings
    Settings.embed_model = embed_model
    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP

    print(f"Chunk size: {CHUNK_SIZE} tokens", file=sys.stderr)
    print(f"Chunk overlap: {CHUNK_OVERLAP} tokens", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Step 4: Setup vector store (reset existing data)
    chroma_client, vector_store = setup_vector_store(db_path, reset=True)

    # Step 5: Load documents (with progress bar)
    documents = load_documents(vault_path)

    if not documents:
        print("No documents found in vault!", file=sys.stderr)
        sys.exit(1)

    # Step 6: Build index (with progress bar for embedding generation)
    print("=" * 60, file=sys.stderr)
    print("Generating embeddings and building index...", file=sys.stderr)

    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,  # Built-in progress bar
    )

    # Step 7: Print summary
    print("=" * 60, file=sys.stderr)
    print("Index build complete!", file=sys.stderr)
    print(f"Documents indexed: {len(documents)}", file=sys.stderr)
    print(f"Database location: {os.path.abspath(db_path)}", file=sys.stderr)
    print(f"Collection name: {COLLECTION_NAME}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)


def main() -> None:
    """Entry point with error handling."""
    try:
        args = parse_arguments()
        build_index_full(VAULT_PATH, DB_PATH)
        sys.exit(0)
    except KeyboardInterrupt:
        print("\nIndexing interrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nError during indexing: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
