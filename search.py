"""Search script for Gemini Obsidian RAG.

This script performs semantic search on the indexed Obsidian vault using
ChromaDB and returns results in JSON format for LLM consumption.

Usage:
    python3 search.py "your search query"
    python3 search.py "your search query" --top-k 10
"""

import sys
import json
import argparse
from typing import Dict, Any, List

from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from config import (
    DB_PATH,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    validate_configuration,
    setup_embedding_model,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Search Obsidian vault using semantic search"
    )
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of results to return (default: {DEFAULT_TOP_K})",
    )

    return parser.parse_args()


def load_vector_store(db_path: str) -> ChromaVectorStore:
    """Load existing ChromaDB vector store.

    Args:
        db_path: Path to ChromaDB database directory.

    Returns:
        ChromaVectorStore instance.

    Raises:
        Exception: If database or collection not found.
    """
    # Create persistent ChromaDB client
    chroma_client = chromadb.PersistentClient(path=db_path)

    # Get existing collection (will raise if not found)
    chroma_collection = chroma_client.get_collection(name=COLLECTION_NAME)

    # Create vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    return vector_store


def search_vault(query: str, top_k: int) -> Dict[str, Any]:
    """Perform semantic search and return results as dict.

    Args:
        query: Search query string.
        top_k: Number of top results to return.

    Returns:
        Dictionary containing query, results, and count.
    """
    try:
        # Step 1: Setup embedding model
        embed_model = setup_embedding_model(verbose=False)

        # Step 2: Configure Settings
        Settings.embed_model = embed_model
        Settings.chunk_size = CHUNK_SIZE
        Settings.chunk_overlap = CHUNK_OVERLAP

        # Step 3: Load vector store
        vector_store = load_vector_store(DB_PATH)

        # Step 4: Create index from existing vector store
        index = VectorStoreIndex.from_vector_store(vector_store)

        # Step 5: Create query engine
        query_engine = index.as_query_engine(similarity_top_k=top_k)

        # Step 6: Execute query
        response = query_engine.query(query)

        # Step 7: Format results
        results = []
        if hasattr(response, "source_nodes"):
            for node in response.source_nodes:
                result_item = {
                    "content": node.node.get_content(),
                    "score": float(node.score) if node.score is not None else 0.0,
                }

                # Add source filepath if available
                if hasattr(node.node, "metadata") and "file_path" in node.node.metadata:
                    result_item["source"] = node.node.metadata["file_path"]
                elif (
                    hasattr(node.node, "metadata") and "file_name" in node.node.metadata
                ):
                    result_item["source"] = node.node.metadata["file_name"]
                else:
                    result_item["source"] = "unknown"

                results.append(result_item)

        return {"query": query, "results": results, "count": len(results)}

    except chromadb.errors.InvalidCollectionException:
        return {
            "query": query,
            "error": "Index not found. Please run: python3 indexer.py --full",
            "results": [],
            "count": 0,
        }
    except FileNotFoundError:
        return {
            "query": query,
            "error": "Database not found. Please run: python3 indexer.py --full",
            "results": [],
            "count": 0,
        }
    except Exception as e:
        return {
            "query": query,
            "error": f"Search error: {str(e)}",
            "results": [],
            "count": 0,
        }


def main() -> None:
    """Entry point - always output JSON to stdout."""
    try:
        # Validate configuration (will exit if invalid)
        validate_configuration()

        # Parse arguments
        args = parse_arguments()

        # Perform search
        result = search_vault(args.query, args.top_k)

        # Output JSON to stdout
        print(json.dumps(result, indent=2, ensure_ascii=False))

    except KeyboardInterrupt:
        # Even for interrupts, output valid JSON
        result = {
            "query": "",
            "error": "Search interrupted by user",
            "results": [],
            "count": 0,
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        sys.exit(1)
    except Exception as e:
        # Catch-all: output error as JSON
        result = {
            "query": "",
            "error": f"Unexpected error: {str(e)}",
            "results": [],
            "count": 0,
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
