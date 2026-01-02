# Gemini Obsidian RAG

**A local, privacy-first semantic search tool for your Obsidian vault**

This tool provides fast, semantic search across your entire Obsidian knowledge base using a local RAG (Retrieval-Augmented Generation) pipeline. Query your notes by meaning, not just keywords, while keeping all data on your machine.

---

## The Problem

Traditional text search (`grep`, `find`) relies on exact keyword matching:
- Searching for "building automation" misses notes about "smart home controller"
- Can't find concepts expressed differently across notes
- No understanding of semantic relationships

Loading entire notes into an LLM context has limitations:
- Context window constraints prevent querying large vaults
- Slow performance with large contexts
- Inefficient API quota usage
- Manual note selection is tedious

---

## The Solution

This tool uses **semantic vector search** to find relevant notes by meaning:

1. **Indexes your vault** - Converts all notes into mathematical representations (embeddings)
2. **Stores locally** - Saves embeddings in a local vector database (ChromaDB)
3. **Retrieves semantically** - Finds relevant content based on conceptual similarity, not just keywords

### Key Features

✅ **Fast**: 1-5 second query time (after initial indexing)  
✅ **Private**: All processing happens locally using `google/embeddinggemma-300m`  
✅ **Smart**: Semantic search understands meaning and context  
✅ **Simple**: Single CLI command interface  
✅ **LLM-agnostic**: Works with any LLM that can execute shell commands  
✅ **Incremental**: Only re-index modified files, not the entire vault

---

## Architecture

```text
LLM (Any)          search.py             Vector DB
  │                       │                     │
  │ 1. Execute command    │                     │
  │ python3 search.py "query"                   │
  ├──────────────────────►│                     │
  │                       │ 2. Embed query      │
  │                       │ 3. Search vectors   │
  │                       ├────────────────────►│
  │                       │ 4. Return chunks    │
  │                       │◄────────────────────┤
  │ 5. Receive JSON       │                     │
  │◄──────────────────────┤                     │
  │ 6. Generate response  │                     │
```

**How it works:**
- LLM executes: `python3 search.py "your query"`
- Script loads the vector database and searches for relevant chunks
- Results are returned as JSON to STDOUT
- LLM parses the JSON and synthesizes an answer

**Trade-offs:**
- ✅ Zero infrastructure - no background servers to manage
- ✅ Stateless - fresh start prevents memory issues
- ✅ Simple debugging - run script directly in terminal
- ⚠️ ~1-2s startup penalty per query (Python imports + DB loading)

---

## Quick Start

### Prerequisites

- Python 3.8+
- Obsidian vault
- ~1GB disk space for the vector database

### Installation

1. **Create a virtual environment:**

```bash
python3 -m venv venv
```

2. **Activate the virtual environment:**

```bash
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Configure your environment:**

```bash
cp .env.example .env
# Edit .env with your vault path and HuggingFace token
```

Get your HuggingFace token from: https://huggingface.co/settings/tokens

### Build the Index

```bash
python3 indexer.py --full
```

This one-time operation:
- Reads all `.md` files from your vault (with progress bar)
- Chunks them into searchable segments
- Generates embeddings using the local model (with progress bar)
- Stores vectors in ChromaDB

**Expected time**: 2-10 minutes for ~1,000 notes on Apple M1 Pro

**Progress indication**: The indexer displays real-time progress bars for both file loading and embedding generation, so you can monitor the indexing process.

### Search Your Vault

```bash
python3 search.py "your search query here"
```

Returns JSON with the most relevant note chunks and metadata.

**Example output:**
```json
{
  "query": "project roadmap",
  "results": [
    {
      "source": "/path/to/vault/project-planning.md",
      "content": "Q2 roadmap includes the automation pipeline...",
      "score": 0.89
    },
    {
      "source": "/path/to/vault/goals-2024.md",
      "content": "Strategic initiatives for the project...",
      "score": 0.82
    }
  ],
  "count": 2
}
```

### Integration with LLMs

Any LLM with shell command execution can use this tool. The JSON output format is designed for easy parsing and integration.

**Example usage:**
```
Execute: python3 search.py "project roadmap"
Then summarize the results.
```

---

## Configuration

### Vault Path

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Then edit `.env` and set your configuration:

```bash
# Path to your Obsidian vault
VAULT_PATH=/path/to/your/obsidian/vault

# Path to ChromaDB vector database
DB_PATH=./chroma_db

# HuggingFace API token (required for gated model)
HF_TOKEN=hf_your_token_here
```

**Note**: The `google/embeddinggemma-300m` model is gated and requires a HuggingFace account. Get your token from: https://huggingface.co/settings/tokens

### Chunk Size & Overlap

Adjust chunking parameters in `config.py`:

```python
CHUNK_SIZE = 512      # tokens per chunk
CHUNK_OVERLAP = 50    # overlap between chunks
```

### Number of Results

Use the `--top-k` flag when searching to control how many chunks are returned:

```bash
python3 search.py "your query" --top-k 10
```

Or modify the default in `config.py`:

```python
DEFAULT_TOP_K = 5  # default number of results
```

### Updating the Index

#### Incremental Updates

**Status:** Not yet implemented (planned for future release)

After adding or modifying notes, incremental indexing will process only changed files:

```bash
python3 indexer.py --incremental  # Coming soon
```

This approach will:
- ✅ Only process new or modified files (based on modification timestamp)
- ✅ Be significantly faster than full re-indexing
- ✅ Keep the database in sync with your vault

#### Full Re-index

To rebuild the entire index from scratch:

```bash
python3 indexer.py --full
```

Use full re-indexing when:
- Running for the first time
- Changing chunk size or overlap settings
- Switching embedding models
- Troubleshooting index corruption

---

## Implementation Details

### Required Files

1. **`requirements.txt`** - Python dependencies
   - `llama-index` - RAG framework
   - `llama-index-embeddings-huggingface` - Local embedding support
   - `llama-index-vector-stores-chroma` - ChromaDB integration
   - `chromadb` - Vector database
   - `python-dotenv` - Environment variable management

2. **`config.py`** - Shared configuration and utilities
   - Loads environment variables from `.env`
   - Validates configuration (vault path, HF token)
   - Detects device (MPS for Apple Silicon, CPU otherwise)
   - Initializes embedding model

3. **`indexer.py`** - Builds the vector database
   - Loads all `.md` files from vault
   - Chunks documents intelligently
   - Generates embeddings with progress tracking
   - Stores in ChromaDB
   - Shows real-time progress bars for user feedback
   - Supports incremental updates via file modification tracking (planned)

4. **`search.py`** - CLI query interface
   - Accepts search query as argument
   - Loads vector database
   - Returns top-k relevant chunks as JSON

### Indexer Pipeline

1. **Parsing & Chunking (LlamaIndex):**
   - Use `SimpleDirectoryReader` to recursively load all `.md` files (with progress bar: `show_progress=True`)
   - Use `SentenceSplitter` to break text into chunks (e.g., 512 tokens) with overlap, preserving sentence boundaries

2. **Embedding (EmbeddingGemma):**
   - Each text chunk is passed to the local embedding model
   - The model converts text into a vector representation
   - Progress bar shows embedding generation status

3. **Storage (ChromaDB):**
   - The vector, raw text, and metadata (filepath, modification time) are saved to the persistent database

**User Feedback:** Built-in progress bars (via LlamaIndex's `show_progress=True` parameter) provide real-time feedback during file loading and embedding generation.

### Technology Stack

| Component         | Technology                      | Why?                                           |
|-------------------|---------------------------------|------------------------------------------------|
| RAG Framework     | LlamaIndex                      | Optimized for RAG workflows, high-level APIs   |
| Vector Database   | ChromaDB (embedded)             | Zero-setup, portable, like SQLite for vectors  |
| Embedding Model   | google/embeddinggemma-300m      | Privacy + Apple Silicon optimization           |

### Embedding Model: EmbeddingGemma

This project uses `google/embeddinggemma-300m` as the embedding model:

- **Privacy-first**: Data never leaves your machine
- **Apple Silicon optimized**: Uses MPS (Metal Performance Shaders) backend
- **Lightweight**: ~600MB RAM usage
- **Fast**: Runs on Mac GPU/Neural Engine
- **Quality**: Comparable to cloud embedding APIs while maintaining full privacy

---

## Performance Expectations

Based on testing with **~1,000 notes** on **Apple M1 Pro**:

| Operation              | Time          | Notes                                          |
|------------------------|---------------|------------------------------------------------|
| Initial Indexing       | 2-10 minutes  | One-time cost, varies with note count          |
| Incremental Update     | TBD           | To be measured after implementation            |
| Query Time             | 1-5 seconds   | Includes Python startup + search + embedding   |
| Vector Search          | <1 second     | The search itself is very fast                 |

---

## Error Handling

The `search.py` script returns JSON in all cases:

**Success:**
```json
{
  "query": "project roadmap",
  "results": [
    {
      "source": "/path/to/vault/note.md",
      "content": "chunk text...",
      "score": 0.89
    }
  ],
  "count": 2
}
```

**Error:**
```json
{
  "query": "invalid query",
  "error": "Index not found. Please run: python3 indexer.py --full",
  "results": [],
  "count": 0
}
```

**No results:**
```json
{
  "query": "nonexistent topic",
  "results": [],
  "count": 0
}
```

---

## References

### Documentation & Tutorials
- [LlamaIndex: A Guide to RAG (Real Python)](https://realpython.com/llamaindex-examples/)
- [Official LlamaIndex GitHub Repository](https://github.com/run-llama/llama_index)
- [Hugging Face EmbeddingGemma Model Card](https://huggingface.co/google/embeddinggemma-300m)

### Research & Papers
- Schechter Vera, H., et al. (2025). *EmbeddingGemma: Powerful and Lightweight Text Representations*. Google DeepMind. [arXiv:2509.20354](https://arxiv.org/abs/2509.20354)

---

## Future Development

### 1. Incremental Update Implementation

**Current Status:** Planned but not yet implemented

**Strategy:**
When a file is modified, the indexer will:
1. Query ChromaDB for existing chunks from that file (using metadata)
2. Delete all old chunks from that file
3. Re-chunk and re-index the entire file with updated content
4. Store new chunks with current modification timestamp

**Alternative Approaches:**

| Strategy | Approach | Pros | Cons |
|----------|----------|------|------|
| **File-level** (Planned) | Delete all chunks from modified file, re-index entire file | Simple, reliable, ensures consistency | Re-processes entire file even for small changes |
| **Chunk-level** | Identify which specific chunks changed, update only those | More efficient for large files with small edits | Complex implementation, potential for inconsistency |
| **Hybrid** | File-level for small files, chunk-level for large files | Balanced efficiency | Increased complexity |

**Technical Notes:**
- ChromaDB supports metadata storage natively
- Metadata structure: `{"filepath": "/path/to/note.md", "modified_time": 1234567890}`
- No separate state file needed
- Modification timestamps compared between filesystem and database

**Future Enhancements:**
- Smart change detection using content hash comparison
- Parallel processing for multiple changed files
- Progress reporting for large incremental updates
- Automatic scheduled updates (cron/Task Scheduler integration)

### 2. Configuration Parameter Optimization

#### Number of Results (top_k)
**Current default:** `top_k = 5`

**Optimization considerations:**
- **Smaller values (3-5):** Faster, more focused results, good for targeted queries
- **Larger values (10-15):** More comprehensive context, better for exploratory queries
- **Trade-off:** LLM context window size vs. recall completeness

**Tuning guidance:** 
- Start with 5 and adjust based on your LLM's context window
- Increase for complex queries requiring more context
- Decrease for faster responses and focused results

#### Similarity Score Threshold
**Current implementation:** Returns top_k results regardless of score

**Potential enhancement:** Add minimum similarity threshold
- Add `--min-score` flag to search.py (e.g., `--min-score 0.7`)
- **Benefits:** Filter out irrelevant results, improve precision
- **Trade-offs:** May return fewer than top_k results, could miss edge cases
- **Use case:** When precision is more important than recall

#### Chunk Overlap
**Current default:** `chunk_overlap = 50` tokens (~10% of 512-token chunks)

**Optimization considerations:**
- **Less overlap (0-25 tokens):** Smaller index, faster indexing, risk of splitting concepts
- **More overlap (75-128 tokens):** Better context preservation, larger index
- **Trade-off:** Index size vs. retrieval quality at chunk boundaries

**Recommendation:** 50 tokens is a good starting point. Increase if relevant content is being split across chunks.

### 3. Enhanced Error Handling

**Current approach:** User-friendly error messages with actionable guidance

**Future enhancements:**
- **Verbose mode:** Add `--verbose` flag for detailed error information and stack traces
- **Error codes:** Structured error codes for programmatic handling
- **Logging:** Optional file logging for debugging and performance analysis
- **Validation:** Pre-flight checks before indexing (disk space, permissions, model availability)

### 4. Advanced Features

**Metadata Filtering:**
- Filter search results by tags, creation date, or folder structure
- Example: `python3 search.py "query" --tags "project,work" --folder "2024/"`

**Multi-vault Support:**
- Index multiple Obsidian vaults simultaneously
- Specify vault name in search queries
- Unified search across all vaults

**Custom Chunk Strategies:**
- Per-note-type chunking (different sizes for daily notes vs. reference docs)
- Markdown-aware chunking (preserve heading hierarchy)
- Code block handling (special treatment for embedded code)

**Performance Monitoring:**
- Query performance metrics (embedding time, search time, total time)
- Index statistics (number of chunks, total size, oldest/newest docs)
- Health checks and diagnostics

**Export/Import:**
- Export index for backup or transfer
- Import pre-built indexes
- Merge indexes from multiple sources

---

## Appendix

### A. Why Semantic Search?

Traditional keyword search vs semantic search:

**Keyword Search (`grep`):**
- ✅ Instant (milliseconds)
- ❌ Brittle - exact matches only
- ❌ Misses synonyms and related concepts
- ❌ No understanding of context

**Semantic Search (This Tool):**
- ✅ Fast (~1-5 seconds)
- ✅ Finds concepts by meaning, not just keywords
- ✅ Handles synonyms, paraphrases, related ideas
- ✅ Understands context and relationships

Example: Searching for "building automation" will successfully find notes about "smart home controller", "Home Assistant configuration", and "IoT device management".

### B. Alternative Approaches

**Deep-Scan Agent:**
An agent-based approach that loops through files, follows links, and performs logical analysis.

- ✅ Extremely smart - can solve complex logical queries
- ❌ Slow (minutes) - requires heavy token usage and sequential file reading
- Best for: Complex root-cause analysis, forensic investigation

**Vector Search (This Tool):**
Semantic understanding with near-instant speed.

- ✅ Fast (~1-5 seconds)
- ✅ Smart enough for most use cases
- Best for: Daily knowledge base queries and content discovery

### C. Privacy Considerations

**Cloud Embedding APIs** (OpenAI, Google):
- ✅ Higher quality in some benchmarks
- ✅ Zero local compute required
- ❌ Requires sending all vault content to external servers
- ❌ Privacy concerns for sensitive personal data
- ❌ Requires internet connection
- ❌ API costs and quota limits

**Local Model** (EmbeddingGemma - This Tool):
- ✅ Complete privacy - data never leaves your machine
- ✅ Offline capability
- ✅ No API costs or quota limits
- ✅ Optimized for Apple Silicon (MPS backend)
- ⚠️ Requires local GPU/CPU resources
- ⚠️ Initial model download (~600MB)

For personal vaults containing journals, financial logs, or sensitive project data, the privacy benefits of local embeddings far outweigh minor quality differences.
