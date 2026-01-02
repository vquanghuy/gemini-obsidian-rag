# Troubleshooting Guide - Gemini Obsidian RAG

This document captures issues encountered during setup and development, along with solutions and lessons learned.

---

## Issues Encountered

### 1. Expired HuggingFace Token

**Problem:** The initial HF_TOKEN was expired, causing authentication failure when trying to download the gated `google/embeddinggemma-300m` model.

**Error Message:**
```
User Access Token "DL-ML" is expired
Invalid user token
```

**Solution:**
1. Generate a new HuggingFace token at: https://huggingface.co/settings/tokens
2. Accept the model license at: https://huggingface.co/google/embeddinggemma-300m
3. Update the `.env` file with the new token:
   ```bash
   HF_TOKEN=hf_your_new_token_here
   ```
4. Login via CLI:
   ```bash
   huggingface-cli login --token hf_your_new_token_here
   ```

**Lesson Learned:**
- Gated models require active authentication tokens
- Always verify token validity before starting
- Tokens can expire and need periodic renewal
- Keep tokens in `.env` file (never commit to git)

---

### 2. ChromaDB API Change - Missing Exception Class

**Problem:** The code referenced `chromadb.errors.InvalidCollectionException` which doesn't exist in ChromaDB 1.4.0.

**Error Message:**
```
module 'chromadb.errors' has no attribute 'InvalidCollectionException'
```

**Original Code (Broken):**
```python
try:
    # ... search code ...
except chromadb.errors.InvalidCollectionException:
    return {"error": "Index not found"}
```

**Fixed Code:**
```python
try:
    # ... search code ...
except FileNotFoundError:
    return {"error": "Database not found"}
except Exception as e:
    return {"error": f"Search error: {str(e)}"}
```

**Lesson Learned:**
- Library APIs change between versions - exception classes can be added/removed
- Use general exception handling as a fallback
- Check library documentation for current API when upgrading versions
- Don't assume exception classes exist without verification

---

### 3. LlamaIndex Trying to Use OpenAI LLM

**Problem:** When using `index.as_query_engine()`, LlamaIndex defaulted to using OpenAI's API for generating responses, which required an API key we didn't have.

**Error Message:**
```
Could not load OpenAI model...
No API key found for OpenAI
Please set either the OPENAI_API_KEY environment variable or openai.api_key
```

**Root Cause:** 
- `query_engine` performs retrieval + LLM synthesis (requires an LLM)
- We only wanted vector similarity search (retrieval only)

**Original Code (Broken):**
```python
# This requires an LLM for synthesis
query_engine = index.as_query_engine(similarity_top_k=top_k)
response = query_engine.query(query)

# Process response.source_nodes
for node in response.source_nodes:
    # ...
```

**Fixed Code:**
```python
# This only performs retrieval, no LLM needed
retriever = index.as_retriever(similarity_top_k=top_k)
nodes = retriever.retrieve(query)

# Process nodes directly
for node in nodes:
    # ...
```

**Lesson Learned:**
- **`query_engine`** = retrieval + LLM synthesis (requires LLM configuration)
- **`retriever`** = pure vector similarity search (no LLM needed)
- For our use case (providing JSON results to external LLM), retriever is the right choice
- Understanding the distinction between retrieval and generation is crucial in RAG systems
- Read LlamaIndex documentation carefully to choose the right abstraction

---

### 4. Vectors Not Persisting to ChromaDB (Critical Issue)

**Problem:** The indexing appeared to complete successfully (showed progress bars, reported success), but ChromaDB collection had 0 vectors stored.

**Symptoms:**
```bash
# Indexing reported success
Index build complete!
Documents indexed: 130
Database location: /path/to/chroma_db
Collection name: obsidian_notes

# But database was empty
$ python3 -c "import chromadb; client = chromadb.PersistentClient(path='./chroma_db'); print(client.get_collection('obsidian_notes').count())"
0
```

- Indexing script reported success ✓
- Progress bars showed 356 embeddings generated ✓
- But `collection.count()` returned 0 ✗
- Searches returned no results ✗

**Root Cause:** 
LlamaIndex 0.14.12 + ChromaDB 1.4.0 version compatibility issue. When passing `vector_store` directly to `VectorStoreIndex.from_documents()`, the vectors weren't being persisted properly to ChromaDB.

**Original Code (Broken):**
```python
# This doesn't persist vectors properly
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

index = VectorStoreIndex.from_documents(
    documents,
    vector_store=vector_store,  # ❌ Doesn't persist
    show_progress=True
)
```

**Fixed Code:**
```python
# This persists vectors correctly
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Create StorageContext explicitly
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,  # ✅ Persists properly
    show_progress=True
)
```

**Verification:**
```python
# Always verify after indexing
import chromadb
from config import DB_PATH, COLLECTION_NAME

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)
count = collection.count()
print(f"Total vectors stored: {count}")  # Should be > 0
```

**Lesson Learned:**
- **Always verify the end result**, not just intermediate success messages
- Silent failures are the most dangerous - the system reported success but didn't actually work
- When integrating multiple libraries (LlamaIndex + ChromaDB), version compatibility issues can manifest in subtle ways
- Test with a small dataset first to verify the pipeline actually works end-to-end
- Use `StorageContext` explicitly for better control over storage behavior
- Progress bars and console output don't guarantee data persistence
- **Validate with direct database queries** after operations complete

---

## Debugging Techniques Used

### 1. Incremental Testing
Start with simple test cases to isolate problems:

```python
# Test with 2 documents first
test_docs = [
    Document(text="Test about ML", metadata={"source": "test1.md"}),
    Document(text="Test about AI", metadata={"source": "test2.md"})
]
```

### 2. Direct Database Inspection
Check the database directly, don't rely on application logs:

```python
# Verify what's actually in the database
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)
print(f"Actual count: {collection.count()}")
print(f"Sample data: {collection.peek(limit=3)}")
```

### 3. Minimal Reproduction
Create standalone test scripts to test specific functionality:

```python
# test_storage.py - minimal script to test vector storage
import chromadb
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from config import setup_embedding_model

# ... minimal test code ...
```

### 4. Version Checking
Verify installed package versions:

```bash
pip list | grep -E "(llama-index|chroma)"
```

### 5. Documentation Review
Check if API patterns changed in newer versions:
- LlamaIndex docs: https://docs.llamaindex.ai/
- ChromaDB docs: https://docs.trychroma.com/

---

## Best Practices Going Forward

### 1. Always Validate Data Persistence
```python
# After indexing, always verify
collection = client.get_collection(name=COLLECTION_NAME)
count = collection.count()
if count == 0:
    raise RuntimeError("Indexing failed: No vectors stored!")
print(f"✓ Successfully stored {count} vectors")
```

### 2. Use Explicit StorageContext
```python
# Best practice for LlamaIndex + ChromaDB
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
```

### 3. Keep Tokens Up to Date
```bash
# Store in .env (never commit)
HF_TOKEN=hf_your_current_token_here

# Add to .gitignore
echo ".env" >> .gitignore
```

### 4. Test End-to-End with Small Datasets
```python
# Test pipeline with small dataset first
if len(documents) > 100:
    print("Warning: Large dataset detected")
    print("Consider testing with a subset first")
```

### 5. Understand Retrieval vs Generation
```python
# For pure search (no LLM needed):
retriever = index.as_retriever(similarity_top_k=5)
nodes = retriever.retrieve(query)

# For search + synthesis (requires LLM):
query_engine = index.as_query_engine(similarity_top_k=5)
response = query_engine.query(query)
```

### 6. Check Library Compatibility
```bash
# Document working versions in requirements.txt
llama-index==0.14.12
llama-index-core==0.14.12
llama-index-vector-stores-chroma==0.5.5
chromadb==1.4.0
```

### 7. Don't Trust Success Messages Alone
```python
# Bad: Only logging
print("Index build complete!")

# Good: Verify actual result
print("Index build complete!")
actual_count = collection.count()
expected_count = len(chunks)
assert actual_count == expected_count, f"Expected {expected_count}, got {actual_count}"
```

---

## Common Error Messages Reference

| Error Message | Likely Cause | Solution |
|--------------|--------------|----------|
| `User Access Token is expired` | HuggingFace token expired | Generate new token at huggingface.co/settings/tokens |
| `module 'chromadb.errors' has no attribute 'InvalidCollectionException'` | ChromaDB API changed | Use general exception handling |
| `No API key found for OpenAI` | Using query_engine without LLM | Use `as_retriever()` instead of `as_query_engine()` |
| Collection count is 0 after indexing | StorageContext not used | Use explicit `StorageContext.from_defaults()` |
| `huggingface/tokenizers: The current process just got forked` | Tokenizers parallelism warning | Set `TOKENIZERS_PARALLELISM=false` in .env (optional) |

---

## Testing Checklist

Before considering the system working, verify:

- [ ] HuggingFace token is valid and not expired
- [ ] MPS backend is enabled (`torch.backends.mps.is_available() == True`)
- [ ] Environment variables are loaded from `.env` file
- [ ] Vault path exists and contains `.md` files
- [ ] Indexing completes without errors
- [ ] ChromaDB collection count > 0 after indexing
- [ ] Search returns relevant results
- [ ] Search output is valid JSON
- [ ] Similarity scores are reasonable (typically 0.1 - 1.0)
- [ ] Source file paths are correct in results

---

## Quick Debug Commands

```bash
# Check MPS availability
python3 -c "import torch; print('MPS:', torch.backends.mps.is_available())"

# Check ChromaDB collection
python3 -c "import chromadb; c = chromadb.PersistentClient('./chroma_db'); print(c.get_collection('obsidian_notes').count())"

# Test embedding model
python3 -c "from config import setup_embedding_model; m = setup_embedding_model(verbose=True)"

# Test search
python3 search.py "test query" --top-k 3

# Check installed versions
pip list | grep -E "(llama-index|chroma|torch)"
```

---

## Summary

The most critical lesson: **Always validate that data is actually persisted, not just that the operation reported success.** Silent failures with version incompatibilities are common in multi-library integrations, and the only way to catch them is through direct verification of the end result.

When working with RAG systems:
1. Understand the difference between retrieval and generation
2. Use explicit configuration (like `StorageContext`) rather than implicit defaults
3. Test incrementally with small datasets
4. Verify results at each step
5. Keep dependencies documented and version-pinned

This troubleshooting guide should help avoid or quickly resolve similar issues in the future.
