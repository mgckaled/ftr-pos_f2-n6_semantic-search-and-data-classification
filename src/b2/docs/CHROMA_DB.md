# ChromaDB - Persistent Storage

## Overview

This document explains the `chroma_db/` directory that is generated when you run Example 4 (Vector Databases) of Block B.

## What is chroma_db?

The `chroma_db/` directory is a **persistent storage location** for ChromaDB collections. It contains the indexed vector embeddings and metadata that allow you to:

- Save vector collections to disk
- Reload collections without re-indexing
- Persist data between program runs
- Deploy in production environments

## Directory Structure

When you run Example 4, the following structure is created:

```text
src/b2/chroma_db/
├── chroma.sqlite3           # SQLite database with metadata
└── [collection_id]/         # Directory for each collection
    ├── data_level0.bin      # HNSW index data
    ├── header.bin           # Index header
    ├── length.bin           # Document lengths
    └── link_lists.bin       # HNSW graph links
```

## What's Inside?

### chroma.sqlite3

A SQLite database that stores:
- Collection metadata (names, IDs, settings)
- Document IDs and metadata
- Configuration settings

### Collection Directories

Each collection has its own directory containing:

1. **HNSW Index Files:**
   - `data_level0.bin`: Vector embeddings at the base layer
   - `link_lists.bin`: Graph connections between vectors
   - `header.bin`: Index configuration and statistics

2. **Document Data:**
   - `length.bin`: Document lengths for normalization
   - Other binary files with document content and metadata

## When is it Created?

The `chroma_db/` directory is created in **Example 4 - Part 5: Data Persistence**:

```python
# Create persistent client
persist_dir = Path(__file__).parent.parent / "chroma_db"
persist_dir.mkdir(exist_ok=True)

client_persistent = chromadb.PersistentClient(
    path=str(persist_dir),
    settings=Settings(anonymized_telemetry=False)
)
```

## How to Use Persistent Storage

### Saving Data

```python
from chromadb import PersistentClient
from pathlib import Path

# Create persistent client
persist_dir = Path("path/to/chroma_db")
client = PersistentClient(path=str(persist_dir))

# Create collection
collection = client.create_collection("my_collection")

# Add documents (automatically saved to disk)
collection.add(
    ids=["doc1", "doc2"],
    documents=["Text 1", "Text 2"],
    metadatas=[{"category": "A"}, {"category": "B"}]
)
```

### Loading Existing Data

```python
from chromadb import PersistentClient

# Connect to existing database
client = PersistentClient(path="path/to/chroma_db")

# Get existing collection (no re-indexing needed!)
collection = client.get_collection("my_collection")

# Query immediately
results = collection.query(
    query_texts=["search query"],
    n_results=5
)
```

## Benefits of Persistence

1. **No Re-indexing:**
   - Load collections instantly
   - Save hours of embedding generation time
   - Resume work from where you left off

2. **Production Ready:**
   - Deploy applications with pre-built indexes
   - Serve queries immediately on startup
   - Update collections incrementally

3. **Data Safety:**
   - Survive program crashes
   - Backup and restore collections
   - Version control indexes (with caution)

## Should You Version Control It?

**Generally NO**, for several reasons:

### ❌ Don't Add to Git

- **Large File Sizes:** Binary files can be 100MB+ for real datasets
- **Binary Format:** Not human-readable, hard to review changes
- **Frequent Changes:** Reindexing creates large diffs
- **Reproducible:** Can be regenerated from source data

### ✅ Add to .gitignore

```gitignore
# ChromaDB persistent storage
src/b2/chroma_db/
*.sqlite3
*.bin
```

### ✅ When to Version Control

Only if:
- Database is small (<10MB)
- Used for testing with fixed data
- Part of a demo that needs to "just work"

**Better Alternatives:**
- Version the source data (JSON, CSV)
- Version the indexing script
- Document how to rebuild the database

## Cleaning Up

If you want to start fresh:

### Delete Entire Database

```bash
# Windows
rmdir /s src\b2\chroma_db

# Linux/Mac
rm -rf src/b2/chroma_db
```

### Delete Specific Collection

```python
client = PersistentClient(path="path/to/chroma_db")
client.delete_collection("collection_name")
```

### Reset Collection

```python
# Delete and recreate
client.delete_collection("my_collection")
collection = client.create_collection("my_collection")
```

## Performance Characteristics

| Operation | In-Memory | Persistent |
|-----------|-----------|------------|
| First Load | Fast (RAM) | Slower (Disk I/O) |
| Query Speed | Very Fast | Fast |
| Startup Time | Must reindex | Instant |
| Memory Usage | High | Lower (disk cache) |
| Data Durability | Lost on exit | Saved |

## Example Use Cases

### Development

```python
# Use in-memory for fast iteration
client = chromadb.Client()
```

### Production

```python
# Use persistent for deployment
client = PersistentClient(path="/var/lib/chroma")
```

### Testing

```python
# Use temporary directory
import tempfile
temp_dir = tempfile.mkdtemp()
client = PersistentClient(path=temp_dir)
# ... run tests ...
shutil.rmtree(temp_dir)  # cleanup
```

## Troubleshooting

### Database Locked Error

**Problem:** `database is locked`

**Solution:**
- Close all connections to the database
- Only one client should access the database at a time
- Use a different directory for parallel processes

### Corrupted Database

**Problem:** Errors loading collections

**Solution:**
1. Delete the `chroma_db/` directory
2. Re-run Example 4 to recreate
3. Or restore from backup

### Out of Disk Space

**Problem:** Large vector databases

**Solution:**
- Monitor disk usage
- Delete unused collections
- Use compression (if available)
- Consider cloud storage for large datasets

## Advanced Configuration

### Custom Storage Path

```python
# Absolute path
client = PersistentClient(path="/data/chroma")

# Relative to current directory
client = PersistentClient(path="./my_chroma_db")

# In user's home directory
from pathlib import Path
home_db = Path.home() / ".my_app" / "chroma_db"
client = PersistentClient(path=str(home_db))
```

### Collection Settings

```python
collection = client.create_collection(
    name="advanced_collection",
    metadata={
        "hnsw:space": "cosine",           # Distance metric
        "hnsw:construction_ef": 200,      # Index build quality
        "hnsw:M": 16                      # Connections per node
    }
)
```

## Migration and Backup

### Backup

```bash
# Simple copy
cp -r src/b2/chroma_db src/b2/chroma_db.backup

# Compressed backup
tar -czf chroma_backup.tar.gz src/b2/chroma_db
```

### Restore

```bash
# From copy
cp -r src/b2/chroma_db.backup src/b2/chroma_db

# From compressed
tar -xzf chroma_backup.tar.gz
```

### Export/Import

```python
# Export to JSON
collection = client.get_collection("my_collection")
data = collection.get(include=["documents", "metadatas", "embeddings"])

import json
with open("export.json", "w") as f:
    json.dump(data, f)

# Import from JSON
with open("export.json", "r") as f:
    data = json.load(f)

new_collection = client.create_collection("imported_collection")
new_collection.add(
    ids=data["ids"],
    documents=data["documents"],
    metadatas=data["metadatas"],
    embeddings=data["embeddings"]
)
```

## References

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Persistent Client Guide](https://docs.trychroma.com/guides/persistent-client)
- [HNSW Algorithm](https://arxiv.org/abs/1603.09320)

## Summary

The `chroma_db/` directory is:

- ✅ **Generated automatically** when using PersistentClient
- ✅ **Contains vector indexes** and metadata
- ✅ **Enables fast restarts** without re-indexing
- ✅ **Should be gitignored** (too large, binary files)
- ✅ **Safe to delete** (can be regenerated)
- ✅ **Production-ready** for deployment

**Key Takeaway:** Think of `chroma_db/` like a database file - it's data storage, not source code, so treat it accordingly!

---

**Last Updated:** 2025-11-03
