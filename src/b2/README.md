<!--markdownlint-disable-->

# Block B: Semantic Search

## Mini Educational Project - Python with Pipenv

This project contains 4 didactic examples demonstrating **semantic search** concepts, using small and free models from HuggingFace and the ChromaDB vector database.

## Overview

Block B covers 4 lessons about semantic search:

1. Semantic search with embeddings
2. Knowledge graphs
3. Semantic search evaluation
4. Vector databases

## Project Structure

```text
src/b2/
├── exemplos/
│   ├── 01_sistema_busca_semantica.py   # Complete semantic search system
│   ├── 02_grafos_conhecimento.py       # Knowledge graphs and navigation
│   ├── 03_avaliacao_busca.py           # Search evaluation metrics
│   └── 04_banco_vetores_chromadb.py    # Vector databases with ChromaDB
├── dados/
│   ├── base_conhecimento.json          # Knowledge base documents
│   ├── queries_teste.json              # Test queries with expected results
│   └── grafo_paises.json               # Knowledge graph data
├── docs/
│   ├── PLANEJAMENTO.md                 # Detailed project planning
│   └── CHROMA_DB.md                    # ChromaDB persistent storage explained
├── visualizacoes/
│   └── (generated visualization images)
├── chroma_db/
│   └── (persistent ChromaDB storage - see docs/CHROMA_DB.md)
└── README.md
```

## Requirements

- Python 3.11+
- Pipenv package manager
- ~300MB disk space (for models and data)

## Installation

1. Make sure you're in the project root directory
2. Install dependencies with Pipenv:

```bash
pipenv install
```

This will install:

- numpy
- pandas
- matplotlib
- scikit-learn
- torch
- sentence-transformers
- chromadb
- networkx
- jupyter
- notebook
- joblib

## Running the Examples

Each example can be run independently:

```bash
# Activate virtual environment
pipenv shell

# Run examples
python src/b2/exemplos/01_sistema_busca_semantica.py
python src/b2/exemplos/02_grafos_conhecimento.py
python src/b2/exemplos/03_avaliacao_busca.py
python src/b2/exemplos/04_banco_vetores_chromadb.py
```

Or run directly with pipenv:

```bash
pipenv run python src/b2/exemplos/01_sistema_busca_semantica.py
```

## Examples Description

### Example 1: Complete Semantic Search System

**Concepts:**

- Building a knowledge base
- Generating embeddings for documents
- Storing in ChromaDB
- Performing semantic searches
- Comparing with keyword-based search

**Model Used:** `sentence-transformers/all-MiniLM-L6-v2`

- Size: ~80MB
- Dimensions: 384
- Fast and efficient

**Key Takeaways:**

- Semantic search finds documents by meaning, not keywords
- Handles synonyms and related concepts automatically
- ChromaDB provides efficient storage and retrieval

**Reference:** Lesson 1 of Block B

### Example 2: Knowledge Graphs

**Concepts:**

- Building knowledge graphs manually
- Representing semantic relationships
- Graph navigation for queries
- Hybrid approach (graphs + embeddings)
- Graph visualization with NetworkX

**Key Takeaways:**

- Knowledge graphs model explicit relationships
- Foundation of semantic search before modern AI
- Still useful in hybrid systems
- Provides explainable results

**Visualizations:**

- Knowledge graph with countries, cities, and monuments
- Highlighted entities and relationships

**Reference:** Lesson 2 of Block B

### Example 3: Semantic Search Evaluation

**Concepts:**

- Test sets with ground truth
- Precision@K metric
- Recall@K metric
- Mean Reciprocal Rank (MRR)
- Online evaluation (A/B testing concept)

**Key Takeaways:**

- Objective metrics to measure search quality
- Multiple metrics capture different aspects
- Offline evaluation for development
- Online evaluation for production validation

**Visualizations:**

- Precision@K and Recall@K curves
- Performance comparison charts

**Reference:** Lesson 3 of Block B

### Example 4: Vector Databases - ChromaDB

**Concepts:**

- Why specialized vector databases are needed
- HNSW (Hierarchical Navigable Small World) algorithm
- Accuracy vs speed trade-off
- Persistent storage
- Metadata filtering

**Key Takeaways:**

- Vector databases enable search at scale
- HNSW provides logarithmic complexity
- Trade-off: 95-99% accuracy for massive speedup
- Essential for production systems

**Visualizations:**

- Speed benchmark: Naive vs ChromaDB
- Time distribution comparison

**Reference:** Lesson 4 of Block B

## Model Information

### all-MiniLM-L6-v2

- **Type:** Sentence Transformer
- **Size:** ~80MB (very lightweight!)
- **Dimensions:** 384
- **Training:** 1B+ sentence pairs
- **Speed:** Very fast (CPU-friendly)
- **Use case:** General-purpose sentence embeddings
- **License:** Apache 2.0
- **HuggingFace:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

### Why This Model?

1. Small size (works on any machine)
2. Fast inference (no GPU needed)
3. High quality embeddings
4. Free and open source
5. Well-documented and widely used

## Vector Database: ChromaDB

### Why ChromaDB?

- **Open Source:** Free and MIT licensed
- **Embedded:** Runs in-process, no server needed
- **Persistent:** Save and reload collections
- **Feature-Rich:** Metadata filtering, CRUD operations
- **Python-First:** Designed for Python developers
- **Production-Ready:** Used in real applications

### Key Features

1. **HNSW Algorithm:** Fast approximate nearest neighbor search
2. **Metadata Filtering:** Combine semantic + structured search
3. **Persistence:** Save to disk and reload
4. **Easy API:** Intuitive Python interface
5. **Scalable:** Handles millions of vectors

## Learning Path

Recommended order to run the examples:

1. **Example 1** - Build a complete semantic search system
2. **Example 2** - Understand knowledge graphs
3. **Example 3** - Learn to evaluate search quality
4. **Example 4** - Master vector databases

## Key Concepts Summary

### Semantic Search

- Search by meaning, not keywords
- Uses embeddings to represent documents and queries
- Finds semantically similar content
- Essential for modern information retrieval

### Three Steps to Build Semantic Search

1. **Select Documents:** Build a knowledge base
2. **Generate Embeddings:** Use pre-trained models
3. **Store Efficiently:** Use vector database (ChromaDB)

### Search Process

1. **Transform Query:** Generate query embedding
2. **Compare:** Find similar document embeddings
3. **Return Results:** Rank by similarity

### Knowledge Graphs

- Model explicit relationships between entities
- Complement embeddings with structure
- Provide explainable results
- Useful for complex queries

### Evaluation Metrics

| Metric | Measures | Best For |
|--------|----------|----------|
| Precision@K | How many retrieved are relevant? | Quality-focused systems |
| Recall@K | How many relevant are retrieved? | Comprehensive systems |
| MRR | How early is first relevant result? | Search engines |

### Vector Database Benefits

- **Speed:** O(log n) vs O(n) complexity
- **Scale:** Handle billions of vectors
- **Features:** Filtering, CRUD, persistence
- **Accuracy:** 95-99% recall (configurable)

## Applications

- **Document Retrieval:** Find relevant documents by meaning
- **Question Answering:** RAG (Retrieval Augmented Generation)
- **Recommendation Systems:** Content-based recommendations
- **Customer Support:** Automated FAQ matching
- **E-commerce:** Product search and recommendations
- **Research:** Scientific paper discovery

## Troubleshooting

### ChromaDB Issues

If you encounter ChromaDB errors:

```bash
# Reinstall ChromaDB
pipenv uninstall chromadb
pipenv install chromadb
```

### Model Download Issues

Models are cached in:

- Windows: `C:\Users\<user>\.cache\huggingface\hub\`
- Linux/Mac: `~/.cache/huggingface/hub/`

You can manually download from HuggingFace if needed.

### Memory Issues

The examples use ~300MB RAM. If you have memory constraints:

- Close other applications
- Use a smaller model
- Process documents in batches

### Import Errors

Make sure you activated the virtual environment:

```bash
pipenv shell
```

## Data Files

### base_conhecimento.json

- 20 documents about countries, cities, and culture
- Categories: geography, climate, culture, sports, history
- Rich metadata (tags, categories)
- Portuguese content

### queries_teste.json

- 20 test queries with expected results
- Ground truth for evaluation
- Diverse query types
- Covers all categories

### grafo_paises.json

- Entities: countries, cities, monuments, museums
- Relationships: capital_of, located_in, neighbor_of
- Properties: population, area, year built
- Suitable for graph navigation

## ChromaDB Persistent Storage

### What is chroma_db/?

The `chroma_db/` directory is automatically created when you run **Example 4** (Vector Databases). It contains persistent storage for ChromaDB collections.

**Key Points:**

- Contains vector indexes and metadata
- Enables instant loading without re-indexing
- Production-ready persistent storage
- Should be added to `.gitignore` (binary files)
- Safe to delete (can be regenerated)

### Benefits

- ✅ No re-indexing on restart
- ✅ Fast application startup
- ✅ Data survives program crashes
- ✅ Deploy with pre-built indexes

### Usage

```python
from chromadb import PersistentClient

# Create persistent client
client = PersistentClient(path="src/b2/chroma_db")

# Get existing collection (loads from disk)
collection = client.get_collection("my_collection")

# Query immediately - no indexing needed!
results = collection.query(query_texts=["search"], n_results=5)
```

### Should You Version Control It?

**No** - Add to `.gitignore`:

```gitignore
# ChromaDB persistent storage
src/b2/chroma_db/
*.sqlite3
*.bin
```

**Why?**
- Large binary files (100MB+)
- Changes frequently
- Can be regenerated from source data

### More Information

See detailed documentation: [`docs/CHROMA_DB.md`](docs/CHROMA_DB.md)

This document covers:
- Directory structure and contents
- How to backup and restore
- Performance characteristics
- Troubleshooting common issues
- Advanced configuration options

## References

### Block B Materials

- Presentations: `.github/docs/content/presentations/b2/`
- Summary: `.github/docs/content/resumes/rb2.md`

### External Resources

- ChromaDB Docs: <https://docs.trychroma.com/>
- Sentence Transformers: <https://www.sbert.net/>
- HNSW Algorithm: <https://arxiv.org/abs/1603.09320>
- NetworkX: <https://networkx.org/>

## License

Educational project for learning purposes.

## Contributing

This is a teaching project. Feel free to:

- Suggest improvements
- Add more examples
- Report issues
- Share with others learning about semantic search

## Next Steps

After completing Block B, explore:

- **Block C:** Advanced classification techniques
- Fine-tuning embedding models
- Building production RAG systems
- Multimodal search (text + images)
- Hybrid search strategies

## Contact

For questions or suggestions about this project, refer to the course materials or instructor.

---

**Happy Learning!**
