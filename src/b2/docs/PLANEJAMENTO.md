# Block B - Semantic Search: Detailed Planning

## Project Overview

This document describes the detailed planning for the Block B mini project, which covers semantic search concepts through 4 practical examples.

## Educational Objectives

### Primary Goals

1. Understand how to build a complete semantic search system
2. Learn about knowledge graphs and their role in semantic search
3. Master evaluation techniques for search systems
4. Understand vector databases and their importance at scale

### Learning Outcomes

After completing Block B, students will be able to:

- Build end-to-end semantic search systems
- Combine embeddings with knowledge graphs
- Evaluate search quality objectively
- Choose appropriate vector databases for applications
- Understand trade-offs between accuracy and speed

## Lesson Mapping

| Lesson | Topic | Example | Key Concepts |
|--------|-------|---------|--------------|
| Lesson 1 | Semantic search with embeddings | Example 1 | Knowledge base, embeddings, ChromaDB |
| Lesson 2 | Knowledge graphs | Example 2 | Graph structure, navigation, hybrid search |
| Lesson 3 | Search evaluation | Example 3 | Precision, Recall, MRR, A/B testing |
| Lesson 4 | Vector databases | Example 4 | HNSW, scalability, persistence |

## Technical Architecture

### Technology Stack

```text
Python 3.11+
├── Core Libraries
│   ├── sentence-transformers (embeddings)
│   ├── chromadb (vector database)
│   ├── networkx (knowledge graphs)
│   └── numpy (numerical operations)
├── Visualization
│   └── matplotlib (charts and graphs)
└── Data Handling
    └── json (data storage)
```

### Data Flow

```text
Documents (JSON)
    ↓
Sentence Transformer Model (all-MiniLM-L6-v2)
    ↓
Embeddings (384-dim vectors)
    ↓
ChromaDB Collection (HNSW index)
    ↓
Query Embedding
    ↓
Similarity Search
    ↓
Ranked Results
```

## Example Breakdown

### Example 1: Complete Semantic Search System

**File:** `01_sistema_busca_semantica.py`

**Duration:** ~15 minutes to run and understand

**Structure:**

1. **Part 1: Build Knowledge Base**
   - Load 20 documents from JSON
   - Show document statistics by category
   - Display sample documents

2. **Part 2: Generate Embeddings and Index**
   - Load sentence transformer model
   - Generate embeddings for all documents
   - Create ChromaDB collection
   - Index documents with metadata

3. **Part 3: Semantic Search Examples**
   - 5 diverse queries demonstrating semantic understanding
   - Display top-3 results with similarity scores
   - Show how semantic search finds related content

4. **Part 4: Comparison with Keyword Search**
   - Same query on both systems
   - Highlight differences in results
   - Explain why semantic search is better

5. **Part 5: Advanced Queries with Filters**
   - Demonstrate metadata filtering
   - Combine semantic + structured search
   - Show practical use cases

**Expected Output:**

- Console output showing search results
- Comparison between semantic and keyword search
- Understanding of how to build a search system

**Key Teaching Points:**

- Semantic search understands meaning, not just keywords
- ChromaDB makes it easy to build search systems
- Metadata filtering enables hybrid approaches
- Three-step process: select → embed → store

### Example 2: Knowledge Graphs

**File:** `02_grafos_conhecimento.py`

**Duration:** ~15 minutes to run and understand

**Structure:**

1. **Part 1: Load and Explore Knowledge Graph**
   - Load graph data (countries, cities, monuments)
   - Show entity and relationship counts
   - Display entity types

2. **Part 2: Graph Navigation Examples**
   - Find capital of France (reverse relation)
   - Navigate from Torre Eiffel to country (multi-hop)
   - Find all monuments in Paris
   - Show relationship traversal

3. **Part 3: Query System**
   - Simple natural language query system
   - Pattern matching for common questions
   - Demonstrate graph-based answers

4. **Part 4: Hybrid Approach**
   - Combine graph navigation with embeddings
   - Show how both complement each other
   - Semantic matching on graph entities

5. **Part 5: Visualization**
   - Generate graph visualization with NetworkX
   - Highlight specific entities
   - Save to file

**Expected Output:**

- Console output with query answers
- Visual graph showing entities and relationships
- Understanding of knowledge graph strengths/limitations

**Key Teaching Points:**

- Graphs model explicit relationships
- Navigation follows semantic connections
- Hybrid systems combine structure and flexibility
- Explainability is a key advantage

### Example 3: Semantic Search Evaluation

**File:** `03_avaliacao_busca.py`

**Duration:** ~10 minutes to run and understand

**Structure:**

1. **Part 1: Load Test Data**
   - Load queries with expected results
   - Show example test query
   - Explain ground truth concept

2. **Part 2: Create Search Index**
   - Build ChromaDB collection
   - Index all documents
   - Prepare for evaluation

3. **Part 3: Offline Evaluation Metrics**
   - Calculate Precision@K for K=[1,3,5,10]
   - Calculate Recall@K for K=[1,3,5,10]
   - Calculate Mean Reciprocal Rank (MRR)
   - Display results in table

4. **Part 4: Explain Metrics**
   - Precision: quality of results
   - Recall: completeness of results
   - MRR: ranking quality
   - When to use each metric

5. **Part 5: Individual Query Analysis**
   - Show best performing queries
   - Show worst performing queries
   - Identify patterns and issues

6. **Part 6: Visualization**
   - Plot Precision@K curves
   - Plot Recall@K curves
   - Compare metrics visually

7. **Part 7: Online Evaluation Concept**
   - Explain A/B testing
   - Behavioral metrics (CTR, time to click)
   - Statistical significance
   - Production considerations

**Expected Output:**

- Evaluation metrics table
- Precision and Recall curves
- Per-query analysis
- Understanding of how to measure search quality

**Key Teaching Points:**

- Objective metrics prevent subjective bias
- Multiple metrics capture different aspects
- Test sets are essential for evaluation
- Online and offline evaluation complement each other

### Example 4: Vector Databases - ChromaDB

**File:** `04_banco_vetores_chromadb.py`

**Duration:** ~15 minutes to run and understand

**Structure:**

1. **Part 1: Why Vector Databases?**
   - Explain scale problem (billions of vectors)
   - Show naive approach limitations
   - Introduce specialized algorithms

2. **Part 2: Create Vector Index**
   - Load documents
   - Generate embeddings
   - Create ChromaDB collection with HNSW

3. **Part 3: Speed Benchmark**
   - Implement naive brute-force search
   - Compare with ChromaDB HNSW search
   - Run on 10 test queries
   - Calculate speedup

4. **Part 4: Metadata Filtering**
   - Search with no filters
   - Search with category filter
   - Compare results
   - Show hybrid search capability

5. **Part 5: Data Persistence**
   - Create persistent ChromaDB client
   - Save collection to disk
   - Show how to reload later
   - Explain production use

6. **Part 6: Visualization**
   - Bar chart: average search times
   - Box plot: time distributions
   - Show speedup annotation

7. **Part 7: HNSW Algorithm Explanation**
   - Hierarchical layers concept
   - Greedy search strategy
   - Approximate vs exact search
   - Complexity analysis

**Expected Output:**

- Benchmark results showing speedup
- Visualization of performance comparison
- Understanding of vector database architecture
- Knowledge of HNSW algorithm

**Key Teaching Points:**

- Vector databases are essential at scale
- HNSW trades accuracy for speed (good trade-off!)
- Persistence enables production deployment
- Metadata filtering combines semantic + structured search

## Data Design

### base_conhecimento.json

**Purpose:** Main document collection for semantic search

**Structure:**

```json
{
  "documentos": [
    {
      "id": "doc_001",
      "titulo": "Document title",
      "conteudo": "Document content...",
      "categoria": "category",
      "tags": ["tag1", "tag2"]
    }
  ]
}
```

**Content Strategy:**

- 20 documents total
- 10 categories (geografia, clima, cultura, etc.)
- Mix of topics: countries, cities, monuments, sports
- Portuguese language
- Rich metadata for filtering

**Size:** ~15KB

### queries_teste.json

**Purpose:** Test queries with ground truth for evaluation

**Structure:**

```json
{
  "queries": [
    {
      "id": "q001",
      "query": "Question in natural language",
      "documentos_relevantes": ["doc_001", "doc_002"],
      "descricao": "What this query tests"
    }
  ]
}
```

**Content Strategy:**

- 20 test queries
- Diverse query types (factual, semantic, specific)
- 1-3 relevant documents per query
- Cover all categories
- Test different search aspects

**Size:** ~5KB

### grafo_paises.json

**Purpose:** Knowledge graph data for Example 2

**Structure:**

```json
{
  "entidades": [
    {
      "id": "entity_id",
      "tipo": "entity_type",
      "nome": "Entity Name",
      "propriedades": {...}
    }
  ],
  "relacoes": [
    {
      "origem": "source_id",
      "tipo": "relation_type",
      "destino": "target_id"
    }
  ]
}
```

**Content Strategy:**

- 12 entities (countries, cities, monuments)
- 15 relationships
- Properties: population, area, year, etc.
- Navigable structure

**Size:** ~5KB

## Code Organization

### Example Structure Template

Each example follows this pattern:

```python
"""
Example N: Title
================

Concepts covered:
- Concept 1
- Concept 2

Reference: Lesson X of Block B

Description...
"""

import libraries


def helper_function_1():
    """Helper function with clear docstring."""
    pass


def helper_function_2():
    """Another helper function."""
    pass


def main():
    print("=" * 70)
    print("EXAMPLE N: TITLE")
    print("=" * 70)
    print()

    # Part 1: ...
    print("PART 1: Title")
    print("-" * 70)
    print()
    # Implementation...

    # Part 2: ...
    print("PART 2: Title")
    print("-" * 70)
    print()
    # Implementation...

    # Conclusion
    print("=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print()
    # Summary...


if __name__ == "__main__":
    main()
```

### Code Style Guidelines

1. **Clear documentation:** Every function has docstring
2. **Descriptive names:** Variables and functions explain themselves
3. **Consistent formatting:** Following PEP 8
4. **Educational comments:** Explain WHY, not just WHAT
5. **Visual separators:** Use print statements to organize output
6. **Progressive complexity:** Start simple, add features gradually

## Testing Strategy

### Manual Testing Checklist

For each example:

- [x] Runs without errors
- [x] Output is clear and well-formatted
- [x] Concepts are demonstrated effectively
- [x] Visualizations are generated correctly
- [x] Files are saved in correct locations
- [x] Execution time is reasonable (~5-15 minutes)

### Expected Behavior

1. **Example 1:** Creates ChromaDB collection, performs searches, shows comparisons
2. **Example 2:** Loads graph, navigates relationships, creates visualization
3. **Example 3:** Calculates metrics, analyzes queries, generates charts
4. **Example 4:** Benchmarks speed, demonstrates persistence, visualizes results

## Extension Opportunities

### For Advanced Students

1. **Add More Documents:** Expand knowledge base to 100+ documents
2. **Multilingual Search:** Test with documents in multiple languages
3. **Custom Embeddings:** Try different sentence transformer models
4. **Reranking:** Implement a two-stage retrieval system
5. **Query Expansion:** Generate variations of user queries
6. **Caching:** Add result caching for repeated queries

### Integration Projects

1. **Web Interface:** Build Flask/Streamlit app for search
2. **Chat Integration:** Add to chatbot for FAQ answering
3. **Real Data:** Index real documents (PDFs, web pages)
4. **Production Deployment:** Deploy with persistent storage
5. **Monitoring:** Add logging and performance tracking

## Common Issues and Solutions

### Issue 1: Model Download Slow

**Solution:** Download once, cache will be reused

```python
# Models are cached in:
# Windows: C:\Users\<user>\.cache\huggingface\
# Linux/Mac: ~/.cache/huggingface/
```

### Issue 2: ChromaDB Initialization Error

**Solution:** Reset collection or use different name

```python
client.delete_collection(name="collection_name")
```

### Issue 3: Out of Memory

**Solution:** Process documents in batches

```python
# Instead of:
embeddings = model.encode(all_docs)

# Use:
embeddings = []
for batch in batches(docs, batch_size=10):
    embeddings.append(model.encode(batch))
```

### Issue 4: Graph Visualization Not Showing

**Solution:** Use plt.show() and check matplotlib backend

```python
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
```

## Success Criteria

Students successfully complete Block B when they can:

1. **Build:** Create a semantic search system from scratch
2. **Evaluate:** Measure search quality using appropriate metrics
3. **Explain:** Describe how HNSW works and why it's needed
4. **Apply:** Choose between different search approaches
5. **Integrate:** Combine semantic search with structured data

## Timeline

**Estimated Time per Example:**

- Example 1: 30-45 minutes (code + understanding)
- Example 2: 30-45 minutes (code + understanding)
- Example 3: 20-30 minutes (code + understanding)
- Example 4: 30-45 minutes (code + understanding)

**Total Block B:** 2-3 hours

## Resources

### Documentation

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [NetworkX Tutorial](https://networkx.org/documentation/stable/tutorial.html)

### Papers

- [HNSW Paper](https://arxiv.org/abs/1603.09320)
- [Sentence-BERT](https://arxiv.org/abs/1908.10084)

### Videos

- CMU Database Group on Vector Indexes
- Two Minute Papers on Similarity Search

## Conclusion

This mini project provides hands-on experience with semantic search, from basic concepts to production-ready techniques. By combining theory with practical examples, students gain both understanding and implementation skills.

The progression from simple semantic search (Example 1) to advanced vector databases (Example 4) mirrors real-world development of search systems.

---

**Document Version:** 1.0

**Last Updated:** 2025-11-03

**Maintained By:** Course Instructor
