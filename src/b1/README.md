# Block A: Embeddings and Semantic Search

## Mini Educational Project - Python with UV

This project contains 5 didactic examples demonstrating fundamental concepts of **embeddings** and **semantic search**, using small and free models from HuggingFace.

## Overview

Block A covers 7 lessons about embeddings and semantic search:

1. Introduction to semantic search and classification
2. Introduction to embeddings
3. Embeddings as vectors
4. Characteristics of embeddings
5. Naive embeddings
6. Generating embeddings with neural networks
7. Embedding similarity

## Project Structure

```text
src/b1/
├── exemplos/
│   ├── 01_embeddings_ingenuos.py      # Naive embeddings (one-hot, bag-of-words)
│   ├── 02_word_embeddings.py          # Neural embeddings with transformers
│   ├── 03_similaridade_embeddings.py  # Similarity metrics comparison
│   ├── 04_busca_semantica.py          # Simple semantic search system
│   └── 05_visualizacao_3d.py          # 3D visualization of embeddings
├── dados/
│   ├── textos_exemplo.json            # Example texts by category
│   └── temperaturas_cidades.csv       # Cities temperature data
├── docs/
│   └── PLANEJAMENTO.md                # Detailed project planning
├── visualizacoes/
│   └── (generated plots)
└── README.md
```

## Requirements

- Python 3.11+
- UV package manager
- ~200MB disk space (for model download)

## Installation

1. Clone the repository
2. Install dependencies with UV:

```bash
uv sync
```

This will install:

- numpy >= 2.3.0
- pandas >= 2.3.0
- matplotlib >= 3.10.0
- scikit-learn >= 1.7.0
- torch >= 2.5.0
- sentence-transformers >= 5.1.0
- jupyter >= 1.1.0

## Running the Examples

Each example can be run independently:

```bash
# Activate virtual environment (Windows)
.venv\Scripts\activate

# Run examples
python src/b1/exemplos/01_embeddings_ingenuos.py
python src/b1/exemplos/02_word_embeddings.py
python src/b1/exemplos/03_similaridade_embeddings.py
python src/b1/exemplos/04_busca_semantica.py
python src/b1/exemplos/05_visualizacao_3d.py
```

Or use the virtual environment Python directly:

```bash
.venv\Scripts\python.exe src/b1/exemplos/01_embeddings_ingenuos.py
```

## Examples Description

### Example 1: Naive Embeddings

**Concepts:**

- One-hot encoding
- Bag-of-words
- Limitations of naive approaches

**Key Takeaways:**

- One-hot vectors are sparse and have no semantics
- All words are equally distant
- Doesn't capture meaning or relationships

**Reference:** Lesson 5 of Block A

### Example 2: Word Embeddings with Neural Networks

**Concepts:**

- Pre-trained transformer models
- Dense representations
- Contextual embeddings

**Model Used:** `sentence-transformers/all-MiniLM-L6-v2`

- Size: ~80MB
- Dimensions: 384
- Fast and efficient

**Key Takeaways:**

- Dense vectors capture semantics
- Similar texts have close embeddings
- 26x smaller than one-hot!

**Reference:** Lesson 6 of Block A

### Example 3: Embedding Similarity Metrics

**Concepts:**

- Euclidean distance (L2)
- Cosine similarity
- Dot product
- When to use each metric

**Key Takeaways:**

- Cosine similarity is most common for NLP
- For normalized embeddings, cosine = dot product
- Different metrics for different use cases

**Visualizations:** Generates 2D PCA projection

**Reference:** Lesson 7 of Block A

### Example 4: Simple Semantic Search System

**Concepts:**

- Document indexing
- Query encoding
- Similarity ranking
- Top-K retrieval

**Key Takeaways:**

- Finds documents by meaning, not keywords
- Handles synonyms automatically
- Practical RAG (Retrieval Augmented Generation) example

**Applications:**

- Document retrieval
- Question answering
- Recommendation systems

**Reference:** Lessons 1-7 (complete application)

### Example 5: 3D Visualization of Embeddings

**Concepts:**

- Dimensionality reduction (PCA)
- Semantic clustering
- Visual exploration

**Key Takeaways:**

- Semantically similar texts cluster together
- Visualizes the concept of "embedding space"
- Distance in space = semantic distance

**Visualizations:**

- 3D scatter plot with categories
- 2D projections (PC1 vs PC2, PC1 vs PC3, PC2 vs PC3)
- Cluster analysis

**Reference:** Lesson 3 of Block A

**Recommended:** CMU Word Embedding Demo - <https://www.cs.cmu.edu/~dst/WordEmbeddingDemo>

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

### Alternative Models

For other languages or requirements:

- **Multilingual:** `paraphrase-multilingual-MiniLM-L12-v2` (supports 50+ languages)
- **Higher quality:** `all-mpnet-base-v2` (768 dims, ~420MB)
- **Even smaller:** `paraphrase-MiniLM-L3-v2` (~60MB)

## Learning Path

Recommended order to run the examples:

1. **Example 1** - Understand why naive approaches fail
2. **Example 2** - See how neural embeddings work
3. **Example 3** - Learn similarity metrics
4. **Example 4** - Build a practical application
5. **Example 5** - Visualize the embedding space

## Key Concepts Summary

### Embeddings

- Numerical representations of objects (text, images, etc)
- Dense vectors in N-dimensional space
- Capture semantic meaning
- Generated by neural networks

### Properties of Good Embeddings

1. **Dense:** Few dimensions, all informative
2. **Semantic:** Similar objects → close vectors
3. **Unique:** Each object has one representation
4. **Learned:** Automatically extracted from data

### Similarity Metrics

| Metric | Range | Best For | Normalized? |
|--------|-------|----------|-------------|
| Cosine | [-1, 1] | NLP, text | Yes |
| Euclidean | [0, ∞) | Clustering, images | No |
| Dot Product | (-∞, ∞) | Neural networks | No |

### Applications

- Semantic search
- Document retrieval
- Recommendation systems
- Question answering (RAG)
- Text classification
- Clustering
- Similarity detection

## Troubleshooting

### Model Download Issues

If download fails, the model is cached in:

- Windows: `C:\Users\<user>\.cache\huggingface\hub\`
- Linux/Mac: `~/.cache/huggingface/hub/`

You can manually download from HuggingFace and place it there.

### Memory Issues

The model uses ~200MB RAM. If you have memory constraints:

- Close other applications
- Use a smaller model (`paraphrase-MiniLM-L3-v2`)

### Import Errors

Make sure you activated the virtual environment:

```bash
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

## References

### Block A Materials

- Presentations: `.github/docs/content/presentations/b1/`
- Summary: `.github/docs/content/resumes/rb1.md`

### External Resources

- Sentence Transformers: <https://www.sbert.net/>
- HuggingFace Models: <https://huggingface.co/models>
- CMU Word Embedding Demo: <https://www.cs.cmu.edu/~dst/WordEmbeddingDemo>

## License

Educational project for learning purposes.

## Contributing

This is a teaching project. Feel free to:

- Suggest improvements
- Add more examples
- Report issues
- Share with others learning about embeddings

## Next Steps

After completing Block A, explore:

- **Block B:** Vector databases and knowledge graphs
- **Block C:** Advanced search techniques
- Fine-tuning embedding models
- Building RAG applications
- Multimodal embeddings (text + images)

## Contact

For questions or suggestions about this project, refer to the course materials or instructor.

---

**Happy Learning!** 🚀
