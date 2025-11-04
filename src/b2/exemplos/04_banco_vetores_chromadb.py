"""
Example 4: Vector Databases - ChromaDB
=======================================

Concepts covered:
- Efficient vector search with specialized algorithms
- HNSW (Hierarchical Navigable Small World) algorithm
- Trade-off between accuracy and speed
- Persistent storage vs in-memory
- Metadata filtering
- Collection management

Reference: Lesson 4 of Block B - "Vector databases"

This example demonstrates why we need specialized vector databases
instead of comparing every query with billions of documents.
"""

import json
import time
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import numpy as np
import matplotlib.pyplot as plt


def load_documents():
    """Load knowledge base documents."""
    data_dir = Path(__file__).parent.parent / "dados"
    with open(data_dir / "base_conhecimento.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['documentos']


def naive_search(embeddings, query_embedding, k=5):
    """
    Naive brute-force search (compares with ALL vectors).

    Args:
        embeddings: numpy array of document embeddings (n_docs, dim)
        query_embedding: query embedding vector (dim,)
        k: number of results

    Returns:
        tuple: (top_k_indices, top_k_scores, search_time)
    """
    start_time = time.time()

    # Calculate cosine similarity with ALL documents
    similarities = np.dot(embeddings, query_embedding)

    # Get top K
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    top_k_scores = similarities[top_k_indices]

    search_time = time.time() - start_time

    return top_k_indices, top_k_scores, search_time


def chromadb_search(collection, query_embedding, k=5):
    """
    ChromaDB search using HNSW algorithm.

    Args:
        collection: ChromaDB collection
        query_embedding: query embedding vector
        k: number of results

    Returns:
        tuple: (ids, distances, search_time)
    """
    start_time = time.time()

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k,
        include=['distances']
    )

    search_time = time.time() - start_time

    return results['ids'][0], results['distances'][0], search_time


def benchmark_search_speed(embeddings, doc_ids, model, queries, collection):
    """
    Benchmark search speed: Naive vs ChromaDB.

    Args:
        embeddings: Document embeddings array
        doc_ids: List of document IDs
        model: SentenceTransformer model
        queries: List of query strings
        collection: ChromaDB collection

    Returns:
        dict: Benchmark results
    """
    naive_times = []
    chroma_times = []

    print("[INFO] Running benchmark...")
    print(f"  - Number of documents: {len(embeddings)}")
    print(f"  - Number of queries: {len(queries)}")
    print(f"  - Embedding dimension: {embeddings.shape[1]}")
    print()

    for i, query in enumerate(queries):
        query_embedding = model.encode(query, convert_to_numpy=True)

        # Naive search
        _, _, naive_time = naive_search(embeddings, query_embedding, k=5)
        naive_times.append(naive_time)

        # ChromaDB search
        _, _, chroma_time = chromadb_search(collection, query_embedding, k=5)
        chroma_times.append(chroma_time)

        if (i + 1) % 5 == 0:
            print(f"  Progress: {i+1}/{len(queries)} queries processed")

    print()
    print("[OK] Benchmark complete!")
    print()

    return {
        'naive': {
            'times': naive_times,
            'avg_time': np.mean(naive_times),
            'std_time': np.std(naive_times)
        },
        'chromadb': {
            'times': chroma_times,
            'avg_time': np.mean(chroma_times),
            'std_time': np.std(chroma_times)
        }
    }


def demonstrate_persistence():
    """
    Demonstrate persistent vs in-memory storage.
    """
    print("DEMONSTRATING PERSISTENCE")
    print("-" * 70)
    print()

    # Create persistent client
    persist_dir = Path(__file__).parent.parent / "chroma_db"
    persist_dir.mkdir(exist_ok=True)

    client_persistent = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False)
    )

    # Create collection
    try:
        client_persistent.delete_collection("persistent_test")
    except:
        pass

    collection = client_persistent.create_collection("persistent_test")

    # Add some data
    collection.add(
        ids=["doc1", "doc2", "doc3"],
        documents=["Paris is beautiful", "Berlin is modern", "Rome is historic"],
        metadatas=[{"city": "Paris"}, {"city": "Berlin"}, {"city": "Rome"}]
    )

    print(f"[OK] Created persistent collection at: {persist_dir}")
    print(f"     Added 3 documents")
    print()

    # Query
    results = collection.query(
        query_texts=["romantic city"],
        n_results=1
    )
    print(f"Query: 'romantic city'")
    print(f"Result: {results['documents'][0][0]}")
    print()

    print("[INFO] Collection is saved to disk!")
    print("      You can reload it later without re-indexing.")
    print()

    return persist_dir


def demonstrate_metadata_filtering(collection, model):
    """
    Demonstrate metadata filtering capabilities.

    Args:
        collection: ChromaDB collection
        model: SentenceTransformer model
    """
    print("METADATA FILTERING")
    print("-" * 70)
    print()

    query = "informações culturais"
    query_embedding = model.encode(query, convert_to_numpy=True)

    # Search without filter
    results_all = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=3,
        include=['documents', 'metadatas', 'distances']
    )

    print(f"Query: \"{query}\" (NO FILTER)")
    print()
    for i in range(len(results_all['ids'][0])):
        title = results_all['metadatas'][0][i]['titulo']
        category = results_all['metadatas'][0][i]['categoria']
        print(f"  {i+1}. {title}")
        print(f"     Category: {category}")
    print()

    # Search with category filter
    results_filtered = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=3,
        where={"categoria": "cultura"},
        include=['documents', 'metadatas', 'distances']
    )

    print(f"Query: \"{query}\" (FILTER: categoria = 'cultura')")
    print()
    for i in range(len(results_filtered['ids'][0])):
        title = results_filtered['metadatas'][0][i]['titulo']
        category = results_filtered['metadatas'][0][i]['categoria']
        print(f"  {i+1}. {title}")
        print(f"     Category: {category}")
    print()

    print("[OK] Metadata filters allow semantic search within subsets!")
    print()


def visualize_benchmark(results, output_filename="benchmark_comparison.png"):
    """
    Visualize benchmark results.

    Args:
        results: Benchmark results dictionary
        output_filename: Output file name
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Average times
    methods = ['Naive\n(Brute Force)', 'ChromaDB\n(HNSW)']
    avg_times = [
        results['naive']['avg_time'] * 1000,  # Convert to ms
        results['chromadb']['avg_time'] * 1000
    ]
    colors = ['#FF6B6B', '#4ECDC4']

    bars = ax1.bar(methods, avg_times, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Average Search Time (ms)', fontsize=12)
    ax1.set_title('Search Speed Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, time in zip(bars, avg_times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.4f} ms',
                ha='center', va='bottom', fontweight='bold')

    # Calculate speedup
    speedup = results['naive']['avg_time'] / results['chromadb']['avg_time']
    ax1.text(0.5, max(avg_times) * 0.85, f'Speedup: {speedup:.1f}x',
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
            transform=ax1.transData)

    # Plot 2: Time distribution
    ax2.boxplot(
        [np.array(results['naive']['times']) * 1000,
         np.array(results['chromadb']['times']) * 1000],
        labels=methods,
        patch_artist=True,
        boxprops=dict(facecolor='lightblue', alpha=0.7),
        medianprops=dict(color='red', linewidth=2)
    )
    ax2.set_ylabel('Search Time (ms)', fontsize=12)
    ax2.set_title('Search Time Distribution', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save
    output_dir = Path(__file__).parent.parent / "visualizacoes"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / output_filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Benchmark visualization saved to: {output_path}")

    plt.show()


def main():
    print("=" * 70)
    print("EXAMPLE 4: VECTOR DATABASES - ChromaDB")
    print("=" * 70)
    print()

    # ========================================================================
    # PART 1: Why Vector Databases?
    # ========================================================================
    print("PART 1: The Need for Vector Databases")
    print("-" * 70)
    print()

    print("Problem: Naive Search at Scale")
    print()
    print("Imagine searching through billions of documents:")
    print("  - Google: ~30+ trillion web pages indexed")
    print("  - Netflix: Millions of videos to recommend")
    print("  - Spotify: Tens of millions of songs")
    print()
    print("Naive approach (compare with ALL vectors):")
    print("  [X] 1 billion docs × 384 dimensions = 384 billion comparisons")
    print("  [X] At 1 microsecond/comparison = 384 seconds per query!")
    print("  [X] Completely impractical")
    print()
    print("Solution: Specialized Algorithms")
    print("  [OK] HNSW (Hierarchical Navigable Small World)")
    print("  [OK] LSH (Locality-Sensitive Hashing)")
    print("  [OK] IVF (Inverted File Index)")
    print("  [OK] These find approximate nearest neighbors FAST")
    print()

    # ========================================================================
    # PART 2: Load Data and Create Index
    # ========================================================================
    print("PART 2: Creating Vector Index")
    print("-" * 70)
    print()

    documents = load_documents()
    print(f"[OK] Loaded {len(documents)} documents")
    print()

    print("[INFO] Loading model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("[OK] Model loaded!")
    print()

    # Create ChromaDB collection
    client = chromadb.Client(Settings(anonymized_telemetry=False, allow_reset=True))

    try:
        client.delete_collection("vector_db_demo")
    except:
        pass

    collection = client.create_collection(
        name="vector_db_demo",
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )

    # Prepare embeddings for both ChromaDB and naive search
    print("[INFO] Generating embeddings...")
    embeddings_list = []
    doc_ids = []

    for doc in documents:
        text = f"{doc['titulo']}. {doc['conteudo']}"
        embedding = model.encode(text, convert_to_numpy=True)
        embeddings_list.append(embedding)
        doc_ids.append(doc['id'])

        collection.add(
            ids=[doc['id']],
            embeddings=[embedding.tolist()],
            metadatas=[{'titulo': doc['titulo'], 'categoria': doc['categoria']}],
            documents=[doc['conteudo']]
        )

    embeddings_matrix = np.array(embeddings_list)
    print(f"[OK] Indexed {len(documents)} documents")
    print(f"     Embedding shape: {embeddings_matrix.shape}")
    print()

    # ========================================================================
    # PART 3: Benchmark Speed
    # ========================================================================
    print("=" * 70)
    print("PART 3: Speed Benchmark - Naive vs ChromaDB")
    print("=" * 70)
    print()

    test_queries = [
        "Qual é a capital da França?",
        "Clima tropical do Brasil",
        "Monumentos em Paris",
        "Futebol brasileiro",
        "Culinária francesa",
        "Praias do Rio de Janeiro",
        "Arte e museus",
        "População europeia",
        "Biodiversidade amazônica",
        "Arquitetura histórica"
    ]

    benchmark_results = benchmark_search_speed(
        embeddings_matrix,
        doc_ids,
        model,
        test_queries,
        collection
    )

    print("BENCHMARK RESULTS:")
    print("-" * 70)
    print()
    print(f"Naive Search (Brute Force):")
    print(f"  Average time: {benchmark_results['naive']['avg_time']*1000:.4f} ms")
    print(f"  Std deviation: {benchmark_results['naive']['std_time']*1000:.4f} ms")
    print()
    print(f"ChromaDB (HNSW Algorithm):")
    print(f"  Average time: {benchmark_results['chromadb']['avg_time']*1000:.4f} ms")
    print(f"  Std deviation: {benchmark_results['chromadb']['std_time']*1000:.4f} ms")
    print()

    speedup = benchmark_results['naive']['avg_time'] / benchmark_results['chromadb']['avg_time']
    print(f"SPEEDUP: {speedup:.2f}x faster!")
    print()

    print("[NOTE] With our small dataset (20 docs), the difference is minimal.")
    print("       With millions of documents, HNSW is MUCH faster!")
    print()

    # ========================================================================
    # PART 4: Metadata Filtering
    # ========================================================================
    print("=" * 70)
    print("PART 4: Advanced Features - Metadata Filtering")
    print("=" * 70)
    print()

    demonstrate_metadata_filtering(collection, model)

    # ========================================================================
    # PART 5: Persistence
    # ========================================================================
    print("=" * 70)
    print("PART 5: Data Persistence")
    print("=" * 70)
    print()

    persist_dir = demonstrate_persistence()

    # ========================================================================
    # PART 6: Visualization
    # ========================================================================
    print("=" * 70)
    print("PART 6: Performance Visualization")
    print("=" * 70)
    print()

    visualize_benchmark(benchmark_results)

    # ========================================================================
    # PART 7: HNSW Algorithm Explanation
    # ========================================================================
    print("=" * 70)
    print("PART 7: How HNSW Works (Simplified)")
    print("=" * 70)
    print()

    print("HNSW = Hierarchical Navigable Small World Graph")
    print()
    print("Key Ideas:")
    print("  1. MULTI-LAYER GRAPH:")
    print("     - Top layers: Sparse, long-range connections (highway)")
    print("     - Bottom layers: Dense, short-range connections (local roads)")
    print()
    print("  2. GREEDY SEARCH:")
    print("     - Start at top layer")
    print("     - Navigate to closest neighbor at each step")
    print("     - Descend to lower layers when stuck")
    print("     - Refine search at bottom layer")
    print()
    print("  3. APPROXIMATE RESULTS:")
    print("     - May not find THE closest neighbor")
    print("     - But finds VERY close neighbor very fast")
    print("     - Accuracy vs speed trade-off is configurable")
    print()
    print("Complexity:")
    print("  - Brute force: O(n) - linear in dataset size")
    print("  - HNSW: O(log n) - logarithmic (MUCH better!)")
    print()
    print("Example:")
    print("  - 1 million docs:")
    print("    Brute force: ~1,000,000 comparisons")
    print("    HNSW: ~20 comparisons (50,000x reduction!)")
    print()

    # ========================================================================
    # CONCLUSION
    # ========================================================================
    print("=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print()
    print("Vector Databases Are Essential:")
    print()
    print("1. PERFORMANCE:")
    print("   [OK] Handle billions of vectors efficiently")
    print("   [OK] Sub-second queries at scale")
    print("   [OK] Logarithmic complexity vs linear")
    print()
    print("2. FEATURES:")
    print("   [OK] Metadata filtering (hybrid search)")
    print("   [OK] Persistent storage")
    print("   [OK] CRUD operations (add, update, delete)")
    print("   [OK] Multiple distance metrics")
    print()
    print("3. TRADE-OFFS:")
    print("   - Accuracy: 95-99% recall (configurable)")
    print("   - Speed: Orders of magnitude faster")
    print("   - Memory: More overhead for index structures")
    print()
    print("Popular Vector Databases:")
    print("  - ChromaDB: Easy to use, embedded")
    print("  - Pinecone: Managed cloud service")
    print("  - Weaviate: Feature-rich, scalable")
    print("  - Qdrant: High performance, Rust-based")
    print("  - Milvus: Distributed, enterprise-grade")
    print("  - FAISS: Meta's library (not a full DB)")
    print()
    print("When to use:")
    print("  [OK] Semantic search in large document collections")
    print("  [OK] Recommendation systems")
    print("  [OK] Image/video similarity search")
    print("  [OK] RAG (Retrieval Augmented Generation)")
    print("  [OK] Anomaly detection")
    print()
    print("This completes Block B examples!")
    print("=" * 70)


if __name__ == "__main__":
    main()
