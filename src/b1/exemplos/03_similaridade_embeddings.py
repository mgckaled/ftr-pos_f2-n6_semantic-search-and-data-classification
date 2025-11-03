"""
Example 3: Embedding Similarity Metrics
========================================

Concepts covered:
- Euclidean distance (L2)
- Cosine similarity
- Dot product
- When to use each metric
- Visualization with dimensionality reduction

Reference: Lesson 7 of Block A - "Embedding similarity"

This example demonstrates the three main similarity metrics used
with embeddings and compares their characteristics and use cases.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


def load_texts():
    """Loads example texts from JSON file."""
    path = Path(__file__).parent.parent / "dados" / "textos_exemplo.json"
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['categories']


def euclidean_distance_matrix(embeddings):
    """Calculates Euclidean distance (L2) between all pairs."""
    return euclidean_distances(embeddings)


def dot_product_similarity(embeddings):
    """Calculates dot product between all pairs."""
    return np.dot(embeddings, embeddings.T)


def main():
    print("=" * 70)
    print("EXAMPLE 3: EMBEDDING SIMILARITY METRICS")
    print("=" * 70)
    print()

    # ========================================================================
    # PART 1: Load Model and Generate Embeddings
    # ========================================================================
    print("PART 1: Generating Embeddings")
    print("-" * 70)
    print()

    print("Loading sentence-transformer model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("[OK] Model loaded")
    print()

    # Test sentences with different levels of similarity
    sentences = [
        "The cat is sleeping on the couch",
        "A feline is resting on the sofa",
        "The dog is playing in the garden",
        "Python is a programming language",
        "Machine learning uses neural networks"
    ]

    print("Test sentences:")
    for i, sent in enumerate(sentences):
        print(f"  {i}: \"{sent}\"")
    print()

    embeddings = model.encode(sentences)
    print(f"Generated embeddings: shape {embeddings.shape}")
    print()

    # ========================================================================
    # PART 2: Metric 1 - Euclidean Distance (L2)
    # ========================================================================
    print("PART 2: Euclidean Distance (L2)")
    print("-" * 70)
    print()

    print("Definition: Direct distance between two vectors in space")
    print("Formula: d(u,v) = sqrt(sum((u_i - v_i)^2))")
    print("Range: [0, infinity) - Lower is MORE similar")
    print()

    distances = euclidean_distance_matrix(embeddings)

    print("Euclidean Distances:")
    print("(lower = more similar)")
    print()
    pairs_to_compare = [
        (0, 1, "Similar meaning (cat/feline)"),
        (0, 2, "Same domain (animals)"),
        (0, 3, "Different domains (animal vs tech)"),
        (3, 4, "Same domain (technology)")
    ]

    for idx1, idx2, description in pairs_to_compare:
        dist = distances[idx1, idx2]
        print(f"  [{idx1}] <-> [{idx2}] ({description}): {dist:.4f}")
    print()

    print("Characteristics:")
    print("  [OK] Intuitive (geometric distance)")
    print("  [!] Sensitive to vector magnitude")
    print("  [!] Not normalized (unbounded)")
    print()

    # ========================================================================
    # PART 3: Metric 2 - Cosine Similarity
    # ========================================================================
    print("PART 3: Cosine Similarity")
    print("-" * 70)
    print()

    print("Definition: Measures angle between vectors")
    print("Formula: cos(u,v) = (u·v) / (||u|| ||v||)")
    print("Range: [-1, 1] - Higher is MORE similar")
    print("  1.0 = same direction (identical)")
    print("  0.0 = orthogonal (unrelated)")
    print(" -1.0 = opposite direction")
    print()

    similarities = cosine_similarity(embeddings)

    print("Cosine Similarities:")
    print("(higher = more similar)")
    print()

    for idx1, idx2, description in pairs_to_compare:
        sim = similarities[idx1, idx2]
        print(f"  [{idx1}] <-> [{idx2}] ({description}): {sim:.4f}")
    print()

    print("Characteristics:")
    print("  [OK] Normalized to [-1, 1]")
    print("  [OK] Ignores magnitude (only direction matters)")
    print("  [OK] Most common for NLP tasks")
    print("  [!] Can miss magnitude differences")
    print()

    # ========================================================================
    # PART 4: Metric 3 - Dot Product
    # ========================================================================
    print("PART 4: Dot Product")
    print("-" * 70)
    print()

    print("Definition: Sum of element-wise products")
    print("Formula: u·v = sum(u_i * v_i)")
    print("Range: (-infinity, infinity) - Higher is MORE similar")
    print()

    dot_products = dot_product_similarity(embeddings)

    print("Dot Products:")
    print("(higher = more similar)")
    print()

    for idx1, idx2, description in pairs_to_compare:
        dot = dot_products[idx1, idx2]
        print(f"  [{idx1}] <-> [{idx2}] ({description}): {dot:.4f}")
    print()

    print("Characteristics:")
    print("  [OK] Fast to compute")
    print("  [OK] Considers both angle AND magnitude")
    print("  [!] Not normalized (unbounded)")
    print("  [!] Favors longer vectors")
    print()

    # ========================================================================
    # PART 5: Comparison and When to Use Each
    # ========================================================================
    print("PART 5: Metric Comparison")
    print("-" * 70)
    print()

    # Normalize embeddings for fair comparison
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    print("Vector norms (lengths):")
    for i in range(len(embeddings)):
        norm = np.linalg.norm(embeddings[i])
        print(f"  Sentence {i}: {norm:.4f}")
    print()

    print("IMPORTANT: For normalized embeddings, cosine similarity")
    print("and dot product are equivalent!")
    print()

    # Verify equivalence
    cos_sim_01 = similarities[0, 1]
    dot_norm_01 = np.dot(embeddings_norm[0], embeddings_norm[1])
    print(f"Cosine similarity [0,1]: {cos_sim_01:.6f}")
    print(f"Dot product (normalized) [0,1]: {dot_norm_01:.6f}")
    print(f"Difference: {abs(cos_sim_01 - dot_norm_01):.10f}")
    print()

    # ========================================================================
    # PART 6: 2D Visualization
    # ========================================================================
    print("PART 6: 2D Visualization with PCA")
    print("-" * 70)
    print()

    print("Reducing 384 dimensions to 2D for visualization...")
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    print(f"[OK] Explained variance: {sum(pca.explained_variance_ratio_)*100:.2f}%")
    print()

    # Create visualization
    plt.figure(figsize=(10, 8))

    colors = ['red', 'red', 'blue', 'green', 'green']
    labels_viz = ['Cat/Couch', 'Feline/Sofa', 'Dog/Garden', 'Python lang', 'ML/Neural']

    for i, (x, y) in enumerate(embeddings_2d):
        plt.scatter(x, y, c=colors[i], s=200, alpha=0.6)
        plt.annotate(f'{i}: {labels_viz[i]}', (x, y),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, bbox=dict(boxstyle='round,pad=0.5',
                    facecolor=colors[i], alpha=0.3))

    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('Sentence Embeddings in 2D Space\n(Similar colors = similar topics)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

    output_path = Path(__file__).parent.parent / "visualizacoes" / "similarity_2d.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"[OK] Visualization saved to: {output_path.name}")
    print()

    # ========================================================================
    # CONCLUSION
    # ========================================================================
    print("=" * 70)
    print("CONCLUSIONS: When to Use Each Metric")
    print("=" * 70)
    print()

    print("1. COSINE SIMILARITY (most common for NLP):")
    print("   Use when:")
    print("   [OK] Working with text embeddings")
    print("   [OK] Vector magnitude doesn't matter")
    print("   [OK] Need normalized similarity scores")
    print("   Examples: document similarity, semantic search")
    print()

    print("2. EUCLIDEAN DISTANCE:")
    print("   Use when:")
    print("   [OK] Magnitude matters (e.g., vector differences)")
    print("   [OK] Working in metric spaces")
    print("   [OK] Clustering algorithms (K-means)")
    print("   Examples: image embeddings, feature vectors")
    print()

    print("3. DOT PRODUCT:")
    print("   Use when:")
    print("   [OK] Need computational efficiency")
    print("   [OK] Working with normalized embeddings")
    print("   [OK] Matrix operations (neural networks)")
    print("   Examples: attention mechanisms, modern LLMs")
    print()

    print("TIP: Many modern embedding models (like this one)")
    print("     produce normalized embeddings, making cosine similarity")
    print("     and dot product equivalent!")
    print()

    print("Next example: Semantic search system")
    print("=" * 70)


if __name__ == "__main__":
    main()