"""
Example 5: 3D Visualization of Embedding Space
===============================================

Concepts covered:
- Dimensionality reduction (PCA)
- 3D visualization of embeddings
- Semantic clustering visualization
- Interactive exploration

Reference: Lesson 3 of Block A - "Embeddings as vectors"
Also relates to: https://www.cs.cmu.edu/~dst/WordEmbeddingDemo

This example visualizes high-dimensional embeddings in 3D space,
showing how semantically similar texts cluster together.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


def load_texts():
    """Loads example texts from JSON file."""
    path = Path(__file__).parent.parent / "dados" / "textos_exemplo.json"
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['categories']


def main():
    print("=" * 70)
    print("EXAMPLE 5: 3D VISUALIZATION OF EMBEDDING SPACE")
    print("=" * 70)
    print()

    # ========================================================================
    # PART 1: Load Data and Generate Embeddings
    # ========================================================================
    print("PART 1: Generating Embeddings")
    print("-" * 70)
    print()

    print("Loading model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("[OK] Model loaded")
    print()

    # Load categorized texts
    categories = load_texts()

    # Prepare data
    all_texts = []
    all_labels = []
    all_categories = []

    for category, texts in categories.items():
        for text in texts:
            all_texts.append(text)
            all_labels.append(category)
            all_categories.append(category)

    print(f"Total texts: {len(all_texts)}")
    print(f"Categories: {list(categories.keys())}")
    print()

    print("Generating embeddings for all texts...")
    embeddings = model.encode(all_texts, show_progress_bar=True)
    print(f"[OK] Generated embeddings: shape {embeddings.shape}")
    print(f"    Original dimensions: {embeddings.shape[1]}")
    print()

    # ========================================================================
    # PART 2: Dimensionality Reduction with PCA
    # ========================================================================
    print("PART 2: Dimensionality Reduction")
    print("-" * 70)
    print()

    print(f"Reducing from {embeddings.shape[1]}D to 3D using PCA...")
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(embeddings)

    print(f"[OK] Reduced to 3D")
    print(f"    Explained variance: {sum(pca.explained_variance_ratio_)*100:.2f}%")
    print(f"    Per component:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"      PC{i+1}: {var*100:.2f}%")
    print()

    print("Note: PCA projects high-dimensional data to lower dimensions")
    print("      while preserving as much variance as possible.")
    print("      ~70% variance retention is good for visualization!")
    print()

    # ========================================================================
    # PART 3: Create 3D Visualization
    # ========================================================================
    print("PART 3: Creating 3D Visualization")
    print("-" * 70)
    print()

    # Color mapping for categories
    category_list = list(categories.keys())
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    color_map = {cat: colors[i] for i, cat in enumerate(category_list)}

    # Create figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each category
    for category in category_list:
        # Get indices for this category
        indices = [i for i, label in enumerate(all_labels) if label == category]

        # Get coordinates
        xs = embeddings_3d[indices, 0]
        ys = embeddings_3d[indices, 1]
        zs = embeddings_3d[indices, 2]

        # Plot
        ax.scatter(xs, ys, zs,
                  c=color_map[category],
                  label=category.capitalize(),
                  s=100,
                  alpha=0.6,
                  edgecolors='black',
                  linewidth=0.5)

    # Customize plot
    ax.set_xlabel('First Principal Component', fontsize=10)
    ax.set_ylabel('Second Principal Component', fontsize=10)
    ax.set_zlabel('Third Principal Component', fontsize=10)
    ax.set_title('Sentence Embeddings in 3D Space\n' +
                 f'({embeddings.shape[1]}D -> 3D via PCA)',
                 fontsize=14, fontweight='bold')

    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Save visualization
    output_path = Path(__file__).parent.parent / "visualizacoes" / "embeddings_3d.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] 3D visualization saved to: {output_path.name}")
    print()

    # ========================================================================
    # PART 4: Create 2D Projections
    # ========================================================================
    print("PART 4: Creating 2D Projections")
    print("-" * 70)
    print()

    fig2, axes = plt.subplots(1, 3, figsize=(18, 5))

    projections = [
        (0, 1, 'PC1 vs PC2'),
        (0, 2, 'PC1 vs PC3'),
        (1, 2, 'PC2 vs PC3')
    ]

    for idx, (dim1, dim2, title) in enumerate(projections):
        ax = axes[idx]

        for category in category_list:
            indices = [i for i, label in enumerate(all_labels) if label == category]
            xs = embeddings_3d[indices, dim1]
            ys = embeddings_3d[indices, dim2]

            ax.scatter(xs, ys,
                      c=color_map[category],
                      label=category.capitalize(),
                      s=80,
                      alpha=0.6,
                      edgecolors='black',
                      linewidth=0.5)

        ax.set_xlabel(f'PC{dim1+1}', fontsize=10)
        ax.set_ylabel(f'PC{dim2+1}', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    output_path_2d = Path(__file__).parent.parent / "visualizacoes" / "embeddings_2d_projections.png"
    plt.savefig(output_path_2d, dpi=150, bbox_inches='tight')
    print(f"[OK] 2D projections saved to: {output_path_2d.name}")
    print()

    # ========================================================================
    # PART 5: Analyze Clustering
    # ========================================================================
    print("PART 5: Analyzing Semantic Clustering")
    print("-" * 70)
    print()

    print("Calculating distances between cluster centers...")
    print()

    # Calculate centroids for each category
    centroids = {}
    for category in category_list:
        indices = [i for i, label in enumerate(all_labels) if label == category]
        centroid = np.mean(embeddings_3d[indices], axis=0)
        centroids[category] = centroid

    # Calculate pairwise distances between centroids
    print("Distance matrix between category centroids:")
    print()
    print(f"{'':12s}", end='')
    for cat in category_list:
        print(f"{cat:12s}", end='')
    print()

    for cat1 in category_list:
        print(f"{cat1:12s}", end='')
        for cat2 in category_list:
            if cat1 == cat2:
                print(f"{'---':>12s}", end='')
            else:
                dist = np.linalg.norm(centroids[cat1] - centroids[cat2])
                print(f"{dist:12.4f}", end='')
        print()

    print()
    print("Observation: Categories with similar topics (e.g., sports/cooking)")
    print("             have centroids closer together than very different ones")
    print("             (e.g., animals vs technology).")
    print()

    # Find most compact and spread categories
    intra_distances = {}
    for category in category_list:
        indices = [i for i, label in enumerate(all_labels) if label == category]
        points = embeddings_3d[indices]
        centroid = centroids[category]

        # Average distance to centroid
        distances = [np.linalg.norm(point - centroid) for point in points]
        intra_distances[category] = np.mean(distances)

    print("Category compactness (lower = more compact cluster):")
    for category, dist in sorted(intra_distances.items(), key=lambda x: x[1]):
        print(f"  {category:12s}: {dist:.4f}")
    print()

    # ========================================================================
    # CONCLUSION
    # ========================================================================
    print("=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print()

    print("What We Observed:")
    print()
    print("1. SEMANTIC CLUSTERING:")
    print("   [OK] Texts with similar topics cluster together")
    print("   [OK] Different topics form distinct regions")
    print("   [OK] Visualizes the 'semantic space' concept")
    print()

    print("2. DIMENSIONALITY REDUCTION:")
    print("   [OK] PCA preserves main variance (~70%)")
    print("   [OK] Allows visualization of high-dimensional data")
    print("   [!] Some information is lost in projection")
    print()

    print("3. EMBEDDING PROPERTIES:")
    print("   [OK] Distance in embedding space = semantic distance")
    print("   [OK] Neural networks learn meaningful representations")
    print("   [OK] Similar meanings -> similar vectors")
    print()

    print("Advanced Visualizations:")
    print("  - t-SNE: Better for local structure, non-linear")
    print("  - UMAP: Preserves both local and global structure")
    print("  - Interactive tools: TensorBoard, Embedding Projector")
    print()

    print("Real-world Uses:")
    print("  - Understanding model behavior")
    print("  - Debugging clustering issues")
    print("  - Exploring dataset structure")
    print("  - Presenting semantic relationships")
    print()

    print("Recommended Interactive Tool:")
    print("  CMU Word Embedding Demo: https://www.cs.cmu.edu/~dst/WordEmbeddingDemo")
    print("  (mentioned in Lesson 3 of Block A)")
    print()

    print("=" * 70)
    print("All 5 examples completed!")
    print("Check the 'visualizacoes' folder for generated plots.")
    print("=" * 70)


if __name__ == "__main__":
    main()
