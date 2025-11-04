"""
Example 3: Evaluating Semantic Search
======================================

Concepts covered:
- Test sets with expected results
- Offline evaluation metrics: Precision@K, Recall@K, MRR
- Online evaluation with A/B testing (concept)
- Comparing different search configurations
- Trade-offs between different approaches

Reference: Lesson 3 of Block B - "Semantic search evaluation"

This example demonstrates how to objectively measure semantic search
effectiveness without relying solely on subjective analysis.
"""

import json
from pathlib import Path

import chromadb
import matplotlib.pyplot as plt
import numpy as np
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


def load_test_data():
    """
    Load test queries and knowledge base.

    Returns:
        tuple: (queries, documents)
    """
    data_dir = Path(__file__).parent.parent / "dados"

    # Load queries
    with open(data_dir / "queries_teste.json", 'r', encoding='utf-8') as f:
        queries_data = json.load(f)

    # Load documents
    with open(data_dir / "base_conhecimento.json", 'r', encoding='utf-8') as f:
        kb_data = json.load(f)

    return queries_data['queries'], kb_data['documentos']


def create_search_index(model, documents, collection_name="eval_collection"):
    """
    Create ChromaDB collection for evaluation.

    Args:
        model: SentenceTransformer model
        documents: List of documents
        collection_name: Name for collection

    Returns:
        collection: ChromaDB collection
    """
    client = chromadb.Client(Settings(
        anonymized_telemetry=False,
        allow_reset=True
    ))

    # Reset if exists
    try:
        client.delete_collection(name=collection_name)
    except:
        pass

    collection = client.create_collection(name=collection_name)

    # Index documents
    ids = []
    embeddings_list = []
    metadatas = []
    documents_text = []

    for doc in documents:
        text = f"{doc['titulo']}. {doc['conteudo']}"
        embedding = model.encode(text, convert_to_numpy=True)

        ids.append(doc['id'])
        embeddings_list.append(embedding.tolist())
        metadatas.append(
            {'titulo': doc['titulo'], 'categoria': doc['categoria']})
        documents_text.append(doc['conteudo'])

    collection.add(
        ids=ids,
        embeddings=embeddings_list,
        metadatas=metadatas,
        documents=documents_text
    )

    return collection


def search(collection, model, query, k=10):
    """
    Perform search and return document IDs.

    Args:
        collection: ChromaDB collection
        model: SentenceTransformer model
        query: Search query
        k: Number of results

    Returns:
        list: Retrieved document IDs in ranked order
    """
    query_embedding = model.encode(query, convert_to_numpy=True)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k,
        include=['distances']
    )

    return results['ids'][0]


def precision_at_k(retrieved, relevant, k):
    """
    Calculate Precision@K metric.

    Precision@K = (Number of relevant docs in top K) / K

    Args:
        retrieved: List of retrieved document IDs (ranked)
        relevant: Set of relevant document IDs
        k: Cutoff rank

    Returns:
        float: Precision@K score
    """
    retrieved_at_k = set(retrieved[:k])
    relevant_retrieved = retrieved_at_k.intersection(relevant)
    return len(relevant_retrieved) / k if k > 0 else 0.0


def recall_at_k(retrieved, relevant, k):
    """
    Calculate Recall@K metric.

    Recall@K = (Number of relevant docs in top K) / (Total relevant docs)

    Args:
        retrieved: List of retrieved document IDs (ranked)
        relevant: Set of relevant document IDs
        k: Cutoff rank

    Returns:
        float: Recall@K score
    """
    retrieved_at_k = set(retrieved[:k])
    relevant_retrieved = retrieved_at_k.intersection(relevant)
    return len(relevant_retrieved) / len(relevant) if len(relevant) > 0 else 0.0


def mean_reciprocal_rank(retrieved, relevant):
    """
    Calculate Mean Reciprocal Rank (MRR).

    MRR = 1 / (rank of first relevant document)

    Args:
        retrieved: List of retrieved document IDs (ranked)
        relevant: Set of relevant document IDs

    Returns:
        float: Reciprocal rank (0 if no relevant docs found)
    """
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_system(collection, model, test_queries, k_values=[1, 3, 5, 10]):
    """
    Evaluate search system on test set.

    Args:
        collection: ChromaDB collection
        model: SentenceTransformer model
        test_queries: List of test queries with expected results
        k_values: List of K values to evaluate

    Returns:
        dict: Evaluation results
    """
    results = {
        'precision': {k: [] for k in k_values},
        'recall': {k: [] for k in k_values},
        'mrr': []
    }

    for query_data in test_queries:
        query = query_data['query']
        relevant = set(query_data['documentos_relevantes'])

        # Perform search
        retrieved = search(collection, model, query, k=max(k_values))

        # Calculate metrics for each K
        for k in k_values:
            prec = precision_at_k(retrieved, relevant, k)
            rec = recall_at_k(retrieved, relevant, k)

            results['precision'][k].append(prec)
            results['recall'][k].append(rec)

        # Calculate MRR
        mrr = mean_reciprocal_rank(retrieved, relevant)
        results['mrr'].append(mrr)

    # Compute averages
    avg_results = {
        'precision': {},
        'recall': {},
        'mrr': np.mean(results['mrr'])
    }

    for k in k_values:
        avg_results['precision'][k] = np.mean(results['precision'][k])
        avg_results['recall'][k] = np.mean(results['recall'][k])

    return avg_results, results


def print_evaluation_results(results, title="EVALUATION RESULTS"):
    """
    Print evaluation results in formatted table.

    Args:
        results: Results dictionary from evaluate_system
        title: Title for the results
    """
    print(title)
    print("-" * 70)
    print()

    print(f"{'Metric':<20} {'Value':>10}")
    print("-" * 35)

    for k in sorted(results['precision'].keys()):
        prec = results['precision'][k]
        rec = results['recall'][k]
        print(f"Precision@{k:<13} {prec:>10.4f}")
        print(f"Recall@{k:<16} {rec:>10.4f}")
        print()

    print(f"{'MRR':<20} {results['mrr']:>10.4f}")
    print()


def visualize_metrics(results_list, labels, output_filename="evaluation_metrics.png"):
    """
    Create visualization of evaluation metrics.

    Args:
        results_list: List of evaluation results
        labels: List of labels for each result set
        output_filename: Output file name
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    k_values = sorted(results_list[0]['precision'].keys())

    # Plot 1: Precision@K
    ax1 = axes[0]
    for results, label in zip(results_list, labels):
        precisions = [results['precision'][k] for k in k_values]
        ax1.plot(k_values, precisions, marker='o', label=label, linewidth=2)

    ax1.set_xlabel('K', fontsize=12)
    ax1.set_ylabel('Precision@K', fontsize=12)
    ax1.set_title('Precision at Different Cutoffs',
                  fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])

    # Plot 2: Recall@K
    ax2 = axes[1]
    for results, label in zip(results_list, labels):
        recalls = [results['recall'][k] for k in k_values]
        ax2.plot(k_values, recalls, marker='s', label=label, linewidth=2)

    ax2.set_xlabel('K', fontsize=12)
    ax2.set_ylabel('Recall@K', fontsize=12)
    ax2.set_title('Recall at Different Cutoffs',
                  fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])

    plt.tight_layout()

    # Save
    output_dir = Path(__file__).parent.parent / "visualizacoes"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / output_filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Visualization saved to: {output_path}")

    plt.show()


def analyze_individual_queries(collection, model, test_queries, top_n=5):
    """
    Analyze performance on individual queries.

    Args:
        collection: ChromaDB collection
        model: SentenceTransformer model
        test_queries: List of test queries
        top_n: Number of worst/best queries to show
    """
    query_results = []

    for query_data in test_queries:
        query = query_data['query']
        relevant = set(query_data['documentos_relevantes'])
        retrieved = search(collection, model, query, k=10)

        prec_5 = precision_at_k(retrieved, relevant, 5)
        rec_5 = recall_at_k(retrieved, relevant, 5)
        mrr = mean_reciprocal_rank(retrieved, relevant)

        query_results.append({
            'query': query,
            'precision@5': prec_5,
            'recall@5': rec_5,
            'mrr': mrr,
            'score': (prec_5 + rec_5 + mrr) / 3  # Combined score
        })

    # Sort by score
    query_results.sort(key=lambda x: x['score'])

    print("WORST PERFORMING QUERIES")
    print("-" * 70)
    print()
    for i, result in enumerate(query_results[:top_n]):
        print(f"{i+1}. Query: \"{result['query']}\"")
        print(f"   Precision@5: {result['precision@5']:.4f}")
        print(f"   Recall@5: {result['recall@5']:.4f}")
        print(f"   MRR: {result['mrr']:.4f}")
        print()

    print()
    print("BEST PERFORMING QUERIES")
    print("-" * 70)
    print()
    for i, result in enumerate(reversed(query_results[-top_n:])):
        print(f"{i+1}. Query: \"{result['query']}\"")
        print(f"   Precision@5: {result['precision@5']:.4f}")
        print(f"   Recall@5: {result['recall@5']:.4f}")
        print(f"   MRR: {result['mrr']:.4f}")
        print()


def main():
    print("=" * 70)
    print("EXAMPLE 3: EVALUATING SEMANTIC SEARCH")
    print("=" * 70)
    print()

    # ========================================================================
    # PART 1: Load Test Data
    # ========================================================================
    print("PART 1: Loading Test Data")
    print("-" * 70)
    print()

    test_queries, documents = load_test_data()

    print(f"[OK] Loaded {len(test_queries)} test queries")
    print(f"[OK] Loaded {len(documents)} documents")
    print()

    # Show example test query
    example = test_queries[0]
    print("Example test query:")
    print(f"  Query: \"{example['query']}\"")
    print(f"  Expected relevant docs: {example['documentos_relevantes']}")
    print(f"  Description: {example['descricao']}")
    print()

    # ========================================================================
    # PART 2: Create Search Index
    # ========================================================================
    print("PART 2: Creating Search Index")
    print("-" * 70)
    print()

    print("[INFO] Loading model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("[OK] Model loaded!")
    print()

    collection = create_search_index(model, documents)
    print(f"[OK] Indexed {len(documents)} documents")
    print()

    # ========================================================================
    # PART 3: Offline Evaluation
    # ========================================================================
    print("PART 3: Offline Evaluation Metrics")
    print("-" * 70)
    print()

    print("[INFO] Evaluating search system on test set...")
    print()

    k_values = [1, 3, 5, 10]
    avg_results, detailed_results = evaluate_system(
        collection, model, test_queries, k_values)

    print_evaluation_results(avg_results)

    # ========================================================================
    # PART 4: Explain Metrics
    # ========================================================================
    print("PART 4: Understanding the Metrics")
    print("-" * 70)
    print()

    print("PRECISION@K:")
    print("  - Measures: How many retrieved docs are relevant?")
    print("  - Formula: (Relevant docs in top K) / K")
    print("  - High precision = Few irrelevant results")
    print("  - Use when: Users expect high quality results")
    print()

    print("RECALL@K:")
    print("  - Measures: How many relevant docs were retrieved?")
    print("  - Formula: (Relevant docs in top K) / (Total relevant docs)")
    print("  - High recall = Most relevant docs are found")
    print("  - Use when: Users want comprehensive results")
    print()

    print("MEAN RECIPROCAL RANK (MRR):")
    print("  - Measures: How early is the first relevant result?")
    print("  - Formula: 1 / (Rank of first relevant doc)")
    print("  - High MRR = Relevant results appear early")
    print("  - Use when: Users typically click first result")
    print()

    # ========================================================================
    # PART 5: Individual Query Analysis
    # ========================================================================
    print("=" * 70)
    print("PART 5: Individual Query Analysis")
    print("=" * 70)
    print()

    analyze_individual_queries(collection, model, test_queries, top_n=3)

    # ========================================================================
    # PART 6: Visualization
    # ========================================================================
    print("=" * 70)
    print("PART 6: Metrics Visualization")
    print("=" * 70)
    print()

    visualize_metrics([avg_results], ['Semantic Search'],
                      "semantic_search_evaluation.png")

    # ========================================================================
    # PART 7: Online Evaluation (Concept)
    # ========================================================================
    print("=" * 70)
    print("PART 7: Online Evaluation (A/B Testing)")
    print("=" * 70)
    print()

    print("Online evaluation complements offline metrics by measuring real user behavior:")
    print()
    print("A/B Testing Setup:")
    print("  1. Split users into two groups:")
    print("     - Group A: Uses current search system")
    print("     - Group B: Uses new search system")
    print()
    print("  2. Measure behavioral metrics:")
    print("     - Click-through rate (CTR): % of users who click results")
    print("     - Time to first click: How fast users find what they need")
    print("     - Session success rate: % of sessions ending with conversion")
    print("     - Engagement: Time spent on result pages")
    print()
    print("  3. Statistical significance:")
    print("     - Run test for sufficient time/users")
    print("     - Use statistical tests (t-test, chi-square)")
    print("     - Ensure results are not due to chance")
    print()
    print("Advantages:")
    print("  [OK] Measures real user satisfaction")
    print("  [OK] Captures business metrics (conversions, revenue)")
    print("  [OK] Reveals unexpected user behavior")
    print()
    print("Challenges:")
    print("  [X] Requires production deployment")
    print("  [X] Needs large user base for significance")
    print("  [X] Can temporarily hurt user experience")
    print()

    # ========================================================================
    # CONCLUSION
    # ========================================================================
    print("=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print()
    print("Evaluation Best Practices:")
    print()
    print("1. OFFLINE EVALUATION:")
    print("   [OK] Create test sets with ground truth")
    print("   [OK] Use multiple metrics (Precision, Recall, MRR)")
    print("   [OK] Analyze per-query performance")
    print("   [OK] Fast iteration during development")
    print()
    print("2. ONLINE EVALUATION:")
    print("   [OK] Measure real user behavior")
    print("   [OK] Use A/B testing carefully")
    print("   [OK] Track business metrics")
    print("   [OK] Validate offline findings")
    print()
    print("3. TRADE-OFFS:")
    print("   - High Precision vs High Recall")
    print("   - Search Speed vs Accuracy")
    print("   - Simplicity vs Feature Richness")
    print()
    print("Recommendations:")
    print("  - Start with offline evaluation")
    print("  - Build representative test sets")
    print("  - Combine multiple metrics")
    print("  - Validate with online testing when possible")
    print()
    print("Next example: Vector databases and efficient search")
    print("=" * 70)


if __name__ == "__main__":
    main()
