"""
Example 4: Simple Semantic Search System
=========================================

Concepts covered:
- Document database with embeddings
- Query encoding
- Similarity ranking
- Top-K results retrieval

Reference: Lessons 1-7 of Block A - Complete application

This example implements a basic semantic search engine that finds
relevant documents based on meaning rather than exact keyword matching.
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SimpleSemanticSearch:
    """Simple semantic search engine using sentence embeddings."""

    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize semantic search engine.

        Args:
            model_name: HuggingFace model identifier
        """
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.document_embeddings = None
        print("[OK] Model loaded")

    def add_documents(self, documents):
        """
        Add documents to the search index.

        Args:
            documents: List of dicts with 'id', 'title', 'text'
        """
        print(f"\nIndexing {len(documents)} documents...")
        self.documents = documents

        # Extract texts for embedding
        texts = [f"{doc['title']}. {doc['text']}" for doc in documents]

        # Generate embeddings
        self.document_embeddings = self.model.encode(texts, show_progress_bar=False)

        print(f"[OK] Indexed {len(documents)} documents")
        print(f"    Embedding shape: {self.document_embeddings.shape}")

    def search(self, query, top_k=5):
        """
        Search for relevant documents.

        Args:
            query: Search query string
            top_k: Number of results to return

        Returns:
            List of (document, score) tuples
        """
        if self.document_embeddings is None:
            raise ValueError("No documents indexed. Call add_documents() first.")

        # Encode query
        query_embedding = self.model.encode([query])[0]

        # Calculate similarities
        similarities = cosine_similarity(
            [query_embedding],
            self.document_embeddings
        )[0]

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Return documents with scores
        results = [
            (self.documents[idx], similarities[idx])
            for idx in top_indices
        ]

        return results


def load_documents():
    """Load documents from JSON file."""
    path = Path(__file__).parent.parent / "dados" / "textos_exemplo.json"
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['search_documents'], data['example_queries']


def print_results(query, results):
    """Pretty print search results."""
    print(f"\nQuery: \"{query}\"")
    print("=" * 70)
    print()

    for i, (doc, score) in enumerate(results, 1):
        print(f"Rank {i} (Score: {score:.4f})")
        print(f"  Title: {doc['title']}")
        print(f"  Text:  {doc['text']}")
        print()


def main():
    print("=" * 70)
    print("EXAMPLE 4: SEMANTIC SEARCH SYSTEM")
    print("=" * 70)
    print()

    # ========================================================================
    # PART 1: Initialize Search Engine
    # ========================================================================
    print("PART 1: Setting Up Semantic Search Engine")
    print("-" * 70)

    search_engine = SimpleSemanticSearch()

    # Load documents
    documents, example_queries = load_documents()

    print(f"\nAvailable documents: {len(documents)}")
    print("\nDocument titles:")
    for doc in documents:
        print(f"  - {doc['title']}")

    # Add documents to search index
    search_engine.add_documents(documents)
    print()

    # ========================================================================
    # PART 2: Example Searches
    # ========================================================================
    print("PART 2: Running Example Searches")
    print("-" * 70)
    print()

    print("Notice: Semantic search finds documents based on MEANING,")
    print("        not just keyword matching!")
    print()

    # Example 1: Programming query
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Programming Query")
    print("=" * 70)

    query1 = example_queries[0]  # "How to program in modern languages?"
    results1 = search_engine.search(query1, top_k=3)
    print_results(query1, results1)

    print("Observation: Found 'Python' and 'Machine Learning' docs")
    print("             even though query doesn't contain those exact words!")
    print()

    # Example 2: Animals query
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Animals Query")
    print("=" * 70)

    query2 = example_queries[1]  # "Which animals make good pets?"
    results2 = search_engine.search(query2, top_k=3)
    print_results(query2, results2)

    print("Observation: Correctly identifies pet-related and animal behavior docs")
    print()

    # Example 3: Health/Exercise query
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Health/Exercise Query")
    print("=" * 70)

    query3 = example_queries[2]  # "Exercises to improve health"
    results3 = search_engine.search(query3, top_k=3)
    print_results(query3, results3)

    print("Observation: Found swimming, physical training, and sports docs")
    print()

    # ========================================================================
    # PART 3: Comparing with Keyword Search
    # ========================================================================
    print("=" * 70)
    print("PART 3: Semantic Search vs Keyword Search")
    print("=" * 70)
    print()

    query_semantic = "artificial intelligence applications"

    print(f"Query: \"{query_semantic}\"")
    print()

    # Keyword search (simple word matching)
    print("KEYWORD SEARCH (naive approach):")
    keywords = set(query_semantic.lower().split())
    keyword_matches = []

    for doc in documents:
        text_words = set((doc['title'] + ' ' + doc['text']).lower().split())
        matches = len(keywords & text_words)
        if matches > 0:
            keyword_matches.append((doc, matches))

    keyword_matches.sort(key=lambda x: x[1], reverse=True)

    if keyword_matches:
        for i, (doc, matches) in enumerate(keyword_matches[:3], 1):
            print(f"  {i}. {doc['title']} ({matches} keyword matches)")
    else:
        print("  No matches found!")

    print()

    # Semantic search
    print("SEMANTIC SEARCH (embeddings):")
    results_semantic = search_engine.search(query_semantic, top_k=3)
    for i, (doc, score) in enumerate(results_semantic, 1):
        print(f"  {i}. {doc['title']} (similarity: {score:.4f})")

    print()
    print("Observation: Semantic search found relevant 'Machine Learning'")
    print("             and 'AI & Society' docs even without exact keywords!")
    print()

    # ========================================================================
    # PART 4: Additional Test Queries
    # ========================================================================
    print("=" * 70)
    print("PART 4: Try Additional Queries")
    print("=" * 70)
    print()

    custom_queries = [
        "learn to code",
        "pet care tips",
        "quantum mechanics",
        "food recipes"
    ]

    print("Testing additional queries:")
    print()

    for query in custom_queries:
        results = search_engine.search(query, top_k=1)
        top_doc, top_score = results[0]
        print(f"Query: \"{query}\"")
        print(f"  -> Top result: {top_doc['title']} (score: {top_score:.4f})")
        print()

    # ========================================================================
    # CONCLUSION
    # ========================================================================
    print("=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print()

    print("How Semantic Search Works:")
    print()
    print("1. INDEXING PHASE:")
    print("   [1] Documents are converted to embeddings")
    print("   [2] Embeddings are stored in the index")
    print()

    print("2. SEARCH PHASE:")
    print("   [1] Query is converted to embedding")
    print("   [2] Calculate similarity with all documents")
    print("   [3] Rank by similarity score")
    print("   [4] Return top-K results")
    print()

    print("Advantages over Keyword Search:")
    print()
    print("  [OK] Understands MEANING, not just words")
    print("  [OK] Handles synonyms automatically")
    print("      ('programming' matches 'Python', 'coding', etc)")
    print("  [OK] Works across languages (with multilingual models)")
    print("  [OK] Finds relevant docs even without exact terms")
    print()

    print("Limitations:")
    print()
    print("  [!] Requires pre-computing embeddings")
    print("  [!] Slower than traditional keyword search")
    print("  [!] Needs powerful embeddings for best results")
    print("  [!] Can miss exact matches if embedding is poor")
    print()

    print("Real-world Applications:")
    print("  - Document retrieval systems")
    print("  - Question answering (RAG - Retrieval Augmented Generation)")
    print("  - Recommendation engines")
    print("  - Customer support chatbots")
    print("  - E-commerce product search")
    print()

    print("Next example: 3D visualization of embedding space")
    print("=" * 70)


if __name__ == "__main__":
    main()
