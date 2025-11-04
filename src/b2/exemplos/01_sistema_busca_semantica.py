"""
Example 1: Complete Semantic Search System
===========================================

Concepts covered:
- Building a knowledge base
- Generating embeddings with pre-trained models
- Storing embeddings in ChromaDB
- Performing semantic searches
- Comparing with keyword-based search

Reference: Lesson 1 of Block B - "Semantic search with embeddings"

This example demonstrates the three steps to build a semantic search system:
1. Select and prepare relevant documents
2. Generate embeddings using pre-trained models
3. Store in an efficient database (ChromaDB)

And how to perform searches:
1. Transform query into embedding
2. Compare with database embeddings
3. Return most similar documents
"""

import json
import os
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


def load_knowledge_base():
    """
    Load knowledge base from JSON file.

    Returns:
        list: List of documents with id, title, content, category, and tags
    """
    data_dir = Path(__file__).parent.parent / "dados"
    kb_path = data_dir / "base_conhecimento.json"

    with open(kb_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data['documentos']


def create_chroma_collection(model, documents):
    """
    Create ChromaDB collection and populate with documents.

    Args:
        model: SentenceTransformer model for generating embeddings
        documents: List of documents to index

    Returns:
        collection: ChromaDB collection object
    """
    # Create ChromaDB client (in-memory for this example)
    client = chromadb.Client(Settings(
        anonymized_telemetry=False,
        allow_reset=True
    ))

    # Create or get collection
    collection_name = "base_conhecimento"

    # Reset if exists
    try:
        client.delete_collection(name=collection_name)
    except:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "Knowledge base about countries and cities"}
    )

    print(f"[INFO] Creating collection '{collection_name}'...")
    print(f"[INFO] Processing {len(documents)} documents...")
    print()

    # Prepare data for batch insertion
    ids = []
    embeddings_list = []
    metadatas = []
    documents_text = []

    for doc in documents:
        # Generate embedding for document content
        text = f"{doc['titulo']}. {doc['conteudo']}"
        embedding = model.encode(text, convert_to_numpy=True)

        ids.append(doc['id'])
        embeddings_list.append(embedding.tolist())
        metadatas.append({
            'titulo': doc['titulo'],
            'categoria': doc['categoria'],
            'tags': ', '.join(doc['tags'])
        })
        documents_text.append(doc['conteudo'])

    # Add all documents at once (efficient!)
    collection.add(
        ids=ids,
        embeddings=embeddings_list,
        metadatas=metadatas,
        documents=documents_text
    )

    print(f"[OK] Successfully indexed {len(documents)} documents!")
    print()

    return collection


def semantic_search(collection, model, query, top_k=3):
    """
    Perform semantic search on collection.

    Args:
        collection: ChromaDB collection
        model: SentenceTransformer model
        query: Search query (text)
        top_k: Number of results to return

    Returns:
        dict: Query results with documents, metadatas, and distances
    """
    # Generate embedding for query
    query_embedding = model.encode(query, convert_to_numpy=True)

    # Search in ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=['documents', 'metadatas', 'distances']
    )

    return results


def keyword_search(documents, query, top_k=3):
    """
    Simple keyword-based search (for comparison).

    Args:
        documents: List of all documents
        query: Search query
        top_k: Number of results to return

    Returns:
        list: Top documents sorted by keyword matches
    """
    query_words = set(query.lower().split())
    scores = []

    for doc in documents:
        text = f"{doc['titulo']} {doc['conteudo']}".lower()

        # Count keyword matches
        matches = sum(1 for word in query_words if word in text)
        scores.append((doc, matches))

    # Sort by score (descending)
    scores.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, score in scores[:top_k]]


def print_results(results, query, search_type="SEMANTIC"):
    """
    Print search results in a formatted way.

    Args:
        results: Search results (ChromaDB format for semantic, list for keyword)
        query: Original query
        search_type: "SEMANTIC", "FILTERED SEMANTIC", or "KEYWORD"
    """
    print(f"[{search_type} SEARCH] Query: \"{query}\"")
    print("-" * 70)
    print()

    if search_type == "KEYWORD":
        # Keyword search results (list of documents)
        for i, doc in enumerate(results):
            print(f"Result {i+1}:")
            print(f"  ID: {doc['id']}")
            print(f"  Title: {doc['titulo']}")
            print(f"  Category: {doc['categoria']}")
            print(f"  Content: {doc['conteudo'][:200]}...")
            print()
    else:
        # ChromaDB results format (semantic and filtered semantic)
        for i in range(len(results['ids'][0])):
            doc_id = results['ids'][0][i]
            title = results['metadatas'][0][i]['titulo']
            category = results['metadatas'][0][i]['categoria']
            distance = results['distances'][0][i]
            similarity = 1 - distance  # Convert distance to similarity
            content = results['documents'][0][i]

            print(f"Result {i+1}:")
            print(f"  ID: {doc_id}")
            print(f"  Title: {title}")
            print(f"  Category: {category}")
            print(f"  Similarity: {similarity:.4f}")
            print(f"  Content: {content[:200]}...")
            print()


def main():
    print("=" * 70)
    print("EXAMPLE 1: COMPLETE SEMANTIC SEARCH SYSTEM")
    print("=" * 70)
    print()

    # ========================================================================
    # PART 1: Build Knowledge Base
    # ========================================================================
    print("PART 1: Building Knowledge Base")
    print("-" * 70)
    print()

    # Load documents
    documents = load_knowledge_base()
    print(f"[OK] Loaded {len(documents)} documents from knowledge base")
    print()

    # Show some statistics
    categories = {}
    for doc in documents:
        cat = doc['categoria']
        categories[cat] = categories.get(cat, 0) + 1

    print("Documents by category:")
    for cat, count in sorted(categories.items()):
        print(f"  - {cat}: {count} documents")
    print()

    # ========================================================================
    # PART 2: Generate Embeddings and Index
    # ========================================================================
    print("PART 2: Generating Embeddings and Indexing")
    print("-" * 70)
    print()

    print("[INFO] Loading model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print(
        f"[OK] Model loaded! Embedding dimension: {model.get_sentence_embedding_dimension()}")
    print()

    # Create collection and index documents
    collection = create_chroma_collection(model, documents)

    # ========================================================================
    # PART 3: Semantic Search Examples
    # ========================================================================
    print("=" * 70)
    print("PART 3: Semantic Search Examples")
    print("=" * 70)
    print()

    # Example queries
    queries = [
        "Qual é a capital da França?",
        "Monumentos famosos de Paris",
        "Clima tropical e quente",
        "Esportes no Brasil",
        "Museus de arte"
    ]

    for query in queries:
        results = semantic_search(collection, model, query, top_k=3)
        print_results(results, query, "SEMANTIC")
        print()

    # ========================================================================
    # PART 4: Comparison with Keyword Search
    # ========================================================================
    print("=" * 70)
    print("PART 4: Semantic vs Keyword Search Comparison")
    print("=" * 70)
    print()

    # Test query that benefits from semantic understanding
    test_query = "Onde fica a Mona Lisa?"

    print("[COMPARISON] Query: \"" + test_query + "\"")
    print()
    print("This query benefits from semantic search because:")
    print("  - 'Mona Lisa' is not directly mentioned in most documents")
    print("  - Semantic search understands 'Mona Lisa' is related to 'Louvre' and 'art'")
    print()

    # Semantic search
    results_semantic = semantic_search(collection, model, test_query, top_k=3)
    print_results(results_semantic, test_query, "SEMANTIC")

    print()
    print("-" * 70)
    print()

    # Keyword search
    results_keyword = keyword_search(documents, test_query, top_k=3)
    print_results(results_keyword, test_query, "KEYWORD")

    # ========================================================================
    # PART 5: Advanced Queries with Filters
    # ========================================================================
    print("=" * 70)
    print("PART 5: Advanced Search with Metadata Filters")
    print("=" * 70)
    print()

    # Search only in specific category
    query = "Informações sobre França"
    query_embedding = model.encode(query, convert_to_numpy=True)

    print(f"[FILTERED SEARCH] Query: \"{query}\"")
    print("Filter: category = 'geografia'")
    print("-" * 70)
    print()

    results_filtered = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=3,
        where={"categoria": "geografia"},
        include=['documents', 'metadatas', 'distances']
    )

    print_results(results_filtered, query, "FILTERED SEMANTIC")

    # ========================================================================
    # CONCLUSION
    # ========================================================================
    print("=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print()
    print("Key advantages of Semantic Search:")
    print()
    print("1. MEANING-BASED:")
    print("   [OK] Finds documents by semantic similarity, not just keywords")
    print("   [OK] Understands synonyms and related concepts")
    print("   [OK] Works with natural language queries")
    print()
    print("2. EFFICIENT:")
    print("   [OK] ChromaDB uses optimized algorithms (HNSW)")
    print("   [OK] Fast searches even with thousands of documents")
    print("   [OK] Embedding generation is done only once per document")
    print()
    print("3. FLEXIBLE:")
    print("   [OK] Can filter by metadata (category, tags, etc)")
    print("   [OK] Supports hybrid approaches (semantic + keyword)")
    print("   [OK] Easy to update and maintain")
    print()
    print("Applications:")
    print("  - Document retrieval systems")
    print("  - Question answering (RAG)")
    print("  - Recommendation engines")
    print("  - Customer support chatbots")
    print()
    print("Next example: Knowledge graphs and semantic navigation")
    print("=" * 70)


if __name__ == "__main__":
    main()
