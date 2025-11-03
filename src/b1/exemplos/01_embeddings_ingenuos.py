"""
Example 1: Naive Embeddings
============================

Concepts covered:
- One-hot encoding of words
- Sparse vectors without semantics
- Limitations: all distances are equal
- Comparison with bag-of-words

Reference: Lesson 5 of Block A - "Naive embeddings"

This example demonstrates why naive approaches like one-hot encoding
DO NOT capture the semantics of objects, resulting in representations
that do not reflect real similarities.
"""

import numpy as np
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity


def create_one_hot(words):
    """
    Creates one-hot representation for a list of words.

    Args:
        words: List of strings

    Returns:
        one_hot_matrix: Numpy array (n_words, n_words)
        vocabulary: Dictionary mapping word -> index
    """
    vocabulary = {word: idx for idx, word in enumerate(words)}
    n_words = len(words)

    # Identity matrix = one-hot encoding
    one_hot_matrix = np.eye(n_words)

    return one_hot_matrix, vocabulary


def bag_of_words(text, vocabulary):
    """
    Creates bag-of-words representation (word count).

    Args:
        text: String with text
        vocabulary: Dictionary word -> index

    Returns:
        vector: Numpy array with count of each word
    """
    vector = np.zeros(len(vocabulary))
    text_words = text.lower().split()

    for word in text_words:
        if word in vocabulary:
            vector[vocabulary[word]] += 1

    return vector


def main():
    print("=" * 70)
    print("EXAMPLE 1: NAIVE EMBEDDINGS")
    print("=" * 70)
    print()

    # ========================================================================
    # PART 1: One-Hot Encoding
    # ========================================================================
    print("PART 1: One-Hot Encoding")
    print("-" * 70)
    print()

    # Example vocabulary with animals and objects
    words = ["cat", "dog", "lion", "tiger", "table", "chair"]

    print(f"Vocabulary: {words}")
    print()

    # Create one-hot representation
    one_hot_matrix, vocab = create_one_hot(words)

    print("One-Hot Matrix:")
    print("(each row represents a word, each column a dimension)")
    print()
    for i, word in enumerate(words):
        print(f"{word:10s}: {one_hot_matrix[i]}")
    print()

    # Characteristics of one-hot representation
    print("One-Hot Encoding Characteristics:")
    print(f"  - Dimensionality: {one_hot_matrix.shape[1]} (equal to vocabulary size)")
    print(f"  - Sparsity: {np.sum(one_hot_matrix == 0) / one_hot_matrix.size * 100:.1f}% zeros")
    print(f"  - Non-zero values per vector: {np.count_nonzero(one_hot_matrix[0])}")
    print()

    # ========================================================================
    # PART 2: Problem - All distances are equal!
    # ========================================================================
    print("PART 2: Limitation - No Semantics")
    print("-" * 70)
    print()

    # Calculate Euclidean distances between all words
    distances = euclidean_distances(one_hot_matrix)

    print("Euclidean Distances between words:")
    print()

    # Examples of word pairs
    pairs = [
        ("cat", "dog"),      # Domestic animals - should be similar
        ("lion", "tiger"),   # Wild felines - should be similar
        ("cat", "table"),    # Animal vs object - should be different
        ("table", "chair"),  # Objects - should be similar
    ]

    for word1, word2 in pairs:
        idx1 = vocab[word1]
        idx2 = vocab[word2]
        dist = distances[idx1, idx2]
        print(f"  {word1:10s} <-> {word2:10s}: {dist:.4f}")

    print()
    print("[!] PROBLEM: All distances are equal (~1.414)!")
    print("    One-hot encoding does NOT capture semantic relationships.")
    print("    'cat' is as distant from 'dog' as it is from 'table'.")
    print()

    # Cosine similarity (also doesn't help)
    similarities = cosine_similarity(one_hot_matrix)

    print("Cosine Similarities between words:")
    print()
    for word1, word2 in pairs:
        idx1 = vocab[word1]
        idx2 = vocab[word2]
        sim = similarities[idx1, idx2]
        print(f"  {word1:10s} <-> {word2:10s}: {sim:.4f}")

    print()
    print("[!] PROBLEM: Similarity is 0 for all different words!")
    print("    One-hot vectors are orthogonal (perpendicular).")
    print()

    # ========================================================================
    # PART 3: Bag-of-Words (Counting)
    # ========================================================================
    print("PART 3: Bag-of-Words (Word Counting)")
    print("-" * 70)
    print()

    # Example texts
    texts = [
        "cat and dog are animals",
        "lion and tiger are wild animals",
        "table and chair are furniture",
    ]

    # Expand vocabulary
    all_words = set()
    for text in texts:
        all_words.update(text.split())
    vocab_bow = {word: idx for idx, word in enumerate(sorted(all_words))}

    print(f"Expanded vocabulary ({len(vocab_bow)} words):")
    print(sorted(vocab_bow.keys()))
    print()

    # Create bag-of-words vectors
    bow_vectors = np.array([bag_of_words(text, vocab_bow) for text in texts])

    print("Bag-of-Words Vectors:")
    for i, text in enumerate(texts):
        print(f"Text {i+1}: \"{text}\"")
        print(f"  Vector: {bow_vectors[i]}")
        print(f"  Non-zero words: {np.count_nonzero(bow_vectors[i])}/{len(vocab_bow)}")
        print()

    # Calculate similarity
    sim_bow = cosine_similarity(bow_vectors)

    print("Similarity between texts (Bag-of-Words):")
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            print(f"  Text {i+1} <-> Text {j+1}: {sim_bow[i,j]:.4f}")
    print()

    print("[OK] Bag-of-Words captures SOME semantics (shared words)")
    print("     but is still limited: doesn't understand context or synonyms.")
    print()

    # ========================================================================
    # CONCLUSION
    # ========================================================================
    print("=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print()
    print("Problems with naive approaches:")
    print()
    print("1. ONE-HOT ENCODING:")
    print("   [X] Very sparse vectors (99%+ zeros)")
    print("   [X] Dimensionality = vocabulary size (poor scalability)")
    print("   [X] No semantics: all words are equally distant")
    print()
    print("2. BAG-OF-WORDS:")
    print("   [X] Still sparse for large vocabularies")
    print("   [X] Ignores word order")
    print("   [X] Doesn't capture synonyms ('cat' != 'feline')")
    print("   [OK] Captures some similarity through shared words")
    print()
    print("3. BITS/PIXELS:")
    print("   [X] Raw representation without semantic meaning")
    print("   [X] Small changes cause very different vectors")
    print()
    print("[!] SOLUTION: Embeddings generated by Neural Networks!")
    print("   [OK] Dense vectors (few dimensions, all informative)")
    print("   [OK] Capture semantics (similar objects = close vectors)")
    print("   [OK] Learn representations automatically from data")
    print()
    print("Next example: Embeddings with language models (Word2Vec, etc)")
    print("=" * 70)


if __name__ == "__main__":
    main()