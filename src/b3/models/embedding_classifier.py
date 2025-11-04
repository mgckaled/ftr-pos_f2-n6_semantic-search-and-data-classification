"""
Abordagem 1: Classificação com Embeddings pré-treinados + KNN.

Esta é uma abordagem "ingênua" mas eficaz que usa embeddings de um modelo
pré-treinado (sentence-transformers) e um classificador KNN simples.
"""

import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier

try:
    from .. import config
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import config


class EmbeddingClassifier:
    """
    Classificador baseado em embeddings pré-treinados e KNN.

    Esta abordagem:
    1. Gera embeddings para os textos usando sentence-transformers
    2. Treina um KNN com os embeddings
    3. Classifica novos textos encontrando os K vizinhos mais próximos
    """

    def __init__(
        self,
        model_name: str | None = None,
        k_neighbors: int | None = None,
        metric: str | None = None,
    ):
        """
        Inicializa o classificador.

        Args:
            model_name: Nome do modelo sentence-transformers
            k_neighbors: Número de vizinhos para o KNN
            metric: Métrica de distância ('cosine', 'euclidean', 'manhattan')
        """
        self.model_name = model_name or config.EMBEDDING_MODEL_NAME
        self.k_neighbors = k_neighbors or config.K_NEIGHBORS
        self.metric = metric or config.KNN_METRIC

        if config.VERBOSE:
            print(f"\n[Embedding Classifier] Inicializando...")
            print(f"  Modelo: {self.model_name}")
            print(f"  K vizinhos: {self.k_neighbors}")
            print(f"  Métrica: {self.metric}")

        # Carregar modelo de embeddings
        self.embedding_model = SentenceTransformer(self.model_name)

        # Inicializar KNN
        self.knn = KNeighborsClassifier(
            n_neighbors=self.k_neighbors,
            metric=self.metric,
            n_jobs=-1  # Usar todos os cores disponíveis
        )

        # Armazenar embeddings de treino (para análise posterior)
        self.train_embeddings = None
        self.train_labels = None

    def fit(self, texts: list[str], labels: list[int | str]) -> None:
        """
        Treina o classificador gerando embeddings e fitando o KNN.

        Args:
            texts: Lista de textos de treino
            labels: Lista de labels correspondentes
        """
        if config.VERBOSE:
            print(f"\n[Embedding Classifier] Gerando embeddings para {len(texts)} textos...")

        start_time = time.time()

        # Gerar embeddings para todos os textos
        self.train_embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=config.VERBOSE,
            convert_to_numpy=True,
            batch_size=32
        )

        self.train_labels = np.array(labels)

        # Treinar KNN com os embeddings
        self.knn.fit(self.train_embeddings, self.train_labels)

        elapsed_time = time.time() - start_time

        if config.VERBOSE:
            print(f"  Embeddings gerados: shape {self.train_embeddings.shape}")
            print(f"  Tempo de preparação: {elapsed_time:.2f}s")
            print(f"  KNN treinado com sucesso!")

    def predict(self, texts: list[str]) -> np.ndarray:
        """
        Prediz labels para novos textos.

        Args:
            texts: Lista de textos para classificar

        Returns:
            Array com labels preditos
        """
        if self.train_embeddings is None:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")

        # Gerar embeddings para os textos de teste
        test_embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=32
        )

        # Predizer usando KNN
        predictions = self.knn.predict(test_embeddings)

        return predictions

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """
        Prediz probabilidades de cada classe para novos textos.

        Args:
            texts: Lista de textos para classificar

        Returns:
            Array com probabilidades para cada classe
        """
        if self.train_embeddings is None:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")

        # Gerar embeddings para os textos de teste
        test_embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=32
        )

        # Predizer probabilidades usando KNN
        probabilities = self.knn.predict_proba(test_embeddings)

        return probabilities

    def get_nearest_neighbors(
        self, text: str, n_neighbors: int = 5
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Encontra os K vizinhos mais próximos de um texto.

        Args:
            text: Texto de consulta
            n_neighbors: Número de vizinhos a retornar

        Returns:
            distances: Distâncias para os vizinhos
            indices: Índices dos vizinhos no conjunto de treino
            labels: Labels dos vizinhos
        """
        if self.train_embeddings is None:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")

        # Gerar embedding do texto
        text_embedding = self.embedding_model.encode(
            [text], convert_to_numpy=True, show_progress_bar=False
        )

        # Encontrar vizinhos mais próximos
        distances, indices = self.knn.kneighbors(
            text_embedding, n_neighbors=n_neighbors, return_distance=True
        )

        # Pegar labels dos vizinhos
        neighbor_labels = self.train_labels[indices[0]]

        return distances[0], indices[0], neighbor_labels


if __name__ == "__main__":
    # Teste do módulo
    try:
        from ..utils.data_loader import load_and_prepare_dataset
    except ImportError:
        from utils.data_loader import load_and_prepare_dataset

    print("Testando Embedding Classifier...")

    # Carregar dados
    train_df, test_df, id2label, label2id = load_and_prepare_dataset(max_samples=1000)

    # Criar classificador
    classifier = EmbeddingClassifier()

    # Treinar
    classifier.fit(train_df["text"].tolist(), train_df["label"].tolist())

    # Testar com algumas amostras
    test_texts = test_df["text"].head(5).tolist()
    test_labels = test_df["label"].head(5).tolist()

    predictions = classifier.predict(test_texts)
    probabilities = classifier.predict_proba(test_texts)

    print("\nPredições:")
    for i, (text, true_label, pred_label, probs) in enumerate(
        zip(test_texts, test_labels, predictions, probabilities)
    ):
        print(f"\n{i+1}. Texto: {text[:80]}...")
        print(f"   Real: {id2label[true_label]} | Predito: {id2label[pred_label]}")
        print(f"   Probabilidades: {dict(zip(id2label.values(), probs))}")

    # Testar vizinhos mais próximos
    print("\n\nVizinhos mais próximos do primeiro texto:")
    distances, indices, neighbor_labels = classifier.get_nearest_neighbors(test_texts[0], n_neighbors=3)

    for i, (dist, idx, label) in enumerate(zip(distances, indices, neighbor_labels)):
        print(f"\n  Vizinho {i+1}:")
        print(f"    Distância: {dist:.4f}")
        print(f"    Label: {id2label[label]}")
