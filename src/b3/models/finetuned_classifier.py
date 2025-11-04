"""
Abordagem 2: Fine-tuning de modelo pré-treinado (DistilBERT).

Esta abordagem adapta um modelo pré-treinado para a tarefa específica
através de fine-tuning, melhorando a performance comparada aos embeddings simples.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

try:
    from .. import config
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import config


class FinetunedClassifier:
    """
    Classificador baseado em fine-tuning de DistilBERT.

    Esta abordagem:
    1. Carrega DistilBERT pré-treinado
    2. Adiciona camada de classificação customizada
    3. Faz fine-tuning no dataset específico
    4. Usa o modelo adaptado para classificação
    """

    def __init__(
        self,
        model_name: str | None = None,
        num_labels: int | None = None,
        id2label: dict[int, str] | None = None,
        label2id: dict[str, int] | None = None,
    ):
        """
        Inicializa o classificador.

        Args:
            model_name: Nome do modelo base do Hugging Face
            num_labels: Número de classes
            id2label: Mapeamento de índice para label
            label2id: Mapeamento de label para índice
        """
        self.model_name = model_name or config.FINETUNED_MODEL_NAME
        self.num_labels = num_labels
        self.id2label = id2label or {}
        self.label2id = label2id or {}

        if config.VERBOSE:
            print(f"\n[Finetuned Classifier] Inicializando...")
            print(f"  Modelo base: {self.model_name}")
            print(f"  Número de labels: {self.num_labels}")

        # Inicializar tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Modelo e trainer serão inicializados no fit()
        self.model = None
        self.trainer = None

    def _prepare_dataset(
        self, texts: list[str], labels: list[int] | None = None
    ) -> Dataset:
        """
        Prepara dataset no formato esperado pelo Trainer.

        Args:
            texts: Lista de textos
            labels: Lista de labels (opcional para predição)

        Returns:
            Dataset tokenizado
        """
        # Criar dicionário de dados
        data_dict = {"text": texts}
        if labels is not None:
            data_dict["label"] = labels

        # Converter para Dataset do Hugging Face
        dataset = Dataset.from_dict(data_dict)

        # Tokenizar
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=config.MAX_LENGTH,
                padding=False,  # Padding dinâmico será feito pelo DataCollator
            )

        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, remove_columns=["text"]
        )

        return tokenized_dataset

    def train(
        self,
        train_texts: list[str],
        train_labels: list[int],
        eval_texts: list[str] | None = None,
        eval_labels: list[int] | None = None,
        epochs: int | None = None,
    ) -> None:
        """
        Faz fine-tuning do modelo.

        Args:
            train_texts: Textos de treino
            train_labels: Labels de treino
            eval_texts: Textos de validação (opcional)
            eval_labels: Labels de validação (opcional)
            epochs: Número de épocas
        """
        epochs = epochs or config.EPOCHS

        if config.VERBOSE:
            print(f"\n[Finetuned Classifier] Iniciando fine-tuning...")
            print(f"  Amostras de treino: {len(train_texts)}")
            if eval_texts:
                print(f"  Amostras de validação: {len(eval_texts)}")
            print(f"  Épocas: {epochs}")

        # Preparar datasets
        train_dataset = self._prepare_dataset(train_texts, train_labels)
        eval_dataset = None
        if eval_texts is not None and eval_labels is not None:
            eval_dataset = self._prepare_dataset(eval_texts, eval_labels)

        # Carregar modelo com camada de classificação
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )

        # Configurar argumentos de treinamento
        training_args = TrainingArguments(
            output_dir=str(config.FINETUNED_MODEL_PATH),
            num_train_epochs=epochs,
            per_device_train_batch_size=config.BATCH_SIZE,
            per_device_eval_batch_size=config.BATCH_SIZE,
            learning_rate=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            warmup_steps=config.WARMUP_STEPS,
            logging_steps=config.LOGGING_STEPS,
            save_steps=config.SAVE_STEPS,
            eval_steps=config.EVAL_STEPS if eval_dataset else None,
            eval_strategy="steps" if eval_dataset else "no",  # Renamed from evaluation_strategy
            save_strategy="steps",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            report_to="none",  # Desabilitar wandb, tensorboard, etc
            logging_dir=str(config.RESULTS_DIR / "logs"),
        )

        # Data collator para padding dinâmico
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # Criar Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        # Treinar
        if config.VERBOSE:
            print("\n  Iniciando treinamento...")

        self.trainer.train()

        if config.VERBOSE:
            print("  Fine-tuning concluído!")

    def predict(self, texts: list[str]) -> np.ndarray:
        """
        Prediz labels para novos textos.

        Args:
            texts: Lista de textos para classificar

        Returns:
            Array com labels preditos
        """
        if self.model is None:
            raise ValueError("Modelo não foi treinado. Execute train() primeiro.")

        # Preparar dataset
        test_dataset = self._prepare_dataset(texts)

        # Fazer predições
        predictions = self.trainer.predict(test_dataset)

        # Extrair labels preditos
        pred_labels = np.argmax(predictions.predictions, axis=1)

        return pred_labels

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """
        Prediz probabilidades de cada classe para novos textos.

        Args:
            texts: Lista de textos para classificar

        Returns:
            Array com probabilidades para cada classe
        """
        if self.model is None:
            raise ValueError("Modelo não foi treinado. Execute train() primeiro.")

        # Preparar dataset
        test_dataset = self._prepare_dataset(texts)

        # Fazer predições
        predictions = self.trainer.predict(test_dataset)

        # Converter logits para probabilidades (softmax)
        logits = predictions.predictions
        probabilities = torch.nn.functional.softmax(
            torch.tensor(logits), dim=1
        ).numpy()

        return probabilities

    def save_model(self, path: str | None = None) -> None:
        """
        Salva o modelo treinado.

        Args:
            path: Caminho para salvar (usa config.FINETUNED_MODEL_PATH por padrão)
        """
        if self.model is None:
            raise ValueError("Modelo não foi treinado. Execute train() primeiro.")

        save_path = path or str(config.FINETUNED_MODEL_PATH)

        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        if config.VERBOSE:
            print(f"\n[Finetuned Classifier] Modelo salvo em: {save_path}")

    def load_model(self, path: str | None = None) -> None:
        """
        Carrega um modelo previamente treinado.

        Args:
            path: Caminho do modelo (usa config.FINETUNED_MODEL_PATH por padrão)
        """
        load_path = path or str(config.FINETUNED_MODEL_PATH)

        if config.VERBOSE:
            print(f"\n[Finetuned Classifier] Carregando modelo de: {load_path}")

        self.model = AutoModelForSequenceClassification.from_pretrained(load_path)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)

        # Reconstruir trainer para poder fazer predições
        training_args = TrainingArguments(
            output_dir=str(config.FINETUNED_MODEL_PATH),
            per_device_eval_batch_size=config.BATCH_SIZE,
            report_to="none",
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        if config.VERBOSE:
            print("  Modelo carregado com sucesso!")


if __name__ == "__main__":
    # Teste do módulo
    try:
        from ..utils.data_loader import load_and_prepare_dataset
    except ImportError:
        from utils.data_loader import load_and_prepare_dataset

    print("Testando Finetuned Classifier...")

    # Carregar dados (pequeno dataset para teste rápido)
    train_df, test_df, id2label, label2id = load_and_prepare_dataset(max_samples=500)

    # Criar classificador
    classifier = FinetunedClassifier(
        num_labels=len(id2label), id2label=id2label, label2id=label2id
    )

    # Treinar (apenas 1 época para teste)
    classifier.train(
        train_texts=train_df["text"].tolist(),
        train_labels=train_df["label"].tolist(),
        epochs=1,
    )

    # Testar predições
    test_texts = test_df["text"].head(3).tolist()
    test_labels = test_df["label"].head(3).tolist()

    predictions = classifier.predict(test_texts)
    probabilities = classifier.predict_proba(test_texts)

    print("\nPredições:")
    for i, (text, true_label, pred_label, probs) in enumerate(
        zip(test_texts, test_labels, predictions, probabilities)
    ):
        print(f"\n{i+1}. Texto: {text[:80]}...")
        print(f"   Real: {id2label[true_label]} | Predito: {id2label[pred_label]}")
        print(f"   Probabilidades: {dict(zip(id2label.values(), probs))}")
