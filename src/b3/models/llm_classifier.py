"""
Abordagem 3: Classificação com LLM usando Prompt Engineering (Gemini).

Esta abordagem usa um modelo de linguagem grande (LLM) com prompts bem
estruturados para classificação zero-shot ou few-shot, sem necessidade de
treinamento.
"""

import json
import sys
import time
from enum import Enum
from pathlib import Path

import numpy as np
from google import genai

try:
    from .. import config
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import config


class LLMClassifier:
    """
    Classificador baseado em LLM (Gemini) com prompt engineering.

    Esta abordagem:
    1. Cria prompts estruturados descrevendo a tarefa
    2. Usa few-shot examples para melhorar precisão
    3. Configura formato de saída JSON estruturado
    4. Classifica usando o conhecimento do LLM
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        labels: list[str] | None = None,
        few_shot_examples: list[dict[str, str]] | None = None,
    ):
        """
        Inicializa o classificador.

        Args:
            api_key: Chave da API do Google Gemini
            model_name: Nome do modelo Gemini
            labels: Lista de labels possíveis
            few_shot_examples: Exemplos few-shot (formato: [{"text": ..., "label": ...}])
        """
        self.api_key = api_key or config.GEMINI_API_KEY
        self.model_name = model_name or config.LLM_MODEL_NAME
        self.labels = labels or []
        self.few_shot_examples = few_shot_examples or config.FEW_SHOT_EXAMPLES

        if not self.api_key:
            raise ValueError(
                "API key do Gemini não configurada. "
                "Defina GEMINI_API_KEY no arquivo .env"
            )

        if config.VERBOSE:
            print(f"\n[LLM Classifier] Inicializando...")
            print(f"  Modelo: {self.model_name}")
            print(f"  Labels: {self.labels}")
            print(f"  Few-shot examples: {len(self.few_shot_examples)}")

        # Inicializar cliente Gemini
        self.client = genai.Client(api_key=self.api_key)

        # Criar enum dinâmico com os labels para resposta estruturada
        self.LabelEnum = None
        if self.labels:
            self._create_label_enum()

    def _create_label_enum(self) -> None:
        """Cria um Enum dinâmico com os labels para resposta estruturada."""
        # Criar dicionário de labels sanitizados (ex: "very positive" -> "VERY_POSITIVE")
        label_dict = {
            label.upper().replace(" ", "_").replace("-", "_"): label
            for label in self.labels
        }

        # Criar Enum
        self.LabelEnum = Enum("LabelEnum", label_dict)

    def _create_prompt(self, text: str) -> str:
        """
        Cria prompt estruturado para classificação.

        Args:
            text: Texto a ser classificado

        Returns:
            Prompt formatado
        """
        # Instrução base
        prompt = f"""Você é um sistema de classificação de texto. Sua tarefa é classificar textos em uma das seguintes categorias:

Categorias disponíveis: {", ".join(self.labels)}

"""

        # Adicionar few-shot examples se disponíveis
        if self.few_shot_examples:
            prompt += "Aqui estão alguns exemplos:\n\n"
            for i, example in enumerate(self.few_shot_examples, 1):
                prompt += f"Exemplo {i}:\n"
                prompt += f"Texto: \"{example['text']}\"\n"
                prompt += f"Categoria: {example['label']}\n\n"

        # Adicionar instrução específica
        prompt += f"""Agora, classifique o seguinte texto em uma das categorias acima.

Texto: "{text}"

Responda apenas com o nome da categoria, sem explicações adicionais."""

        return prompt

    def predict(self, texts: list[str]) -> np.ndarray:
        """
        Prediz labels para lista de textos.

        Args:
            texts: Lista de textos para classificar

        Returns:
            Array com labels preditos (como strings)
        """
        if not self.labels:
            raise ValueError("Labels não foram definidos. Configure no construtor.")

        if config.VERBOSE:
            print(f"\n[LLM Classifier] Classificando {len(texts)} textos...")

        predictions = []

        # Processar em lotes para respeitar rate limits
        batch_size = config.LLM_BATCH_SIZE

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            if config.VERBOSE and len(texts) > batch_size:
                print(f"  Processando lote {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

            for text in batch:
                # Criar prompt
                prompt = self._create_prompt(text)

                # Fazer predição com retries
                prediction = self._predict_with_retry(prompt)
                predictions.append(prediction)

                # Pequeno delay entre requisições individuais (evitar burst)
                time.sleep(1)

            # Delay entre lotes para respeitar rate limits (10 req/min = 1 req a cada 6s)
            if i + batch_size < len(texts):
                time.sleep(30)  # Aguardar 30s entre batches de 5 requisições

        return np.array(predictions)

    def _predict_with_retry(self, prompt: str) -> str:
        """
        Faz predição com retries em caso de erro.

        Args:
            prompt: Prompt formatado

        Returns:
            Label predito
        """
        for attempt in range(config.LLM_MAX_RETRIES):
            try:
                # Chamar API com formato de resposta estruturado
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={
                        "response_mime_type": "text/x.enum",
                        "response_schema": self.LabelEnum,
                    },
                )

                # Extrair label
                label = response.text.strip()

                # Validar se o label é válido
                if label in self.labels:
                    return label
                else:
                    # Tentar encontrar label mais próximo (case-insensitive)
                    for valid_label in self.labels:
                        if label.lower() == valid_label.lower():
                            return valid_label

                    # Se não encontrar, usar primeiro label como fallback
                    if config.VERBOSE:
                        print(f"    Aviso: Label inválido '{label}', usando '{self.labels[0]}'")
                    return self.labels[0]

            except Exception as e:
                if attempt < config.LLM_MAX_RETRIES - 1:
                    if config.VERBOSE:
                        print(f"    Erro na tentativa {attempt + 1}: {e}. Tentando novamente...")
                    time.sleep(config.LLM_RETRY_DELAY)
                else:
                    if config.VERBOSE:
                        print(f"    Erro após {config.LLM_MAX_RETRIES} tentativas: {e}")
                    # Retornar primeiro label como fallback
                    return self.labels[0]

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """
        Prediz probabilidades para lista de textos.

        Nota: LLMs não fornecem probabilidades calibradas diretamente.
        Esta implementação retorna probabilidade 1.0 para a classe predita
        e 0.0 para as demais (one-hot encoding).

        Args:
            texts: Lista de textos para classificar

        Returns:
            Array com probabilidades (one-hot encoded)
        """
        predictions = self.predict(texts)

        # Criar one-hot encoding
        probabilities = np.zeros((len(predictions), len(self.labels)))

        for i, pred in enumerate(predictions):
            # Encontrar índice do label predito
            if pred in self.labels:
                label_idx = self.labels.index(pred)
                probabilities[i, label_idx] = 1.0

        return probabilities

    def set_few_shot_examples(self, examples: list[dict[str, str]]) -> None:
        """
        Define exemplos few-shot para melhorar precisão.

        Args:
            examples: Lista de exemplos (formato: [{"text": ..., "label": ...}])
        """
        self.few_shot_examples = examples

        if config.VERBOSE:
            print(f"\n[LLM Classifier] Few-shot examples atualizados: {len(examples)} exemplos")


if __name__ == "__main__":
    # Teste do módulo
    from dotenv import load_dotenv

    # Carregar variáveis de ambiente do .env
    load_dotenv()

    try:
        from ..utils.data_loader import get_few_shot_examples, load_and_prepare_dataset
    except ImportError:
        from utils.data_loader import get_few_shot_examples, load_and_prepare_dataset

    print("Testando LLM Classifier...")

    # Carregar dados
    train_df, test_df, id2label, label2id = load_and_prepare_dataset(max_samples=1000)

    # Extrair few-shot examples
    few_shot = get_few_shot_examples(train_df, id2label, num_examples=2)

    # Criar classificador
    classifier = LLMClassifier(
        labels=list(id2label.values()), few_shot_examples=few_shot
    )

    # Testar com algumas amostras
    test_texts = test_df["text"].head(3).tolist()
    test_labels = test_df["label"].head(3).tolist()

    print("\nRealizando predições...")
    predictions = classifier.predict(test_texts)

    print("\nResultados:")
    for i, (text, true_label, pred_label) in enumerate(
        zip(test_texts, test_labels, predictions)
    ):
        print(f"\n{i+1}. Texto: {text[:80]}...")
        print(f"   Real: {id2label[true_label]} | Predito: {pred_label}")
