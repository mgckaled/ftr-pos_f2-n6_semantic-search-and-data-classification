"""
Módulo para carregamento e preparação de dados.
"""

import random
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Import relativo (quando importado como módulo) ou absoluto (quando executado diretamente)
try:
    from .. import config
except ImportError:
    # Adicionar diretório pai ao path quando executado diretamente
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import config


def load_and_prepare_dataset(
    dataset_name: str | None = None,
    test_size: float | None = None,
    random_state: int | None = None,
    max_samples: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str], dict[str, int]]:
    """
    Carrega e prepara o dataset para classificação.

    Args:
        dataset_name: Nome do dataset do Hugging Face
        test_size: Proporção do conjunto de teste
        random_state: Seed para reprodutibilidade
        max_samples: Número máximo de amostras (None para usar todas)

    Returns:
        train_df: DataFrame de treino
        test_df: DataFrame de teste
        id2label: Mapeamento de índice para label
        label2id: Mapeamento de label para índice
    """
    # Usar valores padrão do config se não fornecidos
    dataset_name = dataset_name or config.DATASET_NAME
    test_size = test_size or config.TEST_SIZE
    random_state = random_state or config.RANDOM_STATE

    if config.VERBOSE:
        print(f"Carregando dataset: {dataset_name}")

    # Carregar dataset do Hugging Face
    dataset = load_dataset(dataset_name, config.DATASET_CONFIG)

    # Extrair splits de treino e teste
    if "train" in dataset and "test" in dataset:
        train_data = dataset["train"]
        test_data = dataset["test"]
    elif "train" in dataset:
        # Se não houver split de teste, criar a partir do treino
        train_data = dataset["train"]
        test_data = None
    else:
        raise ValueError(f"Dataset {dataset_name} não possui split 'train'")

    # Converter para pandas para facilitar manipulação
    train_df = pd.DataFrame(train_data)

    # Determinar colunas de texto e label baseado no dataset
    if dataset_name == "emotion":
        text_column = "text"
        label_column = "label"
    elif dataset_name == "imdb":
        text_column = "text"
        label_column = "label"
    else:
        # Tentar detectar automaticamente
        text_column = "text" if "text" in train_df.columns else train_df.columns[0]
        label_column = "label" if "label" in train_df.columns else train_df.columns[1]

    # Renomear colunas para padronizar
    train_df = train_df.rename(
        columns={text_column: "text", label_column: "label"})

    # Limitar amostras se especificado
    if max_samples is not None and len(train_df) > max_samples:
        train_df = train_df.sample(n=max_samples, random_state=random_state)
        if config.VERBOSE:
            print(f"Limitando dataset a {max_samples} amostras")

    # Criar mapeamentos de labels
    unique_labels = sorted(train_df["label"].unique())
    id2label = {i: label for i, label in enumerate(unique_labels)}
    label2id = {label: i for i, label in id2label.items()}

    # Se houver nomes de labels no dataset original, usar eles
    if hasattr(train_data.features["label"], "names"):
        label_names = train_data.features["label"].names
        id2label = {i: name for i, name in enumerate(label_names)}
        label2id = {name: i for i, name in enumerate(label_names)}

    # Criar split de teste se não existir
    if test_data is None:
        train_df, test_df = train_test_split(
            train_df, test_size=test_size, random_state=random_state, stratify=train_df[
                "label"]
        )
        if config.VERBOSE:
            print(f"Criando split de teste: {test_size*100:.0f}%")
    else:
        test_df = pd.DataFrame(test_data)
        test_df = test_df.rename(
            columns={text_column: "text", label_column: "label"})

    # Resetar índices
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    if config.VERBOSE:
        print(f"\nDataset carregado com sucesso!")
        print(f"Treino: {len(train_df)} amostras")
        print(f"Teste: {len(test_df)} amostras")
        print(f"Classes: {list(id2label.values())}")
        print(f"Distribuição de classes (treino):")
        print(train_df["label"].value_counts().sort_index())

    return train_df, test_df, id2label, label2id


def preprocess_text(text: str) -> str:
    """
    Pré-processa texto básico (opcional).

    Args:
        text: Texto original

    Returns:
        Texto processado
    """
    # Remover espaços extras
    text = " ".join(text.split())

    # Remover quebras de linha
    text = text.replace("\n", " ").replace("\r", " ")

    return text


def get_few_shot_examples(
    train_df: pd.DataFrame, id2label: dict[int, str], num_examples: int = 3
) -> list[dict[str, str]]:
    """
    Extrai exemplos few-shot balanceados para uso com LLMs.

    Args:
        train_df: DataFrame de treino
        id2label: Mapeamento de índice para label
        num_examples: Número de exemplos por classe

    Returns:
        Lista de exemplos com formato {"text": ..., "label": ...}
    """
    examples = []

    for label_id, label_name in id2label.items():
        # Filtrar exemplos da classe
        class_examples = train_df[train_df["label"] == label_id]

        # Selecionar amostras aleatórias
        sampled = class_examples.sample(
            n=min(num_examples, len(class_examples)), random_state=config.RANDOM_STATE
        )

        for _, row in sampled.iterrows():
            examples.append({"text": row["text"], "label": label_name})

    # Embaralhar exemplos
    random.seed(config.RANDOM_STATE)
    random.shuffle(examples)

    return examples


def get_class_distribution(df: pd.DataFrame, id2label: dict[int, str]) -> dict[str, int]:
    """
    Retorna a distribuição de classes do dataset.

    Args:
        df: DataFrame com coluna 'label'
        id2label: Mapeamento de índice para label

    Returns:
        Dicionário com contagem por classe
    """
    counts = df["label"].value_counts().to_dict()
    distribution = {id2label.get(label_id, str(
        label_id)): count for label_id, count in counts.items()}

    return dict(sorted(distribution.items()))


if __name__ == "__main__":
    # Teste do módulo
    train_df, test_df, id2label, label2id = load_and_prepare_dataset()

    print("\nExemplo de dados:")
    print(train_df.head())

    print("\nMapeamentos:")
    print(f"id2label: {id2label}")
    print(f"label2id: {label2id}")

    print("\nExemplos few-shot:")
    few_shot = get_few_shot_examples(train_df, id2label, num_examples=2)
    for ex in few_shot[:5]:
        print(f"- Texto: {ex['text'][:50]}... | Label: {ex['label']}")

    print("\nDistribuição de classes:")
    print(get_class_distribution(train_df, id2label))
