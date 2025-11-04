"""
Módulo para visualização de resultados de classificação.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_curve
from sklearn.preprocessing import label_binarize

try:
    from .. import config
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import config


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    classifier_name: str = "Classifier",
    save_path: str | None = None,
    normalize: str | None = None,
) -> plt.Figure:
    """
    Plota matriz de confusão.

    Args:
        y_true: Labels verdadeiros
        y_pred: Labels preditos
        labels: Nomes dos labels
        classifier_name: Nome do classificador
        save_path: Caminho para salvar a figura
        normalize: 'true', 'pred', 'all' ou None

    Returns:
        Figura matplotlib
    """
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE, dpi=config.DPI)

    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=labels,
        normalize=normalize,
        cmap="Blues",
        ax=ax,
        colorbar=True,
    )

    title = f"Matriz de Confusão - {classifier_name}"
    if normalize:
        title += f" (normalizada por {normalize})"

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Predito", fontsize=12)
    ax.set_ylabel("Verdadeiro", fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format=config.SAVE_FORMAT, dpi=config.DPI, bbox_inches="tight")
        if config.VERBOSE:
            print(f"  Figura salva em: {save_path}")

    return fig


def plot_metrics_comparison(
    comparison: dict,
    metrics: list[str] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plota gráfico de barras comparando métricas entre classificadores.

    Args:
        comparison: Dicionário com comparação (retorno de compare_classifiers)
        metrics: Lista de métricas a plotar
        save_path: Caminho para salvar a figura

    Returns:
        Figura matplotlib
    """
    metrics = metrics or ["accuracy", "precision", "recall", "f1"]

    # Filtrar métricas disponíveis
    available_metrics = [m for m in metrics if m in comparison and comparison[m]]

    if not available_metrics:
        if config.VERBOSE:
            print("  Aviso: Nenhuma métrica disponível para plotar")
        return None

    # Preparar dados
    classifiers = list(comparison[available_metrics[0]].keys())
    num_metrics = len(available_metrics)
    num_classifiers = len(classifiers)

    # Criar figura
    fig, ax = plt.subplots(figsize=(12, 6), dpi=config.DPI)

    # Configurar barras
    x = np.arange(num_metrics)
    width = 0.8 / num_classifiers

    # Plotar barras para cada classificador
    for i, classifier in enumerate(classifiers):
        values = [comparison[metric].get(classifier, 0) for metric in available_metrics]

        offset = (i - num_classifiers / 2) * width + width / 2
        bars = ax.bar(x + offset, values, width, label=classifier, alpha=0.8)

        # Adicionar valores nas barras
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Configurar eixos e título
    ax.set_xlabel("Métrica", fontsize=12, fontweight="bold")
    ax.set_ylabel("Valor", fontsize=12, fontweight="bold")
    ax.set_title("Comparação de Métricas entre Classificadores", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in available_metrics])
    ax.legend(loc="lower right")
    ax.set_ylim([0, 1.1])
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format=config.SAVE_FORMAT, dpi=config.DPI, bbox_inches="tight")
        if config.VERBOSE:
            print(f"  Figura salva em: {save_path}")

    return fig


def plot_class_distribution(
    labels_count: dict[str, int],
    title: str = "Distribuição de Classes",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plota distribuição de classes.

    Args:
        labels_count: Dicionário com contagem por label
        title: Título do gráfico
        save_path: Caminho para salvar a figura

    Returns:
        Figura matplotlib
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=config.DPI)

    labels = list(labels_count.keys())
    counts = list(labels_count.values())

    colors = sns.color_palette("husl", len(labels))
    bars = ax.bar(labels, counts, color=colors, alpha=0.8)

    # Adicionar valores nas barras
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel("Classe", fontsize=12, fontweight="bold")
    ax.set_ylabel("Contagem", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format=config.SAVE_FORMAT, dpi=config.DPI, bbox_inches="tight")
        if config.VERBOSE:
            print(f"  Figura salva em: {save_path}")

    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    y_pred_proba_dict: dict[str, np.ndarray],
    labels: list[str],
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plota curvas ROC para múltiplos classificadores.

    Args:
        y_true: Labels verdadeiros
        y_pred_proba_dict: Dicionário com probabilidades de cada classificador
        labels: Nomes dos labels
        save_path: Caminho para salvar a figura

    Returns:
        Figura matplotlib
    """
    num_classes = len(labels)

    # Para classificação binária
    if num_classes == 2:
        fig, ax = plt.subplots(figsize=(8, 8), dpi=config.DPI)

        for classifier_name, y_pred_proba in y_pred_proba_dict.items():
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            ax.plot(fpr, tpr, label=classifier_name, linewidth=2)

        ax.plot([0, 1], [0, 1], "k--", label="Random")
        ax.set_xlabel("Taxa de Falsos Positivos", fontsize=12)
        ax.set_ylabel("Taxa de Verdadeiros Positivos", fontsize=12)
        ax.set_title("Curvas ROC", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)

    # Para classificação multiclasse (one-vs-rest)
    else:
        fig, axes = plt.subplots(
            1, num_classes, figsize=(6 * num_classes, 5), dpi=config.DPI
        )

        if num_classes == 1:
            axes = [axes]

        # Binarizar labels
        y_true_bin = label_binarize(y_true, classes=range(num_classes))

        for class_idx, (ax, label) in enumerate(zip(axes, labels)):
            for classifier_name, y_pred_proba in y_pred_proba_dict.items():
                fpr, tpr, _ = roc_curve(
                    y_true_bin[:, class_idx], y_pred_proba[:, class_idx]
                )
                ax.plot(fpr, tpr, label=classifier_name, linewidth=2)

            ax.plot([0, 1], [0, 1], "k--", label="Random")
            ax.set_xlabel("FPR", fontsize=10)
            ax.set_ylabel("TPR", fontsize=10)
            ax.set_title(f"ROC: {label}", fontsize=12, fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format=config.SAVE_FORMAT, dpi=config.DPI, bbox_inches="tight")
        if config.VERBOSE:
            print(f"  Figura salva em: {save_path}")

    return fig


def plot_inference_time_comparison(
    times_dict: dict[str, float],
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plota comparação de tempo de inferência entre classificadores.

    Args:
        times_dict: Dicionário com tempos {classificador: tempo_medio}
        save_path: Caminho para salvar a figura

    Returns:
        Figura matplotlib
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=config.DPI)

    classifiers = list(times_dict.keys())
    times = list(times_dict.values())

    colors = sns.color_palette("muted", len(classifiers))
    bars = ax.barh(classifiers, times, color=colors, alpha=0.8)

    # Adicionar valores nas barras
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2.0,
            f"{width:.3f}s",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Tempo (segundos)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Classificador", fontsize=12, fontweight="bold")
    ax.set_title("Comparação de Tempo de Inferência", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format=config.SAVE_FORMAT, dpi=config.DPI, bbox_inches="tight")
        if config.VERBOSE:
            print(f"  Figura salva em: {save_path}")

    return fig


if __name__ == "__main__":
    # Teste do módulo
    print("Testando módulo de visualização...")

    # Dados de exemplo
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 1, 2, 0, 2, 2, 0, 0, 2])
    labels = ["Class A", "Class B", "Class C"]

    # Teste: Matriz de confusão
    print("\nTestando matriz de confusão...")
    fig1 = plot_confusion_matrix(y_true, y_pred, labels, "Teste")
    plt.show()

    # Teste: Comparação de métricas
    print("\nTestando comparação de métricas...")
    comparison = {
        "accuracy": {"Clf A": 0.85, "Clf B": 0.88, "Clf C": 0.82},
        "precision": {"Clf A": 0.83, "Clf B": 0.86, "Clf C": 0.80},
        "recall": {"Clf A": 0.82, "Clf B": 0.85, "Clf C": 0.79},
        "f1": {"Clf A": 0.82, "Clf B": 0.85, "Clf C": 0.80},
    }

    fig2 = plot_metrics_comparison(comparison)
    plt.show()

    # Teste: Distribuição de classes
    print("\nTestando distribuição de classes...")
    labels_count = {"Class A": 100, "Class B": 150, "Class C": 80}
    fig3 = plot_class_distribution(labels_count)
    plt.show()

    # Teste: Tempo de inferência
    print("\nTestando comparação de tempo...")
    times_dict = {"Embedding KNN": 0.5, "Fine-tuned BERT": 2.3, "LLM Gemini": 5.7}
    fig4 = plot_inference_time_comparison(times_dict)
    plt.show()

    print("\nTodos os testes concluídos!")
