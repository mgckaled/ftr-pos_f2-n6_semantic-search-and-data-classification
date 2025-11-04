"""
Módulo para cálculo de métricas de avaliação de classificadores.
"""

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from .. import config
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import config


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray | None = None,
    labels: list[str] | None = None,
    average: str | None = None,
) -> dict[str, float | np.ndarray]:
    """
    Calcula métricas de classificação.

    Args:
        y_true: Labels verdadeiros
        y_pred: Labels preditos
        y_pred_proba: Probabilidades preditas (para ROC-AUC)
        labels: Nomes dos labels
        average: Método de averaging ('micro', 'macro', 'weighted')

    Returns:
        Dicionário com todas as métricas
    """
    average = average or config.AVERAGE_METHOD

    metrics = {}

    # Accuracy
    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    # Precision, Recall, F1 (macro, micro, weighted)
    metrics["precision"] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, average=average, zero_division=0)

    # Métricas por classe
    metrics["precision_per_class"] = precision_score(
        y_true, y_pred, average=None, zero_division=0
    )
    metrics["recall_per_class"] = recall_score(
        y_true, y_pred, average=None, zero_division=0
    )
    metrics["f1_per_class"] = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Confusion Matrix
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)

    # ROC-AUC (apenas se probabilidades forem fornecidas)
    if y_pred_proba is not None:
        try:
            # Para classificação multiclasse, usar one-vs-rest
            num_classes = len(np.unique(y_true))

            if num_classes == 2:
                # Binário: usar probabilidade da classe positiva
                metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:
                # Multiclasse: usar one-vs-rest com averaging
                metrics["roc_auc"] = roc_auc_score(
                    y_true, y_pred_proba, multi_class="ovr", average=average
                )

                # ROC-AUC por classe
                metrics["roc_auc_per_class"] = roc_auc_score(
                    y_true, y_pred_proba, multi_class="ovr", average=None
                )
        except Exception as e:
            if config.VERBOSE:
                print(f"  Aviso: Não foi possível calcular ROC-AUC: {e}")
            metrics["roc_auc"] = None

    return metrics


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
    output_dict: bool = False,
) -> str | dict:
    """
    Gera relatório de classificação completo.

    Args:
        y_true: Labels verdadeiros
        y_pred: Labels preditos
        labels: Nomes dos labels
        output_dict: Se True, retorna dicionário ao invés de string

    Returns:
        Relatório de classificação
    """
    report = classification_report(
        y_true, y_pred, target_names=labels, output_dict=output_dict, zero_division=0
    )

    return report


def measure_inference_time(
    classifier, texts: list[str], num_runs: int = 3
) -> dict[str, float]:
    """
    Mede tempo de inferência de um classificador.

    Args:
        classifier: Classificador com método predict()
        texts: Lista de textos para testar
        num_runs: Número de execuções para média

    Returns:
        Dicionário com estatísticas de tempo
    """
    times = []

    for _ in range(num_runs):
        start_time = time.time()
        _ = classifier.predict(texts)
        elapsed_time = time.time() - start_time
        times.append(elapsed_time)

    return {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
        "time_per_sample": np.mean(times) / len(texts),
    }


def compare_classifiers(results: dict[str, dict]) -> dict:
    """
    Compara métricas de múltiplos classificadores.

    Args:
        results: Dicionário com resultados de cada classificador
                 formato: {"nome_classificador": {"accuracy": ..., "f1": ..., ...}}

    Returns:
        Dicionário com comparação formatada
    """
    comparison = {}

    # Métricas a comparar
    metrics_to_compare = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    for metric in metrics_to_compare:
        comparison[metric] = {}

        for classifier_name, metrics_dict in results.items():
            if metric in metrics_dict and metrics_dict[metric] is not None:
                comparison[metric][classifier_name] = metrics_dict[metric]

    # Encontrar melhor classificador por métrica
    best_classifiers = {}

    for metric, values in comparison.items():
        if values:
            best_classifier = max(values, key=values.get)
            best_classifiers[metric] = {
                "classifier": best_classifier,
                "value": values[best_classifier],
            }

    comparison["best_per_metric"] = best_classifiers

    return comparison


def print_metrics_summary(
    metrics: dict, classifier_name: str, labels: list[str] | None = None
) -> None:
    """
    Imprime resumo das métricas de forma formatada.

    Args:
        metrics: Dicionário com métricas
        classifier_name: Nome do classificador
        labels: Nomes dos labels
    """
    print(f"\n{'='*60}")
    print(f" Métricas - {classifier_name}")
    print(f"{'='*60}")

    print(f"\nMétricas Gerais:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")

    if metrics.get("roc_auc") is not None:
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

    # Métricas por classe
    if labels and "precision_per_class" in metrics:
        print(f"\nMétricas por Classe:")
        print(f"  {'Classe':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print(f"  {'-'*51}")

        for i, label in enumerate(labels):
            prec = metrics["precision_per_class"][i]
            rec = metrics["recall_per_class"][i]
            f1 = metrics["f1_per_class"][i]
            print(f"  {label:<15} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}")

    print(f"\n{'='*60}\n")


def print_comparison_table(comparison: dict) -> None:
    """
    Imprime tabela comparativa entre classificadores.

    Args:
        comparison: Dicionário com comparação (retorno de compare_classifiers)
    """
    print(f"\n{'='*70}")
    print(" Comparação entre Classificadores")
    print(f"{'='*70}\n")

    # Obter nomes dos classificadores
    first_metric = next(iter(comparison.values()))
    if isinstance(first_metric, dict):
        classifiers = list(first_metric.keys())
    else:
        return

    # Cabeçalho da tabela
    header = f"{'Métrica':<15}"
    for clf in classifiers:
        header += f" {clf:<20}"
    print(header)
    print("-" * len(header))

    # Linhas da tabela
    metrics_to_show = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    for metric in metrics_to_show:
        if metric in comparison and comparison[metric]:
            row = f"{metric.capitalize():<15}"

            for clf in classifiers:
                value = comparison[metric].get(clf, None)
                if value is not None:
                    row += f" {value:<20.4f}"
                else:
                    row += f" {'N/A':<20}"

            print(row)

    # Melhor classificador por métrica
    if "best_per_metric" in comparison:
        print(f"\n{'='*70}")
        print(" Melhor Classificador por Métrica")
        print(f"{'='*70}\n")

        for metric, best_info in comparison["best_per_metric"].items():
            print(
                f"  {metric.capitalize():<15}: {best_info['classifier']} ({best_info['value']:.4f})"
            )

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    # Teste do módulo
    print("Testando módulo de métricas...")

    # Dados de exemplo
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 1, 2, 0, 2, 2])
    y_pred_proba = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.2, 0.3, 0.5],
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.1, 0.1, 0.8],
            [0.85, 0.1, 0.05],
            [0.2, 0.2, 0.6],
            [0.05, 0.1, 0.85],
        ]
    )

    labels = ["Class A", "Class B", "Class C"]

    # Calcular métricas
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba, labels)

    # Imprimir resumo
    print_metrics_summary(metrics, "Classificador de Teste", labels)

    # Gerar relatório
    report = generate_classification_report(y_true, y_pred, labels)
    print("Relatório de Classificação:")
    print(report)

    # Teste de comparação
    results = {
        "Classifier A": {"accuracy": 0.85, "precision": 0.83, "recall": 0.82, "f1": 0.82, "roc_auc": 0.90},
        "Classifier B": {"accuracy": 0.88, "precision": 0.86, "recall": 0.85, "f1": 0.85, "roc_auc": 0.92},
        "Classifier C": {"accuracy": 0.82, "precision": 0.80, "recall": 0.79, "f1": 0.80, "roc_auc": 0.87},
    }

    comparison = compare_classifiers(results)
    print_comparison_table(comparison)
