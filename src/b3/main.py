"""
Script principal para comparação de abordagens de classificação com IA.

Este script executa três abordagens de classificação:
1. Embeddings pré-treinados + KNN
2. Fine-tuning de DistilBERT
3. LLM (Gemini) com prompt engineering

E compara as métricas de cada uma.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from dotenv import load_dotenv

try:
    import config
    from models.embedding_classifier import EmbeddingClassifier
    from models.finetuned_classifier import FinetunedClassifier
    from models.llm_classifier import LLMClassifier
    from utils.data_loader import (
        get_class_distribution,
        get_few_shot_examples,
        load_and_prepare_dataset,
    )
    from utils.metrics import (
        calculate_metrics,
        compare_classifiers,
        measure_inference_time,
        print_comparison_table,
        print_metrics_summary,
    )
    from utils.visualization import (
        plot_class_distribution,
        plot_confusion_matrix,
        plot_inference_time_comparison,
        plot_metrics_comparison,
    )
except ImportError:
    # Se executado diretamente, adicionar diretório ao path
    sys.path.insert(0, str(Path(__file__).parent))
    import config
    from models.embedding_classifier import EmbeddingClassifier
    from models.finetuned_classifier import FinetunedClassifier
    from models.llm_classifier import LLMClassifier
    from utils.data_loader import (
        get_class_distribution,
        get_few_shot_examples,
        load_and_prepare_dataset,
    )
    from utils.metrics import (
        calculate_metrics,
        compare_classifiers,
        measure_inference_time,
        print_comparison_table,
        print_metrics_summary,
    )
    from utils.visualization import (
        plot_class_distribution,
        plot_confusion_matrix,
        plot_inference_time_comparison,
        plot_metrics_comparison,
    )


def main():
    """Função principal do experimento."""

    # Carregar variáveis de ambiente
    load_dotenv()

    print("=" * 80)
    print(" Mini Projeto: Classificação de Dados com IA - Bloco C")
    print("=" * 80)
    print("\nComparando três abordagens:")
    print("  1. Embeddings pré-treinados + KNN (Abordagem 'ingênua')")
    print("  2. Fine-tuning de DistilBERT")
    print("  3. LLM (Gemini) com prompt engineering")
    print("\n" + "=" * 80)

    # =========================================================================
    # ETAPA 1: CARREGAR E PREPARAR DADOS
    # =========================================================================
    print("\n\n[ETAPA 1] Carregando e preparando dados...")
    print("-" * 80)

    train_df, test_df, id2label, label2id = load_and_prepare_dataset(
        dataset_name=config.DATASET_NAME,
        test_size=config.TEST_SIZE,
        max_samples=config.MAX_SAMPLES,
    )

    # Atualizar config com mapeamentos
    config.LABEL2ID = label2id
    config.ID2LABEL = id2label

    # Visualizar distribuição de classes
    train_distribution = get_class_distribution(train_df, id2label)
    test_distribution = get_class_distribution(test_df, id2label)

    fig_train_dist = plot_class_distribution(
        train_distribution,
        title="Distribuição de Classes - Treino",
        save_path=str(config.RESULTS_DIR / "class_distribution_train.png"),
    )
    plt.close(fig_train_dist)

    fig_test_dist = plot_class_distribution(
        test_distribution,
        title="Distribuição de Classes - Teste",
        save_path=str(config.RESULTS_DIR / "class_distribution_test.png"),
    )
    plt.close(fig_test_dist)

    # Preparar dados
    X_train = train_df["text"].tolist()
    y_train = train_df["label"].tolist()
    X_test = test_df["text"].tolist()
    y_test = test_df["label"].tolist()

    # Extrair few-shot examples para LLM
    few_shot_examples = get_few_shot_examples(
        train_df, id2label, num_examples=config.NUM_FEW_SHOT_EXAMPLES
    )
    config.FEW_SHOT_EXAMPLES = few_shot_examples

    # =========================================================================
    # ETAPA 2: ABORDAGEM 1 - EMBEDDINGS + KNN
    # =========================================================================
    print("\n\n[ETAPA 2] Treinando Abordagem 1: Embeddings + KNN...")
    print("-" * 80)

    embedding_clf = EmbeddingClassifier(
        model_name=config.EMBEDDING_MODEL_NAME,
        k_neighbors=config.K_NEIGHBORS,
        metric=config.KNN_METRIC,
    )

    embedding_clf.fit(X_train, y_train)

    print("\n  Realizando predições no conjunto de teste...")
    y_pred_embedding = embedding_clf.predict(X_test)
    y_pred_proba_embedding = embedding_clf.predict_proba(X_test)

    print("  Medindo tempo de inferência...")
    time_embedding = measure_inference_time(embedding_clf, X_test[:100], num_runs=3)

    print("  Calculando métricas...")
    metrics_embedding = calculate_metrics(
        y_test, y_pred_embedding, y_pred_proba_embedding, list(id2label.values())
    )

    print_metrics_summary(metrics_embedding, "Embedding + KNN", list(id2label.values()))

    # Salvar matriz de confusão
    fig_cm_embedding = plot_confusion_matrix(
        y_test,
        y_pred_embedding,
        list(id2label.values()),
        "Embedding + KNN",
        save_path=str(config.RESULTS_DIR / "confusion_matrix_embedding.png"),
    )
    plt.close(fig_cm_embedding)

    # =========================================================================
    # ETAPA 3: ABORDAGEM 2 - FINE-TUNING DISTILBERT
    # =========================================================================
    print("\n\n[ETAPA 3] Treinando Abordagem 2: Fine-tuning DistilBERT...")
    print("-" * 80)

    finetuned_clf = FinetunedClassifier(
        model_name=config.FINETUNED_MODEL_NAME,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )

    # Verificar se já existe modelo treinado
    if config.FINETUNED_MODEL_PATH.exists():
        print("\n  Modelo previamente treinado encontrado. Deseja carregar?")
        print("  1 - Carregar modelo existente")
        print("  2 - Treinar novo modelo")

        # Para execução automática, sempre treinar novo modelo
        # Em produção, você pode adicionar input() aqui
        choice = "2"  # Altere para "1" se quiser carregar modelo existente

        if choice == "1":
            finetuned_clf.load_model()
        else:
            finetuned_clf.train(X_train, y_train, epochs=config.EPOCHS)
            finetuned_clf.save_model()
    else:
        finetuned_clf.train(X_train, y_train, epochs=config.EPOCHS)
        finetuned_clf.save_model()

    print("\n  Realizando predições no conjunto de teste...")
    y_pred_finetuned = finetuned_clf.predict(X_test)
    y_pred_proba_finetuned = finetuned_clf.predict_proba(X_test)

    print("  Medindo tempo de inferência...")
    time_finetuned = measure_inference_time(finetuned_clf, X_test[:100], num_runs=3)

    print("  Calculando métricas...")
    metrics_finetuned = calculate_metrics(
        y_test, y_pred_finetuned, y_pred_proba_finetuned, list(id2label.values())
    )

    print_metrics_summary(metrics_finetuned, "Fine-tuned DistilBERT", list(id2label.values()))

    # Salvar matriz de confusão
    fig_cm_finetuned = plot_confusion_matrix(
        y_test,
        y_pred_finetuned,
        list(id2label.values()),
        "Fine-tuned DistilBERT",
        save_path=str(config.RESULTS_DIR / "confusion_matrix_finetuned.png"),
    )
    plt.close(fig_cm_finetuned)

    # =========================================================================
    # ETAPA 4: ABORDAGEM 3 - LLM (GEMINI)
    # =========================================================================
    print("\n\n[ETAPA 4] Executando Abordagem 3: LLM (Gemini)...")
    print("-" * 80)

    llm_clf = LLMClassifier(
        model_name=config.LLM_MODEL_NAME,
        labels=list(id2label.values()),
        few_shot_examples=few_shot_examples,
    )

    print("\n  Realizando predições no conjunto de teste...")
    print("  (Isso pode demorar devido a chamadas de API...)")

    # Para não estourar rate limits, usar apenas uma amostra do teste
    # Em produção, você pode processar todo o conjunto
    test_sample_size = min(100, len(X_test))
    X_test_sample = X_test[:test_sample_size]
    y_test_sample = y_test[:test_sample_size]

    y_pred_llm = llm_clf.predict(X_test_sample)
    y_pred_proba_llm = llm_clf.predict_proba(X_test_sample)

    print("  Medindo tempo de inferência...")
    time_llm = measure_inference_time(llm_clf, X_test_sample[:20], num_runs=1)

    print("  Calculando métricas...")
    metrics_llm = calculate_metrics(
        y_test_sample, y_pred_llm, y_pred_proba_llm, list(id2label.values())
    )

    print_metrics_summary(metrics_llm, "LLM (Gemini)", list(id2label.values()))

    # Salvar matriz de confusão
    fig_cm_llm = plot_confusion_matrix(
        y_test_sample,
        y_pred_llm,
        list(id2label.values()),
        "LLM (Gemini)",
        save_path=str(config.RESULTS_DIR / "confusion_matrix_llm.png"),
    )
    plt.close(fig_cm_llm)

    # =========================================================================
    # ETAPA 5: COMPARAÇÃO FINAL
    # =========================================================================
    print("\n\n[ETAPA 5] Comparação Final entre Abordagens...")
    print("-" * 80)

    # Preparar resultados para comparação
    results = {
        "Embedding + KNN": metrics_embedding,
        "Fine-tuned DistilBERT": metrics_finetuned,
        "LLM (Gemini)": metrics_llm,
    }

    # Comparar métricas
    comparison = compare_classifiers(results)
    print_comparison_table(comparison)

    # Plot comparativo de métricas
    fig_comparison = plot_metrics_comparison(
        comparison,
        metrics=["accuracy", "precision", "recall", "f1"],
        save_path=str(config.RESULTS_DIR / "metrics_comparison.png"),
    )
    plt.close(fig_comparison)

    # Plot comparativo de tempo de inferência
    times_dict = {
        "Embedding + KNN": time_embedding["mean_time"],
        "Fine-tuned DistilBERT": time_finetuned["mean_time"],
        "LLM (Gemini)": time_llm["mean_time"],
    }

    fig_time = plot_inference_time_comparison(
        times_dict, save_path=str(config.RESULTS_DIR / "inference_time_comparison.png")
    )
    plt.close(fig_time)

    # =========================================================================
    # CONCLUSÃO
    # =========================================================================
    print("\n\n" + "=" * 80)
    print(" EXPERIMENTO CONCLUÍDO!")
    print("=" * 80)
    print(f"\nResultados salvos em: {config.RESULTS_DIR}")
    print("\nArquivos gerados:")
    print("  - Matrizes de confusão (3 arquivos)")
    print("  - Comparação de métricas")
    print("  - Comparação de tempo de inferência")
    print("  - Distribuição de classes")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
