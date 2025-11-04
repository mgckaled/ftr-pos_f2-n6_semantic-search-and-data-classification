# Bloco C - Classificação de Dados

## Aula 1 - Classificação com IA

Exploramos a diferença entre classificação clássica e classificação com IA. Discutimos como os modelos clássicos exigem treinamento do zero, enquanto os modelos de IA, ou foundational models, já vêm pré-treinados e prontos para inferência. Abordei a importância da generalização desses modelos, que são treinados com grandes volumes de dados. Também mencionei que, mesmo com a ascensão da IA, os modelos clássicos ainda têm seu espaço, especialmente em dados estruturados.

## Aula 2 - Avaliação de sistemas de classificação

Discutimos a importância da avaliação de sistemas de classificação, especialmente em contextos como reconhecimento facial. Abordamos como garantir que um modelo realmente funcione, utilizando um conjunto de testes rotulados e a matriz de confusão para identificar acertos e erros. Falei sobre métricas como acurácia, precisão e recall, além da curva ROC e a métrica AUC, que ajudam a entender a performance do modelo. Essas ferramentas são essenciais para validar e comparar sistemas de classificação

## Aula 3 - Abordagens de classificação

Exploramos como realizar classificação com IA na prática, abordando três métodos principais. O primeiro envolve o uso de embeddings pré-treinados, onde um modelo gera um embedding que pode ser interpretado por qualquer sistema, não necessariamente uma IA. O segundo método é o ajuste de modelos pré-treinados, onde fazemos fine-tuning para especializar o modelo na tarefa desejada. Por fim, falamos sobre modelos de LLM, como o ChatGPT, que já têm um vasto conhecimento e podem classificar informações. Também menciono uma quarta abordagem, mas com custos elevados.

## Aula 4 - Exemplos de sistemas de classificação

Abordamos a classificação com embeddings, que chamei de "ingênua" por ser uma técnica simples, mas eficaz. Vamos encontrar os K vizinhos mais próximos de uma instância e determinar a classe majoritária entre eles. Também discutimos o fine-tuning de modelos pré-treinados, como o BERT, e como ajustar um modelo para tarefas específicas. Por fim, falamos sobre o uso de LLMs, como o ChatGPT, e técnicas de prompt engineering para garantir classificações precisas.
