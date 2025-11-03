# Aulas Tóricas - Resumos

## Índice

- [Aulas Tóricas - Resumos](#aulas-tóricas---resumos)
  - [Índice](#índice)
  - [Bloco A - Embeddings e busca semântica](#bloco-a---embeddings-e-busca-semântica)
    - [Aula 1 - Introdução à busca semântica e classificação](#aula-1---introdução-à-busca-semântica-e-classificação)
    - [Aula 2 - Introdução a embeddings](#aula-2---introdução-a-embeddings)
    - [Aula 3 - Embeddings como vetores](#aula-3---embeddings-como-vetores)
    - [Aula 4 - Características de embeddings](#aula-4---características-de-embeddings)
    - [Aula 5 - Embeddings ingênuos](#aula-5---embeddings-ingênuos)
    - [Aula 6 - Gerando embeddings com redes neurais](#aula-6---gerando-embeddings-com-redes-neurais)
    - [Aula 7 - Similaridade de embeddings](#aula-7---similaridade-de-embeddings)
  - [Bloco B - Busca Semântica](#bloco-b---busca-semântica)
    - [Aula 1 - Busca semântica com embeddings](#aula-1---busca-semântica-com-embeddings)
    - [Aula 2 - Grafos de conhecimento](#aula-2---grafos-de-conhecimento)
    - [Aula 3 - Avaliação de busca semântica](#aula-3---avaliação-de-busca-semântica)
    - [Aula 4 - Bancos de dados de vetores](#aula-4---bancos-de-dados-de-vetores)
  - [Bloco C - Classificação de Dados](#bloco-c---classificação-de-dados)
    - [Aula 1 - Classificação com IA](#aula-1---classificação-com-ia)
    - [Aula 2 - Avaliação de sistemas de classificação](#aula-2---avaliação-de-sistemas-de-classificação)
    - [Aula 3 - Abordagens de classificação](#aula-3---abordagens-de-classificação)
    - [Aula 4 - Exemplos de sistemas de classificação](#aula-4---exemplos-de-sistemas-de-classificação)

## Bloco A - Embeddings e busca semântica

### Aula 1 - Introdução à busca semântica e classificação

Exploramos os conceitos de busca semântica e classificação. A busca semântica envolve encontrar objetos semelhantes em significado, como textos ou imagens, enquanto a classificação atribui uma categoria a um objeto. Por exemplo, ao buscar "animais", queremos resultados que incluam gatos e cachorros, não apenas palavras semelhantes. Já na classificação, como em reconhecimento facial ou identificação de spam, o computador precisa entender o contexto para categorizar corretamente. Esses conceitos são fundamentais em diversas aplicações do dia a dia.

### Aula 2 - Introdução a embeddings

Exploramos como representar objetos do mundo real para que os computadores possam entendê-los. Discutimos a limitação dos computadores em "ver" e "ouvir" como nós, e a importância de usar embeddings, que são vetores numéricos que mapeiam objetos em espaços n-dimensionais. Esses vetores permitem que o computador reconheça semelhanças e diferenças entre palavras e imagens, mesmo que essa representação não tenha significado direto para nós.

### Aula 3 - Embeddings como vetores

Exploramos o conceito de embeddings, comparando-os a vetores em um espaço n-dimensional. Usei um exemplo em duas dimensões para ilustrar como diferentes objetos, como animais e palavras, podem ser representados como setas em um plano cartesiano. Também mencionei um site da Carnegie Mellon University, onde é possível visualizar embeddings em três dimensões, permitindo interagir com palavras e observar suas relações de gênero e idade. <https://www.cs.cmu.edu/~dst/WordEmbeddingDemo>

### Aula 4 - Características de embeddings

Exploramos o conceito de embeddings usando cidades do mundo e suas temperaturas médias em janeiro e julho. Mostrei como essas cidades se agrupam em um gráfico bidimensional, revelando diferenças climáticas. Abordei as limitações de criar embeddings manualmente e a importância de representações densas, que são mais eficientes. Também discuti a necessidade de que cada objeto tenha uma representação única, apesar de possíveis ambiguidades. O objetivo é entender como embeddings podem capturar a semântica de forma mais eficaz.

### Aula 5 - Embeddings ingênuos

Discutimos algumas abordagens ingênuas para criar embeddings, começando pela representação one-hot. Embora pareça uma boa ideia, essa técnica resulta em vetores esparsos e grandes, sem semântica real. Exploramos como essa representação não captura a proximidade entre palavras ou objetos, levando a distâncias iguais entre termos sem relação. Também abordamos a contagem de entidades e a conversão de dados em bits, que não resolvem o problema de representação semântica. Vamos explorar alternativas mais eficazes em aulas futuras.

### Aula 6 - Gerando embeddings com redes neurais

Exploramos como criar embeddings significativos a partir de redes neurais, um conceito que já existe há mais de 20 anos. Discuti como as redes transformam inputs, como texto ou imagens, em representações numéricas que capturam seu significado. Apresentei dois exemplos clássicos: o Word2Vec, que gera embeddings para palavras com base em seus contextos, e os autoencoders, que reduzem a dimensionalidade de imagens para criar representações densas. Esses métodos são fundamentais para entender a evolução das redes neurais.

### Aula 7 - Similaridade de embeddings

Explicamos como calcular a similaridade entre embeddings, que são representações vetoriais de objetos, como gatos e cachorros. Vamos explorar três métodos principais: a distância euclidiana, a similaridade de cossenos e o produto escalar. Cada um tem suas particularidades e aplicações, dependendo do contexto dos embeddings. A similaridade de cossenos é ideal para tarefas de texto, enquanto a distância euclidiana pode ser mais adequada em outras situações.

## Bloco B - Busca Semântica

### Aula 1 - Busca semântica com embeddings

Exploramos o conceito de embeddings e sua importância na busca semântica. Discutimos como esses vetores carregam informações semânticas, permitindo buscas mais significativas. Para implementar uma busca semântica, precisamos de uma base de conhecimento e um mecanismo de busca. Abordamos a criação de embeddings para documentos relevantes e a comparação deles com a query usando métricas de similaridade. Por fim, enfatizamos que a resposta não é uma afirmação direta, mas sim a recuperação de documentos que se alinham semanticamente à consulta.

### Aula 2 - Grafos de conhecimento

Falamos sobre grafos de conhecimento, que são redes de entidades interligadas, representando relações semânticas. Usei exemplos simples, como uma árvore genealógica, para ilustrar como podemos navegar por essas conexões. Embora a criação desses grafos tenha sido manual por décadas, eles formaram a base da busca semântica antes da IA moderna. Atualmente, ainda são úteis, mas geralmente em conjunto com técnicas de IA, que facilitam a busca e a validação das respostas.

### Aula 3 - Avaliação de busca semântica

Discutimos a avaliação de resultados em busca semântica, focando em como medir a eficácia sem depender apenas de análises subjetivas. Abordei a importância de conjuntos de teste e métricas como precision e recall para avaliar a relevância dos resultados. Também mencionei a avaliação online, onde você pode comparar seu sistema com um existente, utilizando testes A/B para analisar o comportamento dos usuários. O objetivo é garantir que seu sistema de busca funcione de forma eficiente e relevante.

### Aula 4 - Bancos de dados de vetores

Discutimos a busca semântica e como ela deve ser realizada de forma eficiente, utilizando bancos de dados de vetores. Abordei a importância de algoritmos especializados, como HNSW e LSH, que otimizam a busca ao invés de comparar cada query com bilhões de documentos. Também falamos sobre o trade-off entre acurácia e velocidade, destacando que, embora a busca exata seja mais precisa, ela é mais lenta. Por fim, introduzimos o Chroma como um exemplo de banco de dados de vetores para nossa aplicação prática.

## Bloco C - Classificação de Dados

### Aula 1 - Classificação com IA

Exploramos a diferença entre classificação clássica e classificação com IA. Discutimos como os modelos clássicos exigem treinamento do zero, enquanto os modelos de IA, ou foundational models, já vêm pré-treinados e prontos para inferência. Abordei a importância da generalização desses modelos, que são treinados com grandes volumes de dados. Também mencionei que, mesmo com a ascensão da IA, os modelos clássicos ainda têm seu espaço, especialmente em dados estruturados.

### Aula 2 - Avaliação de sistemas de classificação

Discutimos a importância da avaliação de sistemas de classificação, especialmente em contextos como reconhecimento facial. Abordamos como garantir que um modelo realmente funcione, utilizando um conjunto de testes rotulados e a matriz de confusão para identificar acertos e erros. Falei sobre métricas como acurácia, precisão e recall, além da curva ROC e a métrica AUC, que ajudam a entender a performance do modelo. Essas ferramentas são essenciais para validar e comparar sistemas de classificação

### Aula 3 - Abordagens de classificação

Exploramos como realizar classificação com IA na prática, abordando três métodos principais. O primeiro envolve o uso de embeddings pré-treinados, onde um modelo gera um embedding que pode ser interpretado por qualquer sistema, não necessariamente uma IA. O segundo método é o ajuste de modelos pré-treinados, onde fazemos fine-tuning para especializar o modelo na tarefa desejada. Por fim, falamos sobre modelos de LLM, como o ChatGPT, que já têm um vasto conhecimento e podem classificar informações. Também menciono uma quarta abordagem, mas com custos elevados.

### Aula 4 - Exemplos de sistemas de classificação

Abordamos a classificação com embeddings, que chamei de "ingênua" por ser uma técnica simples, mas eficaz. Vamos encontrar os K vizinhos mais próximos de uma instância e determinar a classe majoritária entre eles. Também discutimos o fine-tuning de modelos pré-treinados, como o BERT, e como ajustar um modelo para tarefas específicas. Por fim, falamos sobre o uso de LLMs, como o ChatGPT, e técnicas de prompt engineering para garantir classificações precisas.
