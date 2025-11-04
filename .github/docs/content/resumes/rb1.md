# Bloco A - Embeddings e busca semântica

## Aula 1 - Introdução à busca semântica e classificação

Exploramos os conceitos de busca semântica e classificação. A busca semântica envolve encontrar objetos semelhantes em significado, como textos ou imagens, enquanto a classificação atribui uma categoria a um objeto. Por exemplo, ao buscar "animais", queremos resultados que incluam gatos e cachorros, não apenas palavras semelhantes. Já na classificação, como em reconhecimento facial ou identificação de spam, o computador precisa entender o contexto para categorizar corretamente. Esses conceitos são fundamentais em diversas aplicações do dia a dia.

## Aula 2 - Introdução a embeddings

Exploramos como representar objetos do mundo real para que os computadores possam entendê-los. Discutimos a limitação dos computadores em "ver" e "ouvir" como nós, e a importância de usar embeddings, que são vetores numéricos que mapeiam objetos em espaços n-dimensionais. Esses vetores permitem que o computador reconheça semelhanças e diferenças entre palavras e imagens, mesmo que essa representação não tenha significado direto para nós.

## Aula 3 - Embeddings como vetores

Exploramos o conceito de embeddings, comparando-os a vetores em um espaço n-dimensional. Usei um exemplo em duas dimensões para ilustrar como diferentes objetos, como animais e palavras, podem ser representados como setas em um plano cartesiano. Também mencionei um site da Carnegie Mellon University, onde é possível visualizar embeddings em três dimensões, permitindo interagir com palavras e observar suas relações de gênero e idade. <https://www.cs.cmu.edu/~dst/WordEmbeddingDemo>

## Aula 4 - Características de embeddings

Exploramos o conceito de embeddings usando cidades do mundo e suas temperaturas médias em janeiro e julho. Mostrei como essas cidades se agrupam em um gráfico bidimensional, revelando diferenças climáticas. Abordei as limitações de criar embeddings manualmente e a importância de representações densas, que são mais eficientes. Também discuti a necessidade de que cada objeto tenha uma representação única, apesar de possíveis ambiguidades. O objetivo é entender como embeddings podem capturar a semântica de forma mais eficaz.

## Aula 5 - Embeddings ingênuos

Discutimos algumas abordagens ingênuas para criar embeddings, começando pela representação one-hot. Embora pareça uma boa ideia, essa técnica resulta em vetores esparsos e grandes, sem semântica real. Exploramos como essa representação não captura a proximidade entre palavras ou objetos, levando a distâncias iguais entre termos sem relação. Também abordamos a contagem de entidades e a conversão de dados em bits, que não resolvem o problema de representação semântica. Vamos explorar alternativas mais eficazes em aulas futuras.

## Aula 6 - Gerando embeddings com redes neurais

Exploramos como criar embeddings significativos a partir de redes neurais, um conceito que já existe há mais de 20 anos. Discuti como as redes transformam inputs, como texto ou imagens, em representações numéricas que capturam seu significado. Apresentei dois exemplos clássicos: o Word2Vec, que gera embeddings para palavras com base em seus contextos, e os autoencoders, que reduzem a dimensionalidade de imagens para criar representações densas. Esses métodos são fundamentais para entender a evolução das redes neurais.

## Aula 7 - Similaridade de embeddings

Explicamos como calcular a similaridade entre embeddings, que são representações vetoriais de objetos, como gatos e cachorros. Vamos explorar três métodos principais: a distância euclidiana, a similaridade de cossenos e o produto escalar. Cada um tem suas particularidades e aplicações, dependendo do contexto dos embeddings. A similaridade de cossenos é ideal para tarefas de texto, enquanto a distância euclidiana pode ser mais adequada em outras situações.
