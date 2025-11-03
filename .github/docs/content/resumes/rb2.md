# Bloco B - Busca Semântica

## Aula 1 - Busca semântica com embeddings

Exploramos o conceito de embeddings e sua importância na busca semântica. Discutimos como esses vetores carregam informações semânticas, permitindo buscas mais significativas. Para implementar uma busca semântica, precisamos de uma base de conhecimento e um mecanismo de busca. Abordamos a criação de embeddings para documentos relevantes e a comparação deles com a query usando métricas de similaridade. Por fim, enfatizamos que a resposta não é uma afirmação direta, mas sim a recuperação de documentos que se alinham semanticamente à consulta.

## Aula 2 - Grafos de conhecimento

Falamos sobre grafos de conhecimento, que são redes de entidades interligadas, representando relações semânticas. Usei exemplos simples, como uma árvore genealógica, para ilustrar como podemos navegar por essas conexões. Embora a criação desses grafos tenha sido manual por décadas, eles formaram a base da busca semântica antes da IA moderna. Atualmente, ainda são úteis, mas geralmente em conjunto com técnicas de IA, que facilitam a busca e a validação das respostas.

## Aula 3 - Avaliação de busca semântica

Discutimos a avaliação de resultados em busca semântica, focando em como medir a eficácia sem depender apenas de análises subjetivas. Abordei a importância de conjuntos de teste e métricas como precision e recall para avaliar a relevância dos resultados. Também mencionei a avaliação online, onde você pode comparar seu sistema com um existente, utilizando testes A/B para analisar o comportamento dos usuários. O objetivo é garantir que seu sistema de busca funcione de forma eficiente e relevante.

## Aula 4 - Bancos de dados de vetores

Discutimos a busca semântica e como ela deve ser realizada de forma eficiente, utilizando bancos de dados de vetores. Abordei a importância de algoritmos especializados, como HNSW e LSH, que otimizam a busca ao invés de comparar cada query com bilhões de documentos. Também falamos sobre o trade-off entre acurácia e velocidade, destacando que, embora a busca exata seja mais precisa, ela é mais lenta. Por fim, introduzimos o Chroma como um exemplo de banco de dados de vetores para nossa aplicação prática.
