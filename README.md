<!--markdownlint-disable-->

<p align="center">
  <img alt="Logo - Rocketseat" src="./.github/assets/logo_ftr.png" width="200px" />
</p>

# Busca Semântica e Classificação de Dados

## Sobre

Repositório pessoal de registro, referência e suporte para fins de aprendizado, consulta e acompanhamento da disciplina de **Busca Semântica e Classificação de Dados** (Nível 6), Fase 2 (**Estratégia e Inovação**), do curso de Pós-Graduação *Tech Developer 360*, desenvolvido pela Faculdade de Tecnologia Rocketseat (FTR).

>[!NOTE]
> [Questionário Avaliativo](./.github/docs/content/assessments/q.md)

## Conteúdo

### Bloco A - Embeddings e Busca Semântica

Embeddings representam uma inovação fundamental em machine learning que permite computadores operarem com dados não estruturados (textos, imagens, áudios) através de representações vetoriais densas em espaços n-dimensionais. Diferentemente de abordagens ingênuas como one-hot encoding que produzem vetores esparsos sem semântica, embeddings gerados por redes neurais (Word2Vec, autoencoders, BERT) preservam relações semânticas através de proximidade geométrica: objetos similares em significado possuem vetores próximos no espaço de representação. Esta propriedade viabiliza busca semântica efetiva, onde sistemas recuperam informações relevantes baseando-se em similaridade de significado ao invés de correspondência exata de palavras-chave, superando limitações fundamentais de sistemas tradicionais de recuperação de informação.

A quantificação de similaridade entre embeddings utiliza métricas específicas cujas características determinam sua aplicabilidade: distância euclidiana mede proximidade absoluta sendo sensível a magnitude, similaridade de cossenos captura apenas direção relativa independente de escala (predominante em NLP), e produto escalar combina ambos aspectos favorecendo vetores de maior magnitude (comum em LLMs modernos). A escolha apropriada depende do contexto: embeddings contextualizados (BERT, GPT) superam representações estáticas (Word2Vec) ao gerar vetores distintos conforme o contexto de uso, enquanto tendências contemporâneas exploram embeddings multimodais (CLIP) que operam através de múltiplas modalidades em espaços compartilhados. Compreender estes conceitos é essencial para profissionais que trabalham com sistemas de recomendação, classificação automática, recuperação de informação e virtualmente qualquer aplicação moderna de IA que processa dados não estruturados.

- [Guia de Referência Teórica: Embeddings e Busca Semântica](./.github/docs/content/b1.md)

- [Projeto: Bloco A](./src/b1/) - 5 exemplos didáticos que demonstram conceitos fundamentais de embeddings e busca semântica, usando modelos pequenos e gratuitos do HuggingFace.

---

## Bloco B - Busca Semântica e Bancos de Dados de Vetores

A busca semântica constitui uma transformação fundamental nos sistemas de recuperação de informação, superando as limitações das abordagens baseadas em correspondência literal de termos. Este documento explora os quatro pilares tecnológicos que sustentam os sistemas modernos de busca semântica: a representação vetorial através de embeddings neurais, que permite capturar significado e contexto em espaços matemáticos de alta dimensionalidade; os grafos de conhecimento, estruturas que explicitam relações semânticas entre entidades e possibilitam raciocínio lógico formal; as metodologias rigorosas de avaliação, tanto offline quanto online, que garantem a relevância e qualidade dos resultados retornados; e os bancos de dados de vetores especializados, que viabilizam operações de busca em escala através de algoritmos aproximados de alta performance.

A integração destes componentes representa o estado da arte em sistemas de recuperação de informação, combinando capacidades complementares que atendem desde aplicações de pequena escala até sistemas corporativos com bilhões de documentos. Os embeddings proporcionam flexibilidade semântica para interpretar nuances linguísticas, enquanto os grafos de conhecimento estruturam informações de forma verificável e interpretável. As métricas de avaliação, como precision, recall e mean reciprocal rank, fornecem critérios objetivos para comparação de sistemas, e os bancos de dados especializados resolvem o desafio crítico de latência através de algoritmos como HNSW e LSH, que mantêm alta acurácia enquanto executam buscas em tempo sub-linear. Este arcabouço técnico fundamenta aplicações que vão desde motores de busca empresariais até assistentes conversacionais baseados em arquiteturas de Retrieval-Augmented Generation.

- [Guia de Referência Teórica: Busca Semântica e Bancos de Dados de Vetores](./.github/docs/content/b2.md)

- [Projeto: Bloco B](./src/b2/) - 4 exemplos didáticos que demonstram conceitos de busca semântica, utilizando modelos pequenos e gratuitos do HuggingFace e o banco de dados vetorial ChromaDB.

---

## Bloco C - Classificação de Dados com Inteligência Artificial

A classificação textual com inteligência artificial representa a evolução de décadas de pesquisa em processamento de linguagem natural, transitando de métodos estatísticos clássicos baseados em features manuais (Naive Bayes, SVM, Regressão Logística com TF-IDF) para arquiteturas neurais profundas fundamentadas em modelos transformer pré-treinados. Esta transformação paradigmática eliminou a dependência crítica de engenharia manual de features, substituindo-a por aprendizado automático de representações semânticas hierárquicas através de self-supervised learning em corpora massivos. O documento explora sistematicamente três abordagens contemporâneas de complexidade e performance crescentes: embeddings pré-treinados combinados com K-Nearest Neighbors, oferecendo implementação trivial sem necessidade de treinamento; fine-tuning de modelos transformer como DistilBERT, equilibrando performance estado-da-arte com eficiência computacional; e prompting de Large Language Models, explorando capacidades emergentes de classificação zero-shot e few-shot sem dados labeled.

A avaliação rigorosa destes sistemas transcende métricas simplistas de acurácia, demandando análise multidimensional através de precision, recall, F1-score e ROC-AUC, com considerações especiais para datasets desbalanceados prevalentes em aplicações reais. A seleção apropriada entre as três abordagens depende criticamente de trade-offs entre múltiplas dimensões: disponibilidade de dados labeled (LLMs requerem zero a poucos exemplos, fine-tuning demanda milhares), recursos computacionais (embeddings executam em CPU, fine-tuning necessita GPU, LLMs impõem custos de API), latência de inferência (modelos locais respondem em milissegundos, LLMs em segundos), e interpretabilidade (K-NN permite inspeção de vizinhos, transformers fine-tuned operam como caixas-pretas, LLMs possibilitam chain-of-thought reasoning). Esta taxonomia técnica fundamenta decisões arquiteturais em domínios que abrangem análise de sentimento, classificação de documentos, detecção de conteúdo tóxico, triagem médica automatizada e virtualmente qualquer aplicação empresarial que demande categorização inteligente de informação textual em escala.

- [Guia de Referência Teórica: Classificação de Dados com Inteligência Artificial](./.github/docs/content/b3.md)

- [Projeto: Bloco C](./src/b3/) - **3 abordagens de classificação** com IA: (1) **Embeddings + KNN** (naive, rápido), (2) **Fine-tuning DistilBERT** (melhor precisão), (3) **LLM com Gemini** (flexível, zero-shot)
