# Glossário - Classificação de Texto com IA

Terminologia técnica do projeto organizada em três níveis de profundidade:

- **Iniciante**: Definição conceitual simplificada
- **Intermediário**: Definição técnica com contexto do projeto
- **Avançado**: Aspectos técnicos detalhados e nuances de implementação

---

## A

### Accuracy (Acurácia)

**Iniciante:** Proporção de predições corretas sobre o total de predições realizadas.

**Intermediário:** Métrica global calculada como (VP + VN) / (VP + VN + FP + FN). No projeto, o modelo fine-tuned alcançou 91.55% de accuracy.

**Avançado:** Métrica que pode ser enganosa em datasets desbalanceados pois não diferencia entre tipos de erro. Para classificação multiclasse, representa a fração de predições corretas mas não considera distribuição de classes. Preferir F1-Score ou balanced accuracy em cenários de desbalanceamento.

---

### API (Application Programming Interface)

**Iniciante:** Interface que permite comunicação entre sistemas de software sem expor implementação interna.

**Intermediário:** Protocolo de comunicação entre cliente e servidor. Projeto utiliza API REST do Gemini para classificação de texto via requisições HTTP.

**Avançado:** Conjunto de endpoints que define contratos de comunicação stateless via HTTP. Implementação requer tratamento de rate limits (429), timeouts (503), autenticação via API keys, e retry logic com exponential backoff.

---

### AUC-ROC (Area Under the Curve)

**Iniciante:** Métrica que avalia capacidade do modelo de distinguir entre classes.

**Intermediário:** Área sob a curva ROC, variando de 0 a 1. Projeto alcançou 0.9964, indicando excelente separabilidade entre classes de emoção.

**Avançado:** Métrica threshold-independent que avalia qualidade das probabilidades preditas. Em multiclass, utiliza estratégia one-vs-rest. Valores acima de 0.9 indicam alta separabilidade. Invariante a calibração de probabilidades e preferível a accuracy em datasets desbalanceados.

---

## B

### Batch Size

**Iniciante:** Quantidade de amostras processadas simultaneamente durante treinamento ou inferência.

**Intermediário:** Hiperparâmetro que define quantos exemplos são processados antes de atualizar pesos do modelo. Projeto utiliza batch size de 16 para fine-tuning.

**Avançado:** Trade-off entre uso de memória GPU/RAM, estabilidade do gradiente e velocidade de convergência. Mini-batch gradient descent (16-128) geralmente superior a batch completo ou SGD puro. Batch sizes em potências de 2 otimizam uso de memória em GPUs.

---

### Baseline

**Iniciante:** Modelo de referência simples usado para comparação de performance.

**Intermediário:** Modelo que estabelece performance mínima esperada. No projeto, Embedding + KNN serve como baseline (75% accuracy) contra o qual outros modelos são comparados.

**Avançado:** Pode ser regra heurística (majority class prediction), modelo clássico (logistic regression), ou modelo pré-treinado sem fine-tuning. Essencial para validar que complexidade adicional justifica custo computacional.

---

## C

### Classe (Class/Label)

**Iniciante:** Categoria ou tipo que o modelo deve prever.

**Intermediário:** Em classificação de emoções, existem 6 classes mutuamente exclusivas: sadness, joy, love, anger, fear, surprise.

**Avançado:** No contexto de classificação supervisionada, classe representa o target y ∈ Y onde Y é o espaço de classes discretas. Projeto utiliza classificação single-label multiclass. Alternativas incluem multilabel e classificação hierárquica.

---

### Confusion Matrix (Matriz de Confusão)

**Iniciante:** Tabela que mostra padrões de acerto e erro do modelo por classe.

**Intermediário:** Matriz onde linhas representam classes verdadeiras e colunas representam predições. Diagonal principal indica acertos; células fora da diagonal mostram confusões específicas.

**Avançado:** Matriz C onde C[i,j] representa quantidade de amostras da classe i classificadas como j. Permite calcular métricas por classe (precision, recall, F1). Normalização por linha mostra recall; por coluna mostra precision. Confusões entre classes semanticamente próximas são esperadas.

---

### Cosine Similarity

**Iniciante:** Métrica que mede similaridade entre vetores baseada no ângulo entre eles.

**Intermediário:** Varia de -1 (opostos) a 1 (idênticos), independente da magnitude dos vetores. Utilizada no KNN para encontrar vizinhos mais próximos no espaço de embeddings.

**Avançado:** Definida como cos(θ) = (A·B)/(||A|| ||B||). Invariante a escala, ideal para embeddings L2-normalized. Distância cosine = 1 - similarity. Captura similaridade semântica melhor que distância euclidiana em espaços de alta dimensão.

---

## D

### Dataset

**Iniciante:** Conjunto de dados utilizado para treinar e avaliar modelos.

**Intermediário:** Projeto utiliza dataset "emotion" do Hugging Face: 16.000 amostras de treino e 2.000 de teste, com 6 classes de emoções.

**Avançado:** Corpus rotulado D = {(x_i, y_i)} com split típico train/validation/test. Considerações incluem balanceamento de classes, prevenção de data leakage, distribuição IID, e domain shift entre treino e produção.

---

### DistilBERT

**Iniciante:** Versão compacta do BERT com menor tamanho e maior velocidade.

**Intermediário:** Modelo obtido via knowledge distillation com 40% menos parâmetros e 60% mais rápido que BERT, mantendo 97% da performance.

**Avançado:** Transformer com 6 camadas (vs 12 do BERT), reduzindo de 110M para 66M parâmetros. Treinamento usa distillation loss, MLM loss e cosine embedding loss. Trade-off: 3% degradação em accuracy por 60% redução em latência.

---

## E

### Embedding

**Iniciante:** Representação numérica vetorial de texto.

**Intermediário:** Vetores densos (384 dimensões) onde textos semanticamente similares possuem embeddings próximos. Projeto usa all-MiniLM-L6-v2 para gerar embeddings.

**Avançado:** Mapeamento f: X → ℝ^d onde textos são projetados em espaço vetorial denso. Propriedades incluem preservação semântica via similaridade cosine e invariância a escala quando L2-normalized. Gerados por transformers pré-treinados em contrastive learning.

---

### Epoch

**Iniciante:** Uma passagem completa através do conjunto de dados de treino.

**Intermediário:** Treinar por 3 epochs significa que cada amostra foi vista 3 vezes durante o treinamento.

**Avançado:** Total de updates = (N_samples / batch_size) × epochs. Trade-off entre underfitting (poucas epochs) e overfitting (muitas epochs). Early stopping monitora validation loss para determinar número ótimo de epochs.

---

## F

### F1-Score

**Iniciante:** Média harmônica entre Precision e Recall.

**Intermediário:** Calculado como F1 = 2 × (P × R) / (P + R). Útil em datasets desbalanceados pois equilibra precision e recall.

**Avançado:** Mais sensível a valores baixos que média aritmética, penalizando desbalanceamento P≠R. Em multiclass: micro (agregar globalmente), macro (média não-ponderada), weighted (ponderada por support). Projeto usa weighted average.

---

### False Negative (FN)

**Iniciante:** Modelo falha em detectar classe positiva quando ela está presente.

**Intermediário:** Amostra pertence à classe mas modelo prediz outra. Impacta negativamente o Recall.

**Avançado:** Tipo II error em classificação binária. Em multiclass one-vs-rest, FN_c representa amostras da classe c classificadas como não-c. Custo de FN varia por domínio. Recall = TP/(TP+FN) quantifica sensibilidade a falsos negativos.

---

### False Positive (FP)

**Iniciante:** Modelo incorretamente atribui classe positiva quando verdadeira é outra.

**Intermediário:** Amostra não pertence à classe mas modelo a prediz. Impacta negativamente a Precision.

**Avançado:** Tipo I error. Em multiclass, FP_c representa amostras não-c classificadas como c. Precision = TP/(TP+FP) quantifica taxa de falsos positivos. Trade-off com FN via threshold tuning.

---

### Few-Shot Learning

**Iniciante:** Aprendizado com poucos exemplos de treinamento.

**Intermediário:** LLM recebe 3 exemplos de cada emoção no prompt para guiar classificação sem atualizar pesos.

**Avançado:** Paradigma com N exemplos onde N << tamanho típico de dataset. LLMs exploram in-context learning através de exemplos no prompt. Projeto usa 3-shot (18 exemplos total). Trade-off entre flexibilidade e accuracy.

---

### Fine-Tuning

**Iniciante:** Ajuste de modelo pré-treinado para tarefa específica.

**Intermediário:** DistilBERT pré-treinado é adaptado para classificação de emoções através de treinamento supervisionado no dataset emotion.

**Avançado:** Transfer learning onde modelo pré-treinado em MLM/NSP é adaptado via supervised training. Utiliza learning rate baixa (2e-5), warmup e gradient clipping para evitar catastrophic forgetting. Projeto: 3 epochs × 16k samples alcançaram 91% accuracy.

---

## G

### Gemini

**Iniciante:** Large Language Model do Google acessível via API.

**Intermediário:** LLM utilizado para classificação via prompt engineering. Projeto usa gemini-2.0-flash-exp com structured outputs.

**Avançado:** Modelo multimodal otimizado para latência/custo. Free tier limita a 10 req/min. Trade-offs vs fine-tuning: sem custo de treino, flexibilidade via prompt, latência maior, dependência de API externa.

---

### Gradient Descent

**Iniciante:** Algoritmo de otimização que ajusta pesos do modelo iterativamente.

**Intermediário:** Calcula gradiente da loss function e atualiza pesos na direção oposta: w ← w - η∇L.

**Avançado:** Otimizador iterativo θ_{t+1} = θ_t - η∇_θ L(θ_t). Variantes incluem SGD, mini-batch e Adam. Projeto usa AdamW com learning rate 2e-5 e warmup para estabilidade.

---

## H

### Hugging Face

**Iniciante:** Plataforma que hospeda modelos e datasets de NLP.

**Intermediário:** Biblioteca transformers fornece API unificada para carregar DistilBERT; biblioteca datasets carrega emotion dataset.

**Avançado:** Ecossistema MLOps com Model Hub, Datasets Hub e Spaces. Projeto integra AutoModelForSequenceClassification, Trainer API e datasets.load_dataset. Suporta PyTorch, TensorFlow e JAX.

---

### Hyperparameter

**Iniciante:** Parâmetro definido antes do treinamento que controla o processo de aprendizado.

**Intermediário:** Exemplos no projeto: learning rate (2e-5), batch size (16), epochs (3), K neighbors (5).

**Avançado:** Configurações não otimizadas por gradiente. Tuning via grid search, random search ou Bayesian optimization. Validação cruzada previne overfitting no hyperparameter space.

---

## I

### Inference

**Iniciante:** Uso do modelo treinado para fazer predições em dados novos.

**Intermediário:** Forward pass sem atualização de pesos. Projeto mede tempo: Embedding (0.5s), Fine-tuned (2.3s), LLM (5.7s) para 100 amostras.

**Avançado:** Forward pass ŷ = f(x; θ*) com pesos fixos. Métricas incluem latência e throughput. Otimizações: quantização, pruning, distillation, batching. Produção requer P95/P99 latency.

---

## K

### K-Nearest Neighbors (KNN)

**Iniciante:** Algoritmo que classifica baseado nos K vizinhos mais próximos.

**Intermediário:** Projeto usa K=5 com distância cosine em espaço de embeddings de 384 dimensões.

**Avançado:** Método não-paramétrico: ŷ = mode({y_i : i ∈ N_k(x)}). Lazy learning com complexidade O(Nd) na inferência. Sensível a escolha de K, métrica e curse of dimensionality.

---

## L

### Label

**Iniciante:** Classe verdadeira de cada exemplo no dataset.

**Intermediário:** Uma das 6 emoções: sadness, joy, love, anger, fear, surprise. Utilizado para treinamento supervisionado e avaliação.

**Avançado:** Ground truth y ∈ Y em aprendizado supervisionado. Qualidade impacta performance: inter-annotator agreement, label noise, ambiguidade. Mapeamento label2id/id2label para interface com modelos.

---

### Learning Rate

**Iniciante:** Controla magnitude dos ajustes de pesos durante treinamento.

**Intermediário:** Projeto usa 2e-5 (0.00002) para fine-tuning. Valor pequeno previne divergência e catastrophic forgetting.

**Avançado:** Hiperparâmetro η em w ← w - η∇L. Trade-off entre velocidade e estabilidade. Técnicas: warmup, decay, cyclic LR. Fine-tuning requer η ~ 1e-5 a 5e-5.

---

### LLM (Large Language Model)

**Iniciante:** Modelo de linguagem com bilhões de parâmetros treinado em corpus massivo.

**Intermediário:** Transformer-based model usado para classificação via prompt engineering. Projeto usa Gemini para zero/few-shot classification.

**Avançado:** Modelos com 10B+ parâmetros pré-treinados em self-supervised tasks. Emergent abilities incluem few-shot learning e chain-of-thought reasoning. Trade-offs: versatilidade vs custo/latência/controle.

---

### Loss Function

**Iniciante:** Função que quantifica erro entre predição e ground truth.

**Intermediário:** Classificação multiclass usa cross-entropy loss. Treinamento busca minimizar loss via gradient descent.

**Avançado:** L(θ) = Σ_i ℓ(f(x_i; θ), y_i). Para multiclass: cross-entropy L = -Σ_c y_c log(ŷ_c) onde y é one-hot e ŷ é softmax output. Regularização: L = L_task + λL_reg.

---

## M

### Macro Average

**Iniciante:** Média simples de métrica calculada por classe.

**Intermediário:** Calcula precision/recall/F1 para cada classe e faz média não-ponderada. Trata todas as classes igualmente.

**Avançado:** M_macro = (1/C)Σ_c M_c. Não pondera por support, útil para detectar performance em classes minoritárias. Contrasta com micro (agregar antes) e weighted (ponderar por frequência).

---

### Metrics

**Iniciante:** Medidas quantitativas de performance do modelo.

**Intermediário:** Projeto avalia accuracy, precision, recall, F1-Score, AUC-ROC e confusion matrix.

**Avançado:** Funções M: (y_true, y_pred) → ℝ. Categorias: threshold-based (accuracy, F1), ranking (AUC), probabilistic (log-loss). Escolha depende de objetivo de negócio e custos assimétricos de erro.

---

## O

### Overfitting

**Iniciante:** Modelo memoriza dados de treino mas falha em generalizar.

**Intermediário:** Alta performance em treino mas baixa em teste. Causado por modelo muito complexo ou treino excessivo.

**Avançado:** L_train baixo mas L_val alto. Mitigação: regularização (L2, dropout), data augmentation, early stopping. Bias-variance tradeoff: underfitting (high bias) vs overfitting (high variance).

---

## P

### Precision

**Iniciante:** Proporção de predições positivas que são corretas.

**Intermediário:** Precision = TP / (TP + FP). Mede confiabilidade quando modelo prediz determinada classe.

**Avançado:** Quantifica taxa de acerto em predições positivas. Em multiclass one-vs-rest, P_c = amostras corretamente classificadas como c / total classificadas como c. Trade-off com recall via threshold.

---

### Prompt Engineering

**Iniciante:** Design de instruções eficazes para LLMs.

**Intermediário:** Projeto estrutura prompts com instruções, exemplos (few-shot) e formato de saída para guiar Gemini na classificação.

**Avançado:** Otimização de input textual para LLMs. Técnicas: zero-shot, few-shot, chain-of-thought, role prompting. Projeto usa 3-shot + structured output (enum). Considerações: token budget, order sensitivity.

---

## R

### Rate Limit

**Iniciante:** Restrição de quantidade de requisições permitidas por unidade de tempo.

**Intermediário:** Gemini free tier limita a 10 requisições/minuto. Projeto implementa delays e batching para respeitar limite.

**Avançado:** Throttling mechanism para controlar uso de recursos. Handling: exponential backoff (erro 429), batching, rate limiting libraries. Projeto: batch_size=5, sleep(30s) entre batches, retry com delay=7s.

---

### Recall

**Iniciante:** Proporção de amostras positivas corretamente identificadas.

**Intermediário:** Recall = TP / (TP + FN). Mede capacidade de encontrar todas as amostras de determinada classe.

**Avançado:** R = TP/(TP+FN) quantifica cobertura. Também chamado sensitivity ou true positive rate (TPR). Trade-off precision-recall: baixar threshold aumenta recall mas reduz precision.

---

### Regularization

**Iniciante:** Técnica para prevenir overfitting através de penalização de complexidade.

**Intermediário:** L2 regularization (weight decay) penaliza pesos grandes. Projeto usa weight_decay=0.01 em AdamW optimizer.

**Avançado:** Modificação de objetivo: L' = L + λR(θ). Tipos: L1 (lasso), L2 (ridge), dropout, data augmentation. Early stopping é regularização implícita. Interpretação Bayesiana: regularização como prior.

---

### ROC Curve

**Iniciante:** Gráfico que mostra performance do modelo em diferentes thresholds.

**Intermediário:** Plota True Positive Rate vs False Positive Rate variando threshold de decisão. Área sob curva (AUC) resume qualidade.

**Avançado:** Plot parametrizado por threshold: (FPR(t), TPR(t)). Diagonal y=x representa classificador aleatório (AUC=0.5). Em multiclass usa one-vs-rest. Projeto: AUC=0.9964 indica probabilidades bem calibradas.

---

## S

### Sentence Transformers

**Iniciante:** Biblioteca especializada em gerar embeddings de sentenças.

**Intermediário:** Projeto usa all-MiniLM-L6-v2 para gerar embeddings de 384 dimensões preservando semântica.

**Avançado:** Framework baseado em Siamese/Triplet networks que fine-tuna BERT/RoBERTa em sentence pair datasets usando contrastive loss. all-MiniLM-L6-v2: 6 camadas, 384-dim, treinado em 1B pares.

---

### Softmax

**Iniciante:** Função que converte valores arbitrários em probabilidades que somam 1.

**Intermediário:** Aplicada na camada final de classificação para converter logits em distribuição de probabilidade sobre classes.

**Avançado:** σ(z)_i = e^{z_i}/Σ_j e^{z_j} mapeando ℝ^C → Δ^{C-1}. Propriedades: monotonicidade, invariância a shifts, diferenciabilidade. Cross-entropy loss opera em softmax outputs. Temperature scaling para calibração.

---

### Supervised Learning

**Iniciante:** Aprendizado a partir de dados rotulados.

**Intermediário:** Modelo aprende mapeamento texto→emoção usando pares (texto, label). Todas as 3 abordagens do projeto utilizam supervised learning.

**Avançado:** Aprendizado de f: X → Y a partir de D = {(x_i, y_i)}. Objetivo: minimizar empirical risk R_emp = (1/N)Σ L(f(x_i), y_i). Contrasta com unsupervised, semi-supervised e reinforcement learning.

---

## T

### Tokenizer

**Iniciante:** Ferramenta que converte texto em sequência de tokens numéricos.

**Intermediário:** DistilBERT usa WordPiece tokenizer com vocabulário de ~30k tokens. Palavras frequentes mapeiam para 1 token; raras são subdivididas.

**Avançado:** Mapping bidirecional texto ↔ IDs. WordPiece inclui special tokens [CLS], [SEP], [PAD], [UNK]. Max_length=128 no projeto (truncate/pad). Fast tokenizers (Rust-based) são 10x+ rápidos.

---

### Training

**Iniciante:** Processo de ajustar pesos do modelo para minimizar erro.

**Intermediário:** Processo iterativo usando gradient descent e backpropagation. Projeto treina Fine-tuned por 3 epochs (~30-60 min).

**Avançado:** Otimização iterativa θ_{t+1} = θ_t - η∇L(θ_t). Pipeline: forward pass, backward pass, parameter update. Projeto: 16k samples × 3 epochs × batch=16 → 3000 steps usando AdamW optimizer.

---

### Transfer Learning

**Iniciante:** Reutilização de conhecimento de uma tarefa para aprender outra.

**Intermediário:** DistilBERT pré-treinado em corpus geral é fine-tunado no dataset emotion específico, economizando tempo e dados.

**Avançado:** Conhecimento de task source T_s transferido para T_t. DistilBERT pré-treinado em MLM → fine-tuned em 6-way emotion classification. Reduz data requirements: 16k samples suficientes vs 100k+ from-scratch.

---

### Transformer

**Iniciante:** Arquitetura de rede neural baseada em mecanismo de atenção.

**Intermediário:** Arquitetura que processa texto inteiro em paralelo usando self-attention. Base de BERT, GPT e DistilBERT.

**Avançado:** Encoder-decoder com multi-head self-attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V. Vantagens vs RNN: paralelização, long-range dependencies, interpretabilidade. Complexity: O(n²d).

---

## U

### Underfitting

**Iniciante:** Modelo muito simples que não captura padrões nos dados.

**Intermediário:** Performance ruim tanto em treino quanto em teste. Soluções: modelo mais complexo, mais features, treinar mais tempo.

**Avançado:** L_train alto devido a capacidade insuficiente. Causas: modelo muito simples, features inadequadas, regularização excessiva. Bias-variance: underfitting = high bias.

---

## V

### Validation Set

**Iniciante:** Conjunto de dados separado para avaliar modelo durante treinamento.

**Intermediário:** Subset usado para ajustar hiperparâmetros e early stopping. Diferente de teste (usado apenas ao final).

**Avançado:** Split D = D_train ∪ D_val ∪ D_test. D_val usado para early stopping, hyperparameter tuning, model selection. K-fold cross-validation para datasets pequenos.

---

### Vector

**Iniciante:** Array de números representando dados.

**Intermediário:** Embeddings são vetores de 384 números representando semântica de textos. Operações: distância, similaridade cosine.

**Avançado:** Elemento v ∈ ℝ^d. Embeddings mapeiam textos para ℝ^384 preservando semântica via estrutura geométrica. Operações: norma, dot product, aritmética vetorial.

---

## W

### Weight

**Iniciante:** Parâmetros do modelo ajustados durante treinamento.

**Intermediário:** Cada conexão em rede neural possui um peso. Treinamento encontra pesos que minimizam loss function.

**Avançado:** Parâmetros θ = {W, b} otimizados via gradient descent. Em linear layer: y = Wx + b. Inicialização crítica: Xavier/He. DistilBERT possui ~66M parâmetros.

---

### Weighted Average

**Iniciante:** Média ponderada pela frequência de cada classe.

**Intermediário:** Classes mais frequentes têm maior peso na métrica final. Projeto usa weighted average para respeitar distribuição real.

**Avançado:** M_weighted = Σ_c (n_c/N) × M_c onde n_c é support. Contrasta com macro (não ponderado) e micro (agregar antes). Alinhado com distribuição real do dataset.

---

## Z

### Zero-Shot Learning

**Iniciante:** Modelo executa tarefa sem exemplos específicos de treinamento.

**Intermediário:** LLM classifica baseado apenas em instruções, sem exemplos no prompt. Mais flexível mas menos preciso que few-shot.

**Avançado:** Inference sem exemplos de treino específicos, generalizando de conhecimento pré-treinado. LLMs exploram semantic understanding e world knowledge. Trade-off: flexibilidade vs accuracy.

---

## Conceitos de Implementação

### Batch Processing

**Iniciante:** Processar múltiplas amostras simultaneamente.

**Intermediário:** LLM processa textos em batches de 5 com delays para respeitar rate limits.

**Avançado:** Agrupa requisições para otimizar throughput respeitando constraints de API (10 req/min). Implementação: batch_size=5, sleep entre batches, retry logic com exponential backoff.

---

### Cache

**Iniciante:** Armazenamento de resultados computados para reutilização.

**Intermediário:** Modelos treinados salvos em disco permitem re-execução rápida sem re-treinar. Embedding e Fine-tuned carregam de cache.

**Avançado:** Persistência de artefatos via pickle/joblib. Trade-off: espaço em disco vs tempo de computação. Projeto: embeddings (~200MB), Fine-tuned model (~250MB) em `models/`.

---

### Configuration Management

**Iniciante:** Centralização de parâmetros em arquivo único.

**Intermediário:** config.py define todos os hiperparâmetros: EPOCHS=3, BATCH_SIZE=16, LEARNING_RATE=2e-5, etc.

**Avançado:** Single source of truth para hyperparameters, paths, API keys. Facilita experimentação e reproducibilidade. Environment variables (.env) para secrets. Alternativas: YAML, Hydra.

---

### Structured Outputs

**Iniciante:** Formato específico esperado na resposta de API.

**Intermediário:** Gemini retorna classificação em formato enum garantindo output válido entre as 6 classes.

**Avançado:** Schema enforcement via response_mime_type='text/x.enum' e response_schema. Previne outputs inválidos, facilita parsing, reduz post-processing. Alternativas: JSON schema, Pydantic models.

---

## Métricas do Projeto

### Performance Comparativa

| Métrica | Embedding+KNN | Fine-tuned | LLM |
|---------|---------------|------------|-----|
| Accuracy | ~75% | 91.55% | ~82% |
| Tempo Treino | ~5 min | ~30-60 min | 0 (API) |
| Tempo Inferência (100 samples) | 0.5s | 2.3s | 5.7s |
| Parâmetros | 0 (lazy) | 66M | API externa |
| Requer GPU | Não | Recomendado | Não |

---

## Referências

- Vaswani et al. (2017). "Attention is All You Need"
- Devlin et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers"
- Sanh et al. (2019). "DistilBERT, a distilled version of BERT"
- Reimers & Gurevych (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"