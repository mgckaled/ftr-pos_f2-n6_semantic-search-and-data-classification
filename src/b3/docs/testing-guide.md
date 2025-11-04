# Guia de Testes e Configuração

## Ordem Recomendada de Testes

### 1. Teste de Ambiente (1 min)

Verificar se todas as dependências foram instaladas:

```bash
cd src/b3
pipenv run python -c "import torch; import transformers; import sentence_transformers; from google import genai; print('✓ Todas as dependências OK!')"
```

### 2. Teste de Carregamento de Dados (2-5 min)

```bash
pipenv run python utils/data_loader.py
```

**O que verifica:**
- Download do dataset (primeira vez demora)
- Processamento dos dados
- Mapeamento de labels
- Distribuição de classes

### 3. Teste de Abordagem 1 - Embeddings (5-10 min)

```bash
pipenv run python models/embedding_classifier.py
```

**O que verifica:**
- Download do modelo sentence-transformers (~80MB)
- Geração de embeddings
- Treinamento do KNN
- Predições

**Tempo estimado:**
- Primeira execução: ~10 min (download do modelo)
- Execuções seguintes: ~5 min

### 4. Teste de Abordagem 3 - LLM (5-10 min)

```bash
pipenv run python models/llm_classifier.py
```

**O que verifica:**
- Conexão com API do Gemini
- Criação de prompts
- Few-shot learning
- Predições via API

**Requer:** API key configurada no `.env`

### 5. Teste de Abordagem 2 - Fine-tuning (30-60 min)

```bash
pipenv run python models/finetuned_classifier.py
```

**O que verifica:**
- Download do DistilBERT (~250MB)
- Fine-tuning (DEMORA!)
- Predições

**⚠️ ATENÇÃO:**
- Este é o teste mais demorado
- Deixe para o final
- Use apenas 1 época para teste (`epochs=1`)

### 6. Teste de Métricas e Visualização (1-2 min)

```bash
pipenv run python utils/metrics.py
pipenv run python utils/visualization.py
```

**O que verifica:**
- Cálculo de métricas
- Geração de gráficos
- Matrizes de confusão

### 7. Experimento Completo (1-2 horas)

```bash
pipenv run python main.py
```

**⚠️ Apenas após todos os testes acima funcionarem!**

---

## Parâmetros Configuráveis

Todos os parâmetros estão em `config.py`. Edite antes de executar os testes.

### Dataset

```python
# Dataset a ser usado
DATASET_NAME = "emotion"  # Opções: "emotion", "imdb"

# Limitar número de amostras (None = todas)
MAX_SAMPLES = None        # Para teste rápido: 500-1000
                         # Para experimento completo: None

# Split treino/teste
TEST_SIZE = 0.2          # 20% para teste
RANDOM_STATE = 42        # Seed para reprodutibilidade
```

**Sugestões por cenário:**

| Cenário | MAX_SAMPLES | DATASET_NAME | Tempo Total |
|---------|-------------|--------------|-------------|
| Teste rápido | 500 | "emotion" | ~30 min |
| Teste médio | 2000 | "emotion" | ~1 hora |
| Experimento completo | None | "emotion" | ~2 horas |
| Classificação binária | None | "imdb" | ~2-3 horas |

### Abordagem 1: Embeddings + KNN

```python
# Modelo de embeddings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# Outras opções:
# - "all-mpnet-base-v2" (melhor qualidade, mais lento)
# - "paraphrase-multilingual-MiniLM-L12-v2" (multilíngue)

# Número de vizinhos
K_NEIGHBORS = 5           # Padrão: 5
                         # Teste: 3, 5, 7, 10

# Métrica de distância
KNN_METRIC = "cosine"    # Opções: "cosine", "euclidean", "manhattan"
```

### Abordagem 2: Fine-tuning DistilBERT

```python
# Modelo base
FINETUNED_MODEL_NAME = "distilbert-base-uncased"
# Outras opções:
# - "bert-base-uncased" (melhor, mas mais lento)
# - "distilroberta-base" (alternativa)

# Hiperparâmetros de treinamento
EPOCHS = 3                    # Teste rápido: 1
                             # Experimento: 3-5
BATCH_SIZE = 16              # CPU: 8-16
                             # GPU: 32-64
LEARNING_RATE = 2e-5         # Padrão para BERT
MAX_LENGTH = 128             # Tokens máximos
                             # Textos longos: 256 ou 512

# Otimização
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
```

**Ajustes para CPU lenta:**

```python
EPOCHS = 1                # Apenas 1 época
BATCH_SIZE = 8            # Batch menor
MAX_LENGTH = 64           # Tokens reduzidos
MAX_SAMPLES = 500         # Dataset pequeno
```

### Abordagem 3: LLM (Gemini)

```python
# Modelo Gemini
LLM_MODEL_NAME = "gemini-2.0-flash-exp"
# Outras opções:
# - "gemini-1.5-flash" (mais estável)
# - "gemini-2.5-flash" (mais recente)

# Rate limiting
LLM_BATCH_SIZE = 10       # Processar em lotes
LLM_MAX_RETRIES = 3       # Tentativas em caso de erro
LLM_RETRY_DELAY = 2       # Segundos entre tentativas

# Few-shot learning
NUM_FEW_SHOT_EXAMPLES = 3 # Exemplos por classe
                         # Mais exemplos = melhor, mas prompt maior
```

### Métricas e Avaliação

```python
# Métricas a calcular
METRICS_TO_CALCULATE = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "confusion_matrix",
]

# Método de averaging (multiclasse)
AVERAGE_METHOD = "weighted"  # Opções: "micro", "macro", "weighted"
```

### Visualização

```python
# Tamanho das figuras
FIGURE_SIZE = (10, 8)    # (largura, altura)
DPI = 100                # Qualidade

# Formato de salvamento
SAVE_FORMAT = "png"      # Opções: "png", "pdf", "svg"
```

### Debug e Logging

```python
# Verbosidade
VERBOSE = True           # False para silencioso

# Seed global
SEED = 42                # Para reprodutibilidade
```

---

## Cenários de Teste Pré-configurados

### Cenário 1: Teste Ultra-Rápido (15 min)

**Objetivo:** Verificar se tudo funciona

**Configuração:**

```python
MAX_SAMPLES = 200
EPOCHS = 1
BATCH_SIZE = 8
MAX_LENGTH = 64
```

**Comandos:**

```bash
pipenv run python utils/data_loader.py
pipenv run python models/embedding_classifier.py
pipenv run python models/llm_classifier.py
# Pular fine-tuning
```

### Cenário 2: Teste Rápido (30-45 min)

**Objetivo:** Testar todas as abordagens superficialmente

**Configuração:**

```python
MAX_SAMPLES = 500
EPOCHS = 1
BATCH_SIZE = 16
```

**Comando:**

```bash
pipenv run python main.py
```

### Cenário 3: Experimento Médio (1-1.5 horas)

**Objetivo:** Resultados razoáveis para análise

**Configuração:**

```python
MAX_SAMPLES = 2000
EPOCHS = 2
BATCH_SIZE = 16
```

**Comando:**

```bash
pipenv run python main.py
```

### Cenário 4: Experimento Completo (2-3 horas)

**Objetivo:** Resultados finais para apresentação

**Configuração:**

```python
MAX_SAMPLES = None  # Todas as amostras
EPOCHS = 3
BATCH_SIZE = 16
```

**Comando:**

```bash
pipenv run python main.py
```

---

## Testes Modulares

### Testar apenas carregamento de dados

```bash
pipenv run python -c "
from utils.data_loader import load_and_prepare_dataset
train_df, test_df, id2label, label2id = load_and_prepare_dataset(max_samples=100)
print(f'Treino: {len(train_df)}, Teste: {len(test_df)}, Classes: {len(id2label)}')
"
```

### Testar apenas embeddings

```bash
pipenv run python -c "
from models.embedding_classifier import EmbeddingClassifier
from utils.data_loader import load_and_prepare_dataset

train_df, test_df, _, _ = load_and_prepare_dataset(max_samples=100)
clf = EmbeddingClassifier()
clf.fit(train_df['text'].tolist(), train_df['label'].tolist())
print('✓ Embeddings OK!')
"
```

### Testar apenas LLM (requer API key)

```bash
pipenv run python -c "
from models.llm_classifier import LLMClassifier

clf = LLMClassifier(labels=['positive', 'negative'])
pred = clf.predict(['This is a test'])
print(f'Predição: {pred[0]}')
"
```

---

## Checklist de Validação

Antes de rodar o experimento completo, verifique:

- [ ] ✅ Dependências instaladas (`pipenv install`)
- [ ] ✅ API key configurada no `.env`
- [ ] ✅ `utils/data_loader.py` funciona
- [ ] ✅ `models/embedding_classifier.py` funciona
- [ ] ✅ `models/llm_classifier.py` funciona (testa API)
- [ ] ✅ Parâmetros ajustados em `config.py`
- [ ] ✅ Espaço em disco suficiente (~2GB)

---

## Troubleshooting por Módulo

### data_loader.py

**Erro:** `ConnectionError` ou `TimeoutError`

**Solução:** Problema de internet. Dataset é baixado do Hugging Face.

```bash
# Tentar novamente
pipenv run python utils/data_loader.py
```

---

**Erro:** `KeyError: 'text'` ou `KeyError: 'label'`

**Solução:** Dataset diferente do esperado. Verificar nome do dataset em `config.py`.

---

### embedding_classifier.py

**Erro:** `OSError: Can't load model`

**Solução:** Modelo não foi baixado. Primeira execução demora (~5 min).

```bash
# Forçar download
pipenv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

---

**Erro:** `numpy.linalg.LinAlgError`

**Solução:** Muitas dimensões vazias. Aumentar `MAX_SAMPLES` ou `K_NEIGHBORS`.

---

### finetuned_classifier.py

**Erro:** `RuntimeError: CUDA out of memory`

**Solução:** Sem GPU ou memória insuficiente.

```python
# config.py
BATCH_SIZE = 4  # Reduzir ainda mais
MAX_LENGTH = 64
```

---

**Erro:** Treinamento muito lento

**Solução:** Normal na CPU. Reduzir configurações:

```python
EPOCHS = 1
MAX_SAMPLES = 500
BATCH_SIZE = 8
```

---

### llm_classifier.py

**Erro:** `ValueError: API key não configurada`

**Solução:** Configurar `.env`:

```bash
echo "GEMINI_API_KEY=sua_chave_aqui" > .env
```

---

**Erro:** `429 Too Many Requests` ou `ResourceExhausted`

**Solução:** Rate limit atingido. Aumentar delays:

```python
# config.py
LLM_BATCH_SIZE = 5        # Reduzir batch
LLM_RETRY_DELAY = 5       # Aumentar delay
```

---

**Erro:** `InvalidArgument: Invalid label returned`

**Solução:** LLM retornando label inválido. Ajustar prompt ou usar enum mais restritivo (já implementado).

---

### main.py

**Erro:** `ImportError: No module named 'config'`

**Solução:** Rodar do diretório correto:

```bash
cd src/b3
pipenv run python main.py
```

---

**Erro:** Script trava no fine-tuning

**Solução:** Normal. Fine-tuning demora 30-60 min na CPU. Verifique com `htop` se está rodando.

---

## Logs e Resultados

### Onde encontrar resultados

```
src/b3/results/
├── confusion_matrix_embedding.png
├── confusion_matrix_finetuned.png
├── confusion_matrix_llm.png
├── metrics_comparison.png
├── inference_time_comparison.png
├── class_distribution_train.png
├── class_distribution_test.png
└── logs/                           # Logs do treinamento
```

### Modelo treinado

```
src/b3/models/finetuned_classifier/
├── config.json
├── model.safetensors
├── tokenizer_config.json
└── vocab.txt
```

---

## Dicas de Otimização

### Para CPU lenta

```python
# Configuração mínima viável
MAX_SAMPLES = 500
EPOCHS = 1
BATCH_SIZE = 4
MAX_LENGTH = 64
K_NEIGHBORS = 3
```

### Para resultados rápidos (sem fine-tuning)

Comentar a seção de fine-tuning no `main.py` (linhas ~100-140) e rodar apenas embeddings + LLM.

### Para economizar chamadas de API

Reduzir amostras testadas no LLM:

```python
# main.py, linha ~150
test_sample_size = 50  # Ao invés de 100
```

---

## Monitoramento de Progresso

### Durante o treinamento

```bash
# Em outro terminal
watch -n 2 'ls -lh src/b3/results/'
```

### Tempo estimado por etapa

```
[ETAPA 1] Carregamento: ~2-5 min
[ETAPA 2] Embeddings:   ~5-10 min
[ETAPA 3] Fine-tuning:  ~30-60 min  ⏰
[ETAPA 4] LLM:          ~10-20 min
[ETAPA 5] Comparação:   ~1-2 min
```

**Total:** ~50-100 min (dependendo da configuração)

---

## Análise de Resultados

Após executar `main.py`, analisar:

1. **Terminal:** Tabela comparativa de métricas
2. **Matrizes de confusão:** Ver erros comuns por classe
3. **Gráfico de métricas:** Qual abordagem teve melhor desempenho
4. **Tempo de inferência:** Trade-off velocidade vs precisão

---

## Próximos Passos

Depois de validar tudo:

1. [ ] Rodar experimento completo
2. [ ] Analisar resultados
3. [ ] Documentar conclusões
4. [ ] Preparar apresentação (se necessário)
