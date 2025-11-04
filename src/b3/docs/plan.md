# Mini Projeto: Classificação de Dados com IA - Bloco C

## Objetivo

Implementar um projeto prático que demonstre as três principais abordagens de classificação com IA:

1. **Classificação com embeddings pré-treinados (KNN - "ingênua")**
2. **Fine-tuning de modelos pré-treinados (DistilBERT)**
3. **Classificação com LLMs usando prompt engineering (Gemini)**

Além disso, avaliar e comparar os sistemas usando métricas tradicionais (accuracy, precision, recall, F1, matriz de confusão).

## Estrutura do Projeto

```text
src/b3/
├── data/
│   ├── raw/                    # Dados brutos
│   └── processed/              # Dados processados
├── models/
│   ├── embedding_classifier.py # Abordagem 1: Embeddings + KNN
│   ├── finetuned_classifier.py # Abordagem 2: Fine-tuning DistilBERT
│   └── llm_classifier.py       # Abordagem 3: LLM + Prompt Engineering
├── utils/
│   ├── data_loader.py          # Carregamento e preparação de dados
│   ├── metrics.py              # Métricas de avaliação
│   └── visualization.py        # Visualização (matriz de confusão, ROC)
├── notebooks/
│   └── exploratory.ipynb       # Análise exploratória
├── docs/
│   └── plan.md                  # Este documento de planejamento
├── main.py                      # Script principal de execução
├── config.py                    # Configurações do projeto
├── .env.example                 # Exemplo de arquivo de configuração
└── README.md                    # Documentação de uso
```

## Dataset

**Dataset escolhido**: Classificação de sentimentos de reviews

**Fonte**: Hugging Face Datasets - `imdb` ou `emotion`

**Classes**:

- **IMDB**: 2 classes (positive, negative)
- **Emotion**: 6 classes (sadness, joy, love, anger, fear, surprise)

**Tamanho**: ~5000-10000 exemplos para treinamento/teste

## Abordagens Implementadas

### 1. Classificação com Embeddings (Ingênua/KNN)

**Biblioteca**: `sentence-transformers`

**Modelo**: `all-MiniLM-L6-v2` (pequeno, rápido, gratuito)

**Funcionamento**:

- Gerar embeddings para todos os textos do conjunto de treinamento
- Gerar embedding para o texto de entrada
- Encontrar os K vizinhos mais próximos (KNN)
- Retornar a classe majoritária entre os K vizinhos

**Vantagens**:

- Simples de implementar
- Não requer treinamento
- Rápido para inferência

**Desvantagens**:

- Pode não capturar nuances específicas da tarefa
- Depende da qualidade dos embeddings pré-treinados

### 2. Fine-tuning de Modelo Pré-treinado

**Biblioteca**: `transformers` (Hugging Face)

**Modelo base**: `distilbert-base-uncased` (versão menor do BERT)

**Funcionamento**:

- Adicionar camada de classificação ao modelo pré-treinado
- Fine-tuning no dataset específico
- Treinar por algumas épocas (3-5)

**Vantagens**:

- Modelo se adapta à tarefa específica
- Melhor performance que embeddings simples
- Ainda aproveita conhecimento pré-treinado

**Desvantagens**:

- Requer GPU/tempo de treinamento
- Mais complexo que abordagem 1

### 3. Classificação com LLM (Prompt Engineering)

**Biblioteca**: `google-genai` (Python SDK)

**Modelo**: `gemini-2.5-flash` (gratuito com API key do Google AI Studio)

**Funcionamento**:

- Criar prompt bem estruturado com:
  - Instrução clara da tarefa
  - Formato de saída desejado (JSON)
  - Few-shot examples (opcional)
- Configurar `response_mime_type` para JSON
- Definir schema de resposta

**Vantagens**:

- Zero-shot ou few-shot learning
- Não requer treinamento
- Flexível para múltiplas tarefas

**Desvantagens**:

- Depende de API externa
- Custo por inferência (mesmo gratuito tem limites)
- Latência de rede

## Métricas de Avaliação

Implementar usando `scikit-learn`:

1. **Matriz de Confusão**: Visualizar acertos/erros por classe
2. **Accuracy**: Percentual geral de acertos
3. **Precision**: Precisão por classe
4. **Recall**: Revocação por classe
5. **F1-Score**: Média harmônica de precision e recall
6. **ROC-AUC**: Área sob a curva ROC (para classificação binária/multiclasse)

## Comparação das Abordagens

Criar tabela comparativa:

| Métrica            | Embeddings+KNN | Fine-tuned DistilBERT | LLM (Gemini) |
|--------------------|----------------|-----------------------|--------------|
| Accuracy           | ?              | ?                     | ?            |
| Precision (média)  | ?              | ?                     | ?            |
| Recall (média)     | ?              | ?                     | ?            |
| F1-Score (média)   | ?              | ?                     | ?            |
| Tempo inferência   | ?              | ?                     | ?            |
| Requer treino      | Não            | Sim                   | Não          |

## Configuração

### 1. Instalar dependências

```bash
pipenv install
```

### 2. Configurar API key do Gemini

Crie um arquivo `.env` na raiz do projeto:

```env
GEMINI_API_KEY=sua_chave_aqui
```

### 3. Executar o projeto

```bash
pipenv run python src/b3/main.py
```

## Tecnologias e Dependências

**Python**: 3.11

**Gerenciador**: pipenv

**Bibliotecas principais**:

- `sentence-transformers`: Embeddings pré-treinados
- `transformers`: Fine-tuning e modelos Hugging Face
- `torch`: Backend para modelos
- `google-genai`: SDK para Gemini API
- `scikit-learn`: Métricas e KNN
- `numpy`, `pandas`: Manipulação de dados
- `matplotlib`, `seaborn`: Visualização
- `datasets`: Carregar datasets do Hugging Face
- `python-dotenv`: Gerenciar variáveis de ambiente
- `accelerate`: Otimização de treinamento

## Fluxo de Execução

### 1. Preparação dos dados

- Carregar dataset do Hugging Face
- Split train/test (80/20)
- Pré-processamento básico

### 2. Treinamento/Preparação

- **Abordagem 1**: Gerar embeddings do conjunto de treino
- **Abordagem 2**: Fine-tuning do DistilBERT
- **Abordagem 3**: Preparar prompts

### 3. Inferência

- Executar cada abordagem no conjunto de teste
- Medir tempo de inferência

### 4. Avaliação

- Calcular métricas para cada abordagem
- Gerar matriz de confusão
- Criar visualizações comparativas

### 5. Análise

- Comparar resultados
- Discutir trade-offs
- Identificar quando usar cada abordagem

## Aspectos Educacionais

O projeto demonstra conceitos do **Bloco C - Classificação de Dados**:

### Aula 1 - Classificação com IA

- Diferença entre classificação clássica (treinar do zero) vs IA (foundational models)
- Uso de modelos pré-treinados prontos para inferência
- Generalização através de grande volume de dados

### Aula 2 - Avaliação de sistemas de classificação

- Avaliação sistemática com conjunto de teste rotulado
- Matriz de confusão para identificar acertos e erros
- Múltiplas métricas: accuracy, precision, recall, F1
- Curva ROC e métrica AUC

### Aula 3 - Abordagens de classificação

- Três métodos práticos de classificação com IA
- Embeddings pré-treinados + classificador simples
- Fine-tuning de modelos pré-treinados
- LLMs com prompt engineering
- Trade-offs entre as abordagens

### Aula 4 - Exemplos de sistemas de classificação

- Implementação real de cada abordagem
- Classificação "ingênua" com embeddings + KNN
- Fine-tuning de DistilBERT para tarefa específica
- Uso de LLMs com técnicas de prompt engineering

## Implementação Detalhada

### Módulo: data_loader.py

**Responsabilidades**:

- Carregar dataset do Hugging Face
- Realizar split train/test
- Pré-processar textos (limpeza básica)
- Balanceamento de classes (se necessário)

**Funções principais**:

- `load_dataset()`: Carrega dataset
- `preprocess_text()`: Limpeza de texto
- `split_data()`: Train/test split
- `get_label_mapping()`: Mapeia labels para índices

### Módulo: embedding_classifier.py

**Responsabilidades**:

- Carregar modelo sentence-transformers
- Gerar embeddings para conjunto de treino
- Implementar classificador KNN
- Realizar predições

**Classe principal**:

```python
class EmbeddingClassifier:
    def __init__(self, model_name='all-MiniLM-L6-v2', k=5)
    def fit(self, texts, labels)
    def predict(self, texts)
    def predict_proba(self, texts)
```

### Módulo: finetuned_classifier.py

**Responsabilidades**:

- Carregar DistilBERT pré-treinado
- Adicionar camada de classificação
- Fine-tuning no dataset
- Salvar/carregar modelo treinado

**Classe principal**:

```python
class FinetunedClassifier:
    def __init__(self, model_name='distilbert-base-uncased', num_labels)
    def train(self, train_dataset, eval_dataset, epochs=3)
    def predict(self, texts)
    def save_model(self, path)
    def load_model(self, path)
```

### Módulo: llm_classifier.py

**Responsabilidades**:

- Configurar cliente Google GenAI
- Construir prompts estruturados
- Realizar classificação via API
- Tratamento de erros e rate limits

**Classe principal**:

```python
class LLMClassifier:
    def __init__(self, api_key, model_name='gemini-2.5-flash')
    def create_prompt(self, text, labels, few_shot_examples=None)
    def predict(self, texts, labels)
    def predict_batch(self, texts, labels, batch_size=10)
```

### Módulo: metrics.py

**Responsabilidades**:

- Calcular todas as métricas de classificação
- Gerar relatórios comparativos
- Calcular tempo de inferência

**Funções principais**:

```python
def calculate_metrics(y_true, y_pred, labels)
def generate_classification_report(y_true, y_pred, labels)
def calculate_roc_auc(y_true, y_pred_proba, labels)
def measure_inference_time(classifier, texts)
```

### Módulo: visualization.py

**Responsabilidades**:

- Plotar matriz de confusão
- Criar gráficos comparativos
- Gerar curvas ROC
- Visualizar distribuição de classes

**Funções principais**:

```python
def plot_confusion_matrix(y_true, y_pred, labels)
def plot_metrics_comparison(metrics_dict)
def plot_roc_curves(y_true, y_pred_proba, labels)
def plot_class_distribution(labels)
```

### Script: main.py

**Responsabilidades**:

- Orquestrar todo o pipeline
- Executar todas as abordagens
- Gerar comparações
- Salvar resultados

**Fluxo**:

1. Carregar dados
2. Treinar/preparar cada classificador
3. Avaliar no conjunto de teste
4. Comparar resultados
5. Gerar visualizações
6. Salvar resultados

### Arquivo: config.py

**Configurações**:

```python
# Datasets
DATASET_NAME = 'emotion'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Modelos
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
FINETUNED_MODEL = 'distilbert-base-uncased'
LLM_MODEL = 'gemini-2.5-flash'

# KNN
K_NEIGHBORS = 5

# Fine-tuning
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5

# Paths
DATA_DIR = 'data/'
MODELS_DIR = 'models/'
RESULTS_DIR = 'results/'
```

## Resultados Esperados

Após executar o projeto, você terá:

1. ✅ Três classificadores implementados e funcionais
2. ✅ Comparação quantitativa (métricas) entre as abordagens
3. ✅ Visualizações (matriz de confusão, gráficos comparativos)
4. ✅ Entendimento prático de quando usar cada abordagem
5. ✅ Experiência com ferramentas modernas de IA

## Cronograma de Implementação

### Fase 1: Setup e Dados (1-2 horas)

- [x] Criar estrutura de pastas
- [ ] Configurar dependências
- [ ] Implementar `data_loader.py`
- [ ] Criar `.env.example`

### Fase 2: Abordagem 1 - Embeddings (2-3 horas)

- [ ] Implementar `embedding_classifier.py`
- [ ] Testar com dataset pequeno
- [ ] Validar predições

### Fase 3: Abordagem 2 - Fine-tuning (3-4 horas)

- [ ] Implementar `finetuned_classifier.py`
- [ ] Configurar treinamento
- [ ] Executar fine-tuning
- [ ] Salvar modelo treinado

### Fase 4: Abordagem 3 - LLM (2-3 horas)

- [ ] Implementar `llm_classifier.py`
- [ ] Testar conexão com API
- [ ] Otimizar prompts
- [ ] Implementar tratamento de erros

### Fase 5: Avaliação (2-3 horas)

- [ ] Implementar `metrics.py`
- [ ] Implementar `visualization.py`
- [ ] Gerar relatórios comparativos

### Fase 6: Integração e Documentação (1-2 horas)

- [ ] Implementar `main.py`
- [ ] Criar README.md
- [ ] Documentar resultados
- [ ] Análise final

**Total estimado**: 11-17 horas

## Considerações Finais

Este projeto oferece uma visão prática e completa sobre classificação de dados com IA, cobrindo desde técnicas simples até abordagens avançadas com LLMs. A comparação sistemática permite entender os trade-offs de cada método e tomar decisões informadas em projetos reais.
