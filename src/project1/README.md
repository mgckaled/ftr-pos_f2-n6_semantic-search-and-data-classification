<!--markdownlint-disable-->

# Projeto: Classificação de Gatos e Cachorros

Implementação de três abordagens para classificação de imagens de gatos e cachorros:
1. **Embeddings CLIP + KNN**: Usando Sentence-Transformers e K-Nearest Neighbors
2. **LLM (Gemini)**: Classificação zero-shot com Large Language Model
3. **Ensemble**: Combinação inteligente dos dois métodos

## Estrutura do Projeto

```
src/project1/
├── classification.ipynb    # Notebook principal (única arquivo necessário)
├── .env.example            # Template para configuração de API key
├── README.md               # Este arquivo
├── images/                 # Dataset (criado automaticamente)
├── cache/                  # Cache de embeddings e checkpoints
└── results/                # Visualizações geradas
```

## Pré-requisitos

### 1. Instalar Dependências com Pipenv

Na raiz do repositório, execute:

```bash
pipenv install
```

**Dependências necessárias** (verificar no Pipfile):
- ✅ `sentence-transformers`
- ✅ `scikit-learn`
- ✅ `google-genai`
- ✅ `matplotlib`, `seaborn`
- ✅ `pandas`, `numpy`
- ✅ `jupyter`, `notebook`
- ⚠️ **Verificar se faltam**: `kagglehub`, `pillow`, `tqdm`, `python-dotenv`

Se alguma dependência estiver faltando, adicione:

```bash
pipenv install kagglehub pillow tqdm python-dotenv
```

### 2. Configurar API Key do Gemini

1. Obtenha uma API key gratuita em: https://aistudio.google.com/app/apikey
2. Copie o arquivo de exemplo:
   ```bash
   cp src/project1/.env.example src/project1/.env
   ```
3. Edite `.env` e adicione sua API key:
   ```
   GEMINI_API_KEY=sua_chave_aqui
   ```

## Executar o Projeto

### Ativar o ambiente virtual

```bash
pipenv shell
```

### Iniciar Jupyter Notebook

```bash
cd src/project1
jupyter notebook classification.ipynb
```

### Executar as células

Execute as células sequencialmente. O notebook é dividido em 5 fases:

1. **Fase 1**: Setup e configuração
2. **Fase 2**: Download e preparação do dataset (~800MB)
3. **Fase 3**: Embeddings CLIP, KNN e visualização T-SNE
4. **Fase 4**: Classificação com LLM (Gemini) - ~7-8 minutos
5. **Fase 5**: Serviço unificado e comparação

## Parâmetros Configuráveis

Todos os parâmetros estão centralizados na **Célula 2**:

```python
# Dataset
NUM_SAMPLES_PER_CLASS = 500      # Imagens por classe (total: 1000)
LLM_TEST_SUBSET = 100            # Subset para LLM (rate limit)

# Modelos
CLIP_MODEL = 'clip-ViT-B-32'     # Modelo de embeddings
GEMINI_MODEL = 'gemini-2.0-flash-exp'

# T-SNE
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000

# KNN
KNN_K = 5
KNN_METRIC = 'cosine'

# Rate Limiting (Gemini Free Tier)
GEMINI_REQUESTS_PER_MINUTE = 15  # Limite: 15 RPM
SLEEP_BETWEEN_REQUESTS = 4.5     # Segundos entre requests
```

## Rate Limits do Gemini API (Tier Gratuito)

| Métrica | Limite | Impacto |
|---------|--------|---------|
| RPM (Requests/Min) | 15 | 1 request a cada 4 segundos |
| RPD (Requests/Day) | 200 | Máx 200 imagens por dia |
| TPM (Tokens/Min) | 1M | Não limitante para imagens |

**Estratégia implementada**:
- Subset de teste LLM: 100 imagens (50% do limite diário)
- Sleep de 4.5s entre requests (margem de segurança)
- Sistema de checkpoint a cada 20 requests
- Possibilidade de resume em caso de interrupção

## Tempo de Execução

- **Primeira execução** (com downloads):
  - Download dataset: ~3-5 min
  - Geração de embeddings: ~2-3 min
  - T-SNE: ~1-2 min
  - KNN: instantâneo
  - LLM: ~7-8 min (100 imagens × 4.5s)
  - **Total**: ~15-20 min

- **Execuções subsequentes** (com cache):
  - LLM: ~7-8 min
  - Resto: ~1-2 min
  - **Total**: ~10 min

## Outputs Gerados

Todas as visualizações são salvas em `results/`:

1. `sample_images.png` - Grid 4×4 de amostras do dataset
2. `tsne_visualization.png` - Visualização T-SNE dos embeddings
3. `knn_confusion_matrix.png` - Matriz de confusão do KNN
4. `llm_confusion_matrix.png` - Matriz de confusão do LLM
5. `llm_errors.png` - Exemplos de erros do LLM
6. `comparison_knn_llm.png` - Comparação de métricas
7. `demo_multi_method.png` - Demonstração multi-método

## Cache e Checkpoints

O sistema implementa cache inteligente em `cache/`:

- `embeddings.pkl` - Cache dos embeddings CLIP (evita recalcular)
- `tsne_embeddings.pkl` - Cache do T-SNE
- `llm_predictions.pkl` - Checkpoint das predições LLM (permite resume)

**Deletar cache**: Se quiser recalcular tudo, delete a pasta `cache/`.

## Troubleshooting

### Erro: "GEMINI_API_KEY não encontrada"
- Verifique se criou o arquivo `.env`
- Verifique se a API key está correta
- Reinicie o kernel do Jupyter

### Erro: "Rate limit exceeded"
- Aguarde alguns minutos
- O sistema tem checkpoint automático, pode continuar depois
- Reduza `LLM_TEST_SUBSET` para menos de 100

### Erro ao carregar imagens
- Verifique conexão com internet (download do Kaggle)
- Delete pasta `images/` e tente novamente
- Verifique espaço em disco (~1GB necessário)

### Embeddings muito lentos
- Primeira execução sempre demora (download do modelo CLIP ~600MB)
- Execuções subsequentes usam cache
- Reduza `NUM_SAMPLES_PER_CLASS` para testes rápidos

## Melhorias Futuras

- [ ] Suporte a outros datasets (via CLI argument)
- [ ] Fine-tuning de modelo CLIP para domínio específico
- [ ] Implementação de HNSW para busca ANN escalável
- [ ] API REST para deploy em produção
- [ ] Suporte a outros LLMs (Claude, GPT-4V)
- [ ] Ensemble com voting mais sofisticado

## Contato

Para dúvidas ou sugestões, abra uma issue no repositório.
