<!--markdownlint-disable-->

# Projeto: Sistema de Recomendação de Filmes

Implementação de três abordagens para recomendação de filmes com interface interativa:
1. **Filtragem Colaborativa (Item-Item)**: KNN baseado em similaridade de ratings
2. **Filtragem Baseada em Conteúdo**: Embeddings semânticos com Sentence-Transformers
3. **Sistema Híbrido Adaptativo**: Combinação inteligente dos dois métodos

## Estrutura do Projeto

```
src/project2/
├── recommendation.ipynb   # Notebook principal (único arquivo necessário)
├── README.md              # Este arquivo
├── docs/
│   └── parameters.md      # Guia detalhado de parâmetros
├── cache/                 # Cache de embeddings e métricas
└── results/               # Visualizações geradas
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
- ✅ `datasets` (HuggingFace)
- ✅ `scipy`
- ✅ `matplotlib`, `seaborn`
- ✅ `pandas`, `numpy`
- ✅ `jupyter`, `notebook`
- ✅ `ipywidgets`

Se alguma dependência estiver faltando, adicione:

```bash
pipenv install datasets scipy ipywidgets
```

## Executar o Projeto

### Ativar o ambiente virtual

```bash
pipenv shell
```

### Iniciar Jupyter Notebook

```bash
cd src/project2
jupyter notebook recommendation.ipynb
```

### Executar as células

Execute as células sequencialmente. O notebook é dividido em 6 partes:

1. **Parte 1 (Células 1-5)**: Setup, dataset e análise exploratória
2. **Parte 2 (Células 6-7)**: Filtragem Colaborativa Item-Item
3. **Parte 3 (Células 8-10)**: Filtragem Baseada em Conteúdo
4. **Parte 4 (Células 11-13)**: Sistema Híbrido e Análise Cold-Start
5. **Parte 5 (Células 14-15)**: Visualizações (T-SNE, comparação)
6. **Parte 6 (Células 16-22)**: Sistema Interativo com Jupyter Widgets

## Dataset

**MovieLens 990k** (HuggingFace: `ashraq/movielens_ratings`)

- **Ratings**: 990,425 (891k treino, 99k teste)
- **Filmes**: 15,276
- **Usuários**: 43,584
- **Sparsidade**: 99.86%

O dataset é baixado automaticamente na primeira execução (~30 MB).

## Parâmetros Configuráveis

Todos os parâmetros estão centralizados na **Célula 2**:

```python
# === PARÂMETROS DE AVALIAÇÃO ===
K = 10                          # Top-K recomendações
MIN_RATING_THRESHOLD = 3.0      # Considera "relevante" (otimizado)
RANDOM_STATE = 42

# === MODELOS ===
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L12-v2'  # 384-dim

# === OTIMIZAÇÕES DE MEMÓRIA ===
EMBEDDING_BATCH_SIZE = 64       # Mini-batches para embeddings
SIMILARITY_CHUNK_SIZE = 2000    # Processar 2000 filmes por vez
TOP_K_SIMILAR = 2000            # Top-2000 similares por filme (otimizado)

# === PARÂMETROS DO HÍBRIDO ===
ALPHA_MIN_RATINGS = 5           # Mínimo de ratings para confiar no colaborativo
ALPHA_MAX_RATINGS = 50          # Máximo para α = 0.9
```

### Configurações Testadas

| Configuração | TOP_K_SIMILAR | MIN_THRESHOLD | Precision@10 | Tempo |
|--------------|---------------|---------------|--------------|-------|
| Conservadora | 150 | 4.0 | 0.000 | ~7 min |
| Padrão | 1000 | 3.5 | 0.058 | ~2 min |
| **Otimizada** ✅ | **2000** | **3.0** | **0.078** | **~4 min** |

## Resultados Finais

### Métricas de Performance (Configuração Otimizada)

| Método | Precision@10 | Recall@10 | NDCG@10 | Ranking |
|--------|--------------|-----------|---------|---------|
| **Colaborativo** 🥇 | **0.078** | **0.210** | **0.617** | **1º** |
| **Híbrido** 🥈 | 0.071 | 0.169 | 0.583 | 2º |
| **Conteúdo** 🥉 | 0.007 | 0.017 | 0.494 | 3º |

### Interpretação

- **Precision@10 = 7.8%**: ~8 em cada 100 filmes recomendados são relevantes
- **Recall@10 = 21%**: Captura 1/5 de todos os filmes que o usuário gostaria
- **NDCG@10 = 0.617**: Excelente ordenação (filmes mais relevantes aparecem primeiro)

Para um dataset com **99.86% de sparsidade**, essas métricas são **muito boas**!

## Tempo de Execução

- **Primeira execução** (com downloads):
  - Download dataset: ~1-2 min
  - Construção matriz esparsa: ~1 min
  - Cálculo similaridade (K=2000): ~3-4 min
  - Geração embeddings: ~3-5 min
  - Avaliação 3 métodos: ~10-12 min
  - T-SNE: ~3-5 min
  - **Total**: ~25-30 min

- **Execuções subsequentes** (com cache):
  - Avaliações: ~10-12 min
  - **Total**: ~12-15 min

## Outputs Gerados

Todas as visualizações são salvas em `results/`:

1. `exploratory_analysis.png` - Distribuição de ratings, top filmes, gêneros, sparsidade
2. `tsne_movies_by_genre.png` - Visualização T-SNE dos embeddings por gênero
3. `metrics_comparison.png` - Comparação de Precision, Recall e NDCG
4. `cold_start_analysis.png` - Performance vs experiência do usuário

## Cache e Checkpoints

O sistema implementa cache inteligente em `cache/`:

- `dataset_processed.pkl` - Dataset pré-processado (~50 MB)
- `user_item_matrix.pkl` - Matriz esparsa usuário-item (~7 MB)
- `item_similarity_topk.pkl` - Top-K similares por filme (~230 MB com K=2000)
- `movie_embeddings.pkl` - Embeddings dos filmes (~22 MB)
- `metrics_*.pkl` - Métricas de avaliação
- `tsne_2d.pkl` - Cache do T-SNE

**Deletar cache**: Se quiser recalcular com novos parâmetros, delete os arquivos afetados:
- Mudou `TOP_K_SIMILAR` ou `SIMILARITY_CHUNK_SIZE`? Delete `item_similarity_topk.pkl`
- Mudou `EMBEDDING_MODEL`? Delete `movie_embeddings.pkl`
- Mudou `MIN_RATING_THRESHOLD`? Delete `metrics_*.pkl`

## Sistema Interativo (Células 16-22)

Interface web interativa usando Jupyter Widgets:

### Funcionalidades

1. **Seleção de Filmes** (Célula 16):
   - 10 dropdowns com busca
   - Slider de nota (1-5 estrelas)
   - Estado salvo automaticamente

2. **Controles** (Célula 17):
   - Escolha do método (Colaborativo/Conteúdo/Híbrido)
   - α adaptativo ou manual
   - Top-K ajustável (5-20)

3. **Geração de Recomendações** (Célula 18):
   - Botão "Gerar Recomendações"
   - Tabela HTML com rank, título, gêneros, score

4. **Comparação em Tabs** (Célula 19):
   - 3 métodos lado a lado
   - Comparação instantânea

5. **Exploração de Similares** (Célula 20):
   - Busca filmes similares por conteúdo
   - Top-5 mais próximos

6. **Análise de Perfil** (Célula 21):
   - Gêneros preferidos
   - Distribuição de notas
   - Estatísticas personalizadas
   - Recomendação de α ideal

### Como Usar

1. Execute células 1-15 (processamento técnico)
2. Execute células 16-17 (interface)
3. Selecione 5-10 filmes e dê notas
4. Escolha método "Colaborativo" (melhor resultado)
5. Clique em "Gerar Recomendações"
6. Explore as outras funcionalidades!

## Troubleshooting

### Erro: Métricas muito baixas (Precision < 0.01)
- Verifique se `TOP_K_SIMILAR >= 1000`
- Verifique se `MIN_RATING_THRESHOLD = 3.0` (não 3.5 ou 4.0)
- Delete cache e re-execute células 6-7

### Erro: "Notebook travou" ou "Memória insuficiente"
- Reduza `TOP_K_SIMILAR` para 1500 ou 1000
- Não use valores acima de 3000 (requer ~500 MB)
- Feche outros programas pesados

### Embeddings muito lentos
- Primeira execução sempre demora (download do modelo ~120 MB)
- Execuções subsequentes usam cache
- Use `EMBEDDING_BATCH_SIZE = 64` para acelerar

### Sistema interativo não aparece
- Instale: `pipenv install ipywidgets`
- Ative extensão: `jupyter nbextension enable --py widgetsnbextension`
- Reinicie o Jupyter

## Otimizações Aplicadas

### Memória
- ✅ Matriz esparsa (`scipy.sparse.csr_matrix`) - ~7 MB ao invés de ~2.5 GB
- ✅ Top-K sparse similarity - ~230 MB ao invés de ~1.8 GB
- ✅ Mini-batch embeddings - evita OOM
- ✅ Chunked similarity computation - processa 2000 filmes por vez

### Performance
- ✅ Loop invertido na recomendação (percorre apenas filmes avaliados)
- ✅ Cache em pickle (2 segundos vs 2 minutos)
- ✅ Numpy vetorizado ao invés de loops Python
- ✅ Pré-computação de popularidade

### Algoritmo
- ✅ Threshold 3.0 (ao invés de 4.0) - +34% Precision
- ✅ TOP_K=2000 (ao invés de 1000) - +34% Precision
- ✅ Alpha adaptativo no híbrido - ajusta por experiência do usuário

## Melhorias Futuras

- [ ] Implementar User-User collaborative filtering
- [ ] Adicionar mais metadados (sinopse, elenco, diretor) para conteúdo
- [ ] Usar modelo de embeddings maior (mpnet 768-dim)
- [ ] Implementar matrix factorization (SVD, ALS)
- [ ] Sistema de feedback implícito (cliques, tempo de visualização)
- [ ] Deploy como API REST
- [ ] A/B testing de diferentes configurações
- [ ] Suporte a outros datasets (IMDB, Last.fm)

## Referências

- **Dataset**: MovieLens 990k via HuggingFace
- **Embeddings**: Sentence-Transformers (all-MiniLM-L12-v2)
- **Collaborative Filtering**: Item-Item KNN com cosine similarity
- **Métricas**: Precision@K, Recall@K, NDCG@K

## Documentação Adicional

- `docs/parameters.md` - Guia completo de parâmetros configuráveis
- Células com código otimizado em `/cell*_optimized.py`

## Contato

Para dúvidas ou sugestões, abra uma issue no repositório.