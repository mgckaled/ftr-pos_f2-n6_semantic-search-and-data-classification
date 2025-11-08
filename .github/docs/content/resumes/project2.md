# Projeto 2: Sistema Interativo de Recomendação de Filmes

## Contexto

Sistemas de recomendação representam uma das aplicações mais impactantes de machine learning na indústria, responsáveis por bilhões de dólares em receita para empresas como Netflix, Amazon e Spotify. Este projeto implementa e compara três abordagens fundamentais de sistemas de recomendação aplicadas ao domínio cinematográfico, culminando em uma interface interativa onde o usuário pode explorar recomendações personalizadas em tempo real.

## Objetivo

Desenvolver um sistema completo de recomendação de filmes que:

1. **Implementa 3 abordagens técnicas** com avaliação rigorosa (Parte 1)
2. **Oferece interface interativa** para usuários explorarem recomendações (Parte 2)
3. **Permite comparação em tempo real** entre os métodos
4. **Otimizado para RAM limitada** (4-6GB) trocando tempo por memória

## Dataset

**Fonte**: `ashraq/movielens_ratings` (HuggingFace)

**Características**:
- **990.425 ratings** de 44.100 usuários sobre 15.600 filmes
- Escala: 0.5 a 5.0 estrelas (incrementos de 0.5)
- **Metadados ricos**:
  - `title`: Nome do filme
  - `genres`: Gêneros (pipe-separated, ex: "Action|Sci-Fi|Thriller")
  - `imdbId`, `tmdbId`: Identificadores externos
  - `posters`: URLs de imagens
- Sparsidade: ~99.86% (média de 22 ratings por usuário)
- **Splits**:
  - Train: 891k ratings (90%)
  - Validation: 99k ratings (10%)

**Tamanho**: ~30MB download

---

## Otimizações de Memória

### Estratégias (Tempo × RAM)

| Operação | Abordagem Naive | Abordagem Otimizada | Ganho RAM |
|----------|-----------------|---------------------|-----------|
| Similaridade Item-Item | Matriz completa (15.6k×15.6k) | ANN top-100 por filme | ~1GB → 50MB |
| Embeddings | Batch completo | Mini-batches (32) | Pico 2GB → 500MB |
| Cache | Sem cache | Pickle agressivo | Re-run: 30min → 10s |
| Matriz Usuário-Item | Densa | Scipy sparse CSR | 10GB → 10MB |

**Trade-off**: Primeira execução ~30-40 min, re-runs ~1 min

**RAM Total Estimada**: 4-6GB (confortável com 16GB total)

---

## Abordagens Técnicas

### Abordagem 1: Filtragem Colaborativa Item-Item

**Conceito**: "Usuários que gostaram do filme A também gostaram do filme B"

**Algoritmo Otimizado**:
```python
# Matriz esparsa
R = csr_matrix((ratings, (users, movies)))  # ~10MB

# Similaridade por chunks (evita 15.6k×15.6k na RAM)
for i in range(0, n_movies, chunk_size=1000):
    chunk_sim = cosine_similarity(R[:, i:i+1000].T, R.T)
    # Guardar apenas top-100 similares
    top_k_indices = np.argsort(chunk_sim, axis=1)[:, -100:]
    sparse_similarity[i:i+1000] = top_k_indices

# Cache para re-uso
pickle.dump(sparse_similarity, 'cache/item_similarity.pkl')
```

**Vantagens**:
- Captura padrões complexos
- Funciona bem para usuários ativos
- Interpretável

**Limitações**:
- Cold-start para novos filmes/usuários

---

### Abordagem 2: Filtragem Baseada em Conteúdo com Embeddings

**Conceito**: "Recomendar filmes semanticamente similares"

**Algoritmo Otimizado**:
```python
# Embeddings em mini-batches
model = SentenceTransformer('all-MiniLM-L6-v2')  # 80MB
embeddings = []

for i in range(0, len(movies), batch_size=32):
    batch = movies[i:i+32]
    batch_emb = model.encode(batch, show_progress_bar=True)
    embeddings.append(batch_emb)

    # Liberar memória a cada 1000 filmes
    if i % 1000 == 0:
        gc.collect()

# Cache
pickle.dump(np.vstack(embeddings), 'cache/embeddings.pkl')
```

**Modelo**: `sentence-transformers/all-MiniLM-L6-v2`
- 384 dimensões
- ~80MB
- Rápido em CPU

**Vantagens**:
- Resolve cold-start de novos filmes
- Captura semântica
- Zero custo de API

---

### Abordagem 3: Sistema Híbrido Adaptativo

**Conceito**: Combinar pontos fortes de ambas

**Algoritmo**:
```python
# α adaptativo baseado em experiência do usuário
num_ratings = user_rating_count[user_id]
alpha = min(0.9, 0.3 + 0.6 * (num_ratings / 50))

# Combinar scores normalizados
score_hybrid = alpha * score_colab + (1-alpha) * score_content
```

**Regras**:
- Usuário novo (< 5 ratings): α ≈ 0.3 → favorece conteúdo
- Usuário casual (10-20 ratings): α ≈ 0.5 → balanceado
- Usuário ativo (50+ ratings): α ≈ 0.9 → favorece colaborativo

---

## Estrutura do Projeto

```
src/project2/
├── recommendation.ipynb          # Notebook principal (21 células)
├── README.md                      # Documentação completa
├── INSTALL.md                     # Guia de instalação
├── .env.example                   # Template (não usado aqui)
├── cache/                         # Pickles para re-uso
│   ├── dataset.pkl               # Dataset processado
│   ├── embeddings.pkl            # Embeddings de filmes (15.6k×384)
│   ├── item_similarity.pkl       # Top-100 similares por filme
│   └── user_profiles.pkl         # Perfis pré-computados
└── results/                       # Visualizações
    ├── tsne_movies_by_genre.png  # Clusters de filmes
    ├── metrics_comparison.png    # Precision/Recall/NDCG
    ├── cold_start_analysis.png   # Performance vs nº ratings
    └── interactive_demo.png       # Screenshot da interface
```

---

## Pipeline de Execução

### PARTE 1: Implementação Técnica (Células 1-15)

**Fase 1: Setup** (Células 1-4)
- Célula 1: Importações
- Célula 2: Configuração (paths, parâmetros de otimização)
- Célula 3: Download e cache do dataset (HuggingFace)
- Célula 4: Análise exploratória (distribuição, sparsidade, top filmes)

**Fase 2: Filtragem Colaborativa** (Células 5-7)
- Célula 5: Construir matriz esparsa usuário-item
- Célula 6: Calcular similaridade item-item (chunked, top-100)
- Célula 7: Implementar recomendação + avaliar (Precision@10, Recall@10, NDCG@10)

**Fase 3: Baseada em Conteúdo** (Células 8-10)
- Célula 8: Gerar embeddings (mini-batches, cache)
- Célula 9: Construir perfis de usuários (média ponderada)
- Célula 10: Implementar recomendação + avaliar

**Fase 4: Sistema Híbrido** (Células 11-13)
- Célula 11: Implementar ensemble com α adaptativo
- Célula 12: Avaliar performance
- Célula 13: Análise de cold-start (usuários com 1, 5, 10, 20, 50+ ratings)

**Fase 5: Visualizações** (Células 14-15)
- Célula 14: T-SNE dos embeddings (colorir por gênero)
- Célula 15: Comparação gráfica das 3 abordagens

---

### PARTE 2: Sistema Interativo (Células 16-21)

**Célula 16: Interface de Seleção**
```python
import ipywidgets as widgets

# Busca de filmes
search_box = widgets.Text(
    description='Buscar:',
    placeholder='Digite o nome do filme...'
)

# Seleção de filmes (10 slots)
movie_selectors = []
for i in range(10):
    movie_dropdown = widgets.Dropdown(
        options=[''] + movie_titles,
        description=f'Filme {i+1}:'
    )
    rating_slider = widgets.IntSlider(
        min=1, max=5, value=4,
        description='Nota:'
    )
    movie_selectors.append((movie_dropdown, rating_slider))
```

**Célula 17: Controles do Sistema**
```python
# Método de recomendação
method_selector = widgets.RadioButtons(
    options=['Colaborativo', 'Conteúdo', 'Híbrido'],
    description='Método:',
    value='Híbrido'
)

# Peso do ensemble (ativo apenas se Híbrido)
alpha_slider = widgets.FloatSlider(
    min=0.0, max=1.0, step=0.05, value=0.7,
    description='α (Colab):',
    disabled=False
)

# Número de recomendações
k_slider = widgets.IntSlider(
    min=5, max=20, value=10,
    description='Top-K:'
)
```

**Célula 18: Geração de Recomendações**
```python
output = widgets.Output()

def on_generate_click(b):
    with output:
        output.clear_output()

        # Obter seleções do usuário
        user_ratings = {}
        for movie_drop, rating_slide in movie_selectors:
            if movie_drop.value != '':
                movie_id = movie_name_to_id[movie_drop.value]
                user_ratings[movie_id] = rating_slide.value

        if len(user_ratings) < 3:
            print("⚠️ Selecione pelo menos 3 filmes!")
            return

        # Gerar recomendações
        method = method_selector.value
        k = k_slider.value

        if method == 'Colaborativo':
            recs = recommend_collaborative_interactive(user_ratings, k)
        elif method == 'Conteúdo':
            recs = recommend_content_interactive(user_ratings, k)
        else:
            alpha = alpha_slider.value
            recs = recommend_hybrid_interactive(user_ratings, alpha, k)

        # Exibir resultados com HTML
        display(HTML(format_recommendations_html(recs)))

generate_btn = widgets.Button(
    description='🎬 Gerar Recomendações',
    button_style='success'
)
generate_btn.on_click(on_generate_click)
```

**Célula 19: Comparação em Tabs**
```python
# Tabs para comparar as 3 abordagens
tab = widgets.Tab()

outputs = [widgets.Output() for _ in range(3)]
tab.children = outputs
tab.titles = ['Colaborativo', 'Conteúdo', 'Híbrido']

# Gerar nas 3 simultaneamente
user_ratings = get_current_user_ratings()
k = k_slider.value

methods = [
    ('Colaborativo', recommend_collaborative_interactive),
    ('Conteúdo', recommend_content_interactive),
    ('Híbrido', lambda ur, k: recommend_hybrid_interactive(ur, 0.7, k))
]

for i, (name, func) in enumerate(methods):
    with outputs[i]:
        recs = func(user_ratings, k)
        display(HTML(format_recommendations_html(recs)))

display(tab)
```

**Célula 20: Exploração Interativa**
```python
# Selecionar um filme recomendado para explorar
selected_movie = widgets.Dropdown(
    options=recommended_movie_titles,
    description='Explorar:'
)

exploration_output = widgets.Output()

def on_movie_explore(change):
    with exploration_output:
        exploration_output.clear_output()

        movie_id = movie_name_to_id[change['new']]

        # 1. Filmes similares (top-5)
        similar = get_top_similar_movies(movie_id, k=5)
        print("🎯 Filmes Similares:")
        for title, score in similar:
            print(f"  - {title} (score: {score:.3f})")

        # 2. T-SNE com highlight
        plot_tsne_with_highlight(movie_id)

        # 3. Explicação da recomendação
        explanation = generate_explanation(movie_id, user_ratings)
        display(HTML(f"<div style='background:#f0f0f0; padding:10px'>{explanation}</div>"))

selected_movie.observe(on_movie_explore, 'value')
display(widgets.VBox([selected_movie, exploration_output]))
```

**Célula 21: Análise de Perfil do Usuário**
```python
# Gêneros preferidos (baseado nas notas dadas)
def plot_user_genre_preferences(user_ratings):
    genre_scores = defaultdict(list)

    for movie_id, rating in user_ratings.items():
        genres = movie_genres[movie_id].split('|')
        for genre in genres:
            genre_scores[genre].append(rating)

    genre_avg = {g: np.mean(scores) for g, scores in genre_scores.items()}

    plt.figure(figsize=(10, 5))
    plt.barh(list(genre_avg.keys()), list(genre_avg.values()))
    plt.xlabel('Nota Média')
    plt.title('Seus Gêneros Preferidos')
    plt.show()

# Distribuição de ratings
def plot_rating_distribution(user_ratings):
    plt.figure(figsize=(8, 5))
    plt.hist(list(user_ratings.values()), bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    plt.xlabel('Nota')
    plt.ylabel('Frequência')
    plt.title('Distribuição das Suas Notas')
    plt.show()

# Diversidade das recomendações
def plot_recommendation_diversity(recommendations):
    genres_in_recs = []
    for movie_id in recommendations:
        genres_in_recs.extend(movie_genres[movie_id].split('|'))

    genre_counts = Counter(genres_in_recs)

    plt.figure(figsize=(10, 5))
    plt.bar(genre_counts.keys(), genre_counts.values())
    plt.xlabel('Gênero')
    plt.ylabel('Frequência nas Recomendações')
    plt.title('Diversidade das Recomendações')
    plt.xticks(rotation=45)
    plt.show()

# Executar análises
user_ratings = get_current_user_ratings()
plot_user_genre_preferences(user_ratings)
plot_rating_distribution(user_ratings)
plot_recommendation_diversity(current_recommendations)
```

---

## Métricas de Avaliação

### Métricas Offline (Validation Set)

**1. Precision@K**
```python
Precision@K = (Recomendados relevantes) / K
# Relevante = rating ≥ 4.0
```

**2. Recall@K**
```python
Recall@K = (Recomendados relevantes) / (Total de relevantes do usuário)
```

**3. NDCG@K** (Normalized Discounted Cumulative Gain)
```python
DCG@K = Σ(rel_i / log2(i+1))  para i=1..K
NDCG@K = DCG@K / IDCG@K
```

**4. Coverage**
```python
Coverage = (Filmes únicos recomendados) / (Total de filmes)
```

**5. Diversidade de Gêneros**
```python
Diversity = 1 - Gini(distribuição de gêneros nas recomendações)
```

---

## Requisitos Técnicos

### Dependências

**Já no Pipfile** ✅:
- `datasets` - HuggingFace datasets
- `sentence-transformers` - Embeddings
- `scikit-learn` - Métricas, similaridade
- `scipy` - Matrizes esparsas
- `pandas`, `numpy` - Manipulação
- `matplotlib`, `seaborn` - Visualizações
- `jupyter`, `notebook` - Ambiente

**Adicionar**:
- `ipywidgets` - Interface interativa

### Recursos Computacionais

**RAM**: 4-6GB durante execução
- Matriz esparsa: ~10MB
- Embeddings: ~25MB (15.6k × 384 × 4 bytes)
- Similaridade top-100: ~50MB
- Cache total: ~200MB

**CPU**: Suficiente (qualquer i5/i7 moderno)
- Primeira execução: ~30-40 min
  - Download dataset: ~1 min
  - Embeddings (batched): ~15-20 min
  - Similaridade (chunked): ~10-15 min
  - Métricas: ~5 min
- Re-runs com cache: ~1 min

**Armazenamento**: ~500MB
- Dataset: 30MB
- Cache: 200MB
- Results: 50MB

**Custo**: Zero (tudo local)

---

## Resultados Esperados

### Métricas Offline (Validation Set)

| Método | Precision@10 | Recall@10 | NDCG@10 | Coverage |
|--------|--------------|-----------|---------|----------|
| Colaborativo | 0.35-0.45 | 0.20-0.30 | 0.40-0.50 | 60% |
| Conteúdo | 0.25-0.35 | 0.15-0.25 | 0.30-0.40 | 85% |
| Híbrido | 0.40-0.55 | 0.25-0.35 | 0.45-0.60 | 75% |

### Experiência Interativa

**Cenário 1**: Usuário seleciona filmes de ação
- Colaborativo: Blockbusters populares (Fast & Furious, Marvel)
- Conteúdo: Ação nichada (John Wick, Mad Max)
- Híbrido: Mix balanceado

**Cenário 2**: Usuário eclético (ação + romance + ficção)
- Colaborativo: Prioriza padrões majoritários
- Conteúdo: Alta diversidade de gêneros
- Híbrido: Diversidade moderada

---

## Diferenciais do Projeto

✅ **Interatividade Total**: Jupyter Widgets responsivos
✅ **Comparação em Tempo Real**: Tabs com 3 abordagens
✅ **Explicabilidade**: Mostra "por quê" de cada recomendação
✅ **Otimizado para RAM**: 4-6GB (chunking, batching, ANN)
✅ **Cache Inteligente**: Primeira execução lenta, re-runs instantâneos
✅ **Dataset Real**: MovieLens 990k (industrial)
✅ **Zero Custo**: Sem APIs pagas
✅ **Análise Profunda**: T-SNE, métricas, cold-start

---

## Referências

- Harper & Konstan (2015). "The MovieLens Datasets: History and Context"
- Ricci et al. (2011). "Recommender Systems Handbook"
- Koren et al. (2009). "Matrix Factorization Techniques"
- Burke (2002). "Hybrid Recommender Systems"
- Dataset: https://huggingface.co/datasets/ashraq/movielens_ratings
- GroupLens: https://grouplens.org/datasets/movielens/
