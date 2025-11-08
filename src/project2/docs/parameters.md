<!--markdownlint-disable-->

# Guia de Parâmetros do Sistema de Recomendação

Este documento descreve todos os parâmetros configuráveis na **Célula 2** do notebook e seus impactos em performance, memória e qualidade das recomendações.

---

## Índice

1. [Parâmetros de Avaliação](#parâmetros-de-avaliação)
2. [Modelos](#modelos)
3. [Otimizações de Memória](#otimizações-de-memória)
4. [Parâmetros do Sistema Híbrido](#parâmetros-do-sistema-híbrido)
5. [Cenários de Uso](#cenários-de-uso)
6. [Troubleshooting](#troubleshooting)

---

## Parâmetros de Avaliação

### `K = 10`

**O que faz:** Número de recomendações retornadas (Top-K)

**Impacto:**
- **Performance:** Mínimo (não afeta tempo de processamento)
- **Memória:** Desprezível
- **Qualidade:** Afeta métricas de Precision/Recall/NDCG

**Valores recomendados:**

| Valor | Uso | Trade-offs |
|-------|-----|------------|
| `5` | Testes rápidos, interfaces móveis | Precision alta, Recall baixo |
| `10` | **Padrão** - Balanceado | Bom equilíbrio |
| `20` | Exploração, usuários exigentes | Recall alto, Precision pode cair |
| `50` | Análise de diversidade | Métricas perdem significado |

**Recomendação:** Manter `K = 10` para avaliação. Ajustar no widget interativo conforme necessário.

---

### `MIN_RATING_THRESHOLD = 4.0`

**O que faz:** Define o que é considerado "relevante" para cálculo de métricas

**Impacto:**
- **Performance:** Nenhum
- **Memória:** Nenhum
- **Qualidade:** Afeta diretamente Precision/Recall

**Valores recomendados:**

| Valor | Interpretação | Quando usar |
|-------|---------------|-------------|
| `3.0` | "Gostei minimamente" | Datasets com ratings baixos |
| `3.5` | "Gostei razoavelmente" | Mais leniente |
| `4.0` | **Padrão** - "Gostei bastante" | MovieLens (balanceado) |
| `4.5` | "Adorei" | Análise de favoritos |
| `5.0` | Apenas perfeitos | Muito restritivo |

**Recomendação:** `4.0` para MovieLens (alinhado com literatura)

---

### `RANDOM_STATE = 42`

**O que faz:** Seed para reprodutibilidade (splits, sampling, T-SNE)

**Impacto:**
- **Performance:** Nenhum
- **Memória:** Nenhum
- **Qualidade:** Resultados idênticos entre execuções

**Recomendação:** Manter `42` (Douglas Adams reference 😉)

---

## Modelos

### `EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'`

**O que faz:** Modelo de embeddings para representação semântica dos filmes

**Características do modelo padrão:**
- **Dimensões:** 384
- **Tamanho:** ~80 MB
- **Velocidade:** ~500-1000 sentenças/seg (CPU)
- **Qualidade:** Balanceada

**Alternativas:**

| Modelo | Dimensões | Tamanho | Velocidade | Qualidade | RAM Adicional |
|--------|-----------|---------|------------|-----------|---------------|
| `all-MiniLM-L6-v2` | 384 | 80 MB | 🟢 Rápido | 🟡 Boa | +25 MB |
| `all-mpnet-base-v2` | 768 | 420 MB | 🟡 Médio | 🟢 Excelente | +50 MB |
| `all-MiniLM-L12-v2` | 384 | 120 MB | 🟡 Médio | 🟢 Muito boa | +25 MB |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 420 MB | 🟡 Médio | 🟢 Boa + Multilíngue | +25 MB |

**Para testar modelo mais robusto:**

```python
# Melhor qualidade (768 dims, +50MB RAM, ~2x mais lento)
EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'
```

**Impactos:**
- ✅ Semântica mais rica (captura nuances)
- ✅ Melhor performance em conteúdo nichado
- ❌ Dobra tempo de geração de embeddings (~30-40 min)
- ❌ +50 MB de RAM
- ❌ Necessário deletar cache: `rm cache/movie_embeddings.pkl`

**Recomendação:**
- Manter `all-MiniLM-L6-v2` para prototipagem
- Usar `all-mpnet-base-v2` para produção/publicação

---

## Otimizações de Memória

### `EMBEDDING_BATCH_SIZE = 32`

**O que faz:** Número de filmes processados simultaneamente ao gerar embeddings

**Impacto:**

| Valor | Tempo (15k filmes) | Pico de RAM | Quando usar |
|-------|-------------------|-------------|-------------|
| `8` | ~25 min | ~300 MB | RAM crítica (< 8GB) |
| `16` | ~20 min | ~400 MB | Seguro |
| `32` | **~15 min** | ~500 MB | **Padrão** |
| `64` | ~12 min | ~800 MB | RAM confortável (16GB+) |
| `128` | ~10 min | ~1.5 GB | RAM abundante (32GB+) |

**Para máquinas robustas:**

```python
EMBEDDING_BATCH_SIZE = 64  # Sua máquina aguenta fácil!
```

**Recomendação:** Com 16GB, pode usar `64` ou até `128` tranquilamente.

---

### `SIMILARITY_CHUNK_SIZE = 1000`

**O que faz:** Número de filmes processados por vez ao calcular similaridade

**Impacto:**

| Valor | Tempo | Pico de RAM | Trade-off |
|-------|-------|-------------|-----------|
| `500` | ~25 seg | ~1.5 GB | Muito seguro |
| `1000` | **~15 seg** | ~2.5 GB | **Padrão** |
| `2000` | ~12 seg | ~4.0 GB | Rápido, RAM OK |
| `5000` | ~10 seg | ~8.0 GB | Máximo |
| `15276` | ~8 seg | ~15 GB | Matriz completa (perigoso!) |

**Para máquinas robustas:**

```python
SIMILARITY_CHUNK_SIZE = 2000  # 20% mais rápido, RAM ok
```

**Recomendação:** Sua máquina pode usar `2000` sem problemas.

---

### `TOP_K_SIMILAR = 100`

**O que faz:** Quantos filmes similares guardar por filme (K-NN aproximado)

**Impacto:**

| Valor | Memória | Qualidade | Trade-off |
|-------|---------|-----------|-----------|
| `50` | ~6 MB | 🟡 Boa | Economia extrema |
| `100` | **~12 MB** | 🟢 Muito boa | **Padrão** |
| `200` | ~24 MB | 🟢 Excelente | Overkill |
| `500` | ~60 MB | 🟢 Máxima | Desperdício |
| `15276` | ~1 GB | 🟢 Completa | Inviável |

**Para testes mais robustos:**

```python
TOP_K_SIMILAR = 200  # Dobra memória, melhora qualidade ~2%
```

**Análise:**
- Após top-100, ganhos marginais (~1-2% em métricas)
- Colaborativo usa apenas vizinhos mais próximos
- 100 é sweet spot (qualidade × memória)

**Recomendação:** Manter `100` (retorno decrescente após isso)

---

## Parâmetros do Sistema Híbrido

### `ALPHA_MIN_RATINGS = 5`

**O que faz:** Mínimo de ratings para começar a confiar no colaborativo

**Impacto:**
- **Performance:** Nenhum
- **Memória:** Nenhum
- **Qualidade:** Afeta cold-start

**Valores recomendados:**

| Valor | Comportamento | Quando usar |
|-------|---------------|-------------|
| `3` | Confia cedo no colaborativo | Usuários ativos |
| `5` | **Padrão** - Balanceado | Geral |
| `10` | Conservador | Prioriza conteúdo |
| `20` | Muito conservador | Cold-start severo |

**Recomendação:** Manter `5` (literatura sugere 3-10)

---

### `ALPHA_MAX_RATINGS = 50`

**O que faz:** Número de ratings para α atingir 0.9 (máximo peso colaborativo)

**Impacto:**
- **Performance:** Nenhum
- **Memória:** Nenhum
- **Qualidade:** Define curva de confiança

**Fórmula do α adaptativo:**
```python
alpha = min(0.9, 0.3 + 0.6 * (num_ratings / ALPHA_MAX_RATINGS))
```

**Exemplos:**

| `ALPHA_MAX_RATINGS` | 5 ratings → α | 25 ratings → α | 50 ratings → α |
|---------------------|---------------|----------------|----------------|
| `25` | 0.42 | 0.90 | 0.90 |
| `50` | **0.36** | **0.60** | **0.90** |
| `100` | 0.33 | 0.45 | 0.60 |

**Para testes mais robustos:**

```python
ALPHA_MAX_RATINGS = 100  # Confia mais lentamente no colaborativo
```

**Recomendação:** `50` é bom para MovieLens (média de 22 ratings/usuário)

---

## Cenários de Uso

### Cenário 1: Máquina Modesta (8GB RAM, CPU médio)

**Objetivo:** Garantir execução sem crashes

```python
# Célula 2
K = 10
MIN_RATING_THRESHOLD = 4.0
RANDOM_STATE = 42

EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

EMBEDDING_BATCH_SIZE = 16          # ← Reduzir
SIMILARITY_CHUNK_SIZE = 500        # ← Reduzir
TOP_K_SIMILAR = 50                 # ← Reduzir

ALPHA_MIN_RATINGS = 5
ALPHA_MAX_RATINGS = 50
```

**Resultado esperado:**
- ✅ RAM: 3-4 GB
- ⏱️ Tempo total: ~40-50 min
- 📊 Qualidade: ~95% do ótimo

---

### Cenário 2: Máquina Robusta (16GB+ RAM, CPU rápido) **← SUA MÁQUINA**

**Objetivo:** Melhor performance mantendo qualidade

```python
# Célula 2
K = 10
MIN_RATING_THRESHOLD = 4.0
RANDOM_STATE = 42

EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

EMBEDDING_BATCH_SIZE = 64          # ← Aumentar (2x velocidade)
SIMILARITY_CHUNK_SIZE = 2000       # ← Aumentar (1.5x velocidade)
TOP_K_SIMILAR = 100                # ← Manter

ALPHA_MIN_RATINGS = 5
ALPHA_MAX_RATINGS = 50
```

**Resultado esperado:**
- ✅ RAM: 5-7 GB
- ⏱️ Tempo total: ~15-20 min (sua máquina comprovou!)
- 📊 Qualidade: 100% (baseline)

---

### Cenário 3: Máxima Qualidade (Publicação/Pesquisa)

**Objetivo:** Melhores métricas possíveis

```python
# Célula 2
K = 10
MIN_RATING_THRESHOLD = 4.0
RANDOM_STATE = 42

EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'  # ← Modelo melhor

EMBEDDING_BATCH_SIZE = 64
SIMILARITY_CHUNK_SIZE = 2000
TOP_K_SIMILAR = 200                # ← Aumentar

ALPHA_MIN_RATINGS = 3              # ← Mais agressivo
ALPHA_MAX_RATINGS = 100            # ← Mais conservador
```

**Resultado esperado:**
- ✅ RAM: 7-9 GB
- ⏱️ Tempo total: ~30-40 min (primeira vez)
- 📊 Qualidade: +2-5% em métricas
- ⚠️ **Lembre-se:** Deletar `cache/movie_embeddings.pkl` antes!

---

### Cenário 4: Prototipagem Rápida (Desenvolvimento)

**Objetivo:** Iteração rápida durante desenvolvimento

```python
# Célula 2 - MODO DEBUG
K = 5                              # ← Reduzir
MIN_RATING_THRESHOLD = 4.0
RANDOM_STATE = 42

EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

EMBEDDING_BATCH_SIZE = 128         # ← Máximo
SIMILARITY_CHUNK_SIZE = 5000       # ← Máximo
TOP_K_SIMILAR = 50                 # ← Mínimo

ALPHA_MIN_RATINGS = 5
ALPHA_MAX_RATINGS = 50
```

**+ Reduzir amostra de avaliação:**
```python
# Células 7, 10, 12
for user_id in tqdm(test_users[:100], desc="Avaliando"):  # ← 100 ao invés de 1000
```

**Resultado esperado:**
- ✅ RAM: 6-8 GB
- ⏱️ Tempo total: ~8-10 min
- 📊 Qualidade: Aproximada (para debug)

---

## Troubleshooting

### Problema: "MemoryError" ou kernel crashing

**Solução:**

1. **Reduzir batches:**
   ```python
   EMBEDDING_BATCH_SIZE = 16
   SIMILARITY_CHUNK_SIZE = 500
   ```

2. **Reduzir Top-K:**
   ```python
   TOP_K_SIMILAR = 50
   ```

3. **Fechar outros programas** (Chrome, VS Code, etc.)

4. **Usar subset do dataset:**
   ```python
   # Célula 3 - após carregar dataset
   train_df = train_df.sample(frac=0.5, random_state=42)  # 50% dos dados
   ```

---

### Problema: Execução muito lenta

**Diagnóstico:**

| Célula | Tempo Esperado (16GB) | Se > 2x | Solução |
|--------|----------------------|---------|---------|
| 3 | 1-2 min | Rede lenta | Aguardar download |
| 6 | 10-30 seg | CPU lento | Aumentar `SIMILARITY_CHUNK_SIZE` |
| 8 | 10-20 min | CPU/RAM limitado | Aumentar `EMBEDDING_BATCH_SIZE` |
| 7,10,12 | 3-5 min | Muitos usuários | Reduzir amostra (`:1000` → `:500`) |

---

### Problema: Métricas muito baixas (< 0.20)

**Possíveis causas:**

1. **Threshold muito alto:**
   ```python
   MIN_RATING_THRESHOLD = 3.5  # Ao invés de 4.0
   ```

2. **K muito grande:**
   ```python
   K = 5  # Ao invés de 10
   ```

3. **Dataset muito esparso:**
   - Normal para MovieLens (99.86% sparsidade)
   - Precision@10 de 0.30-0.45 é **ótimo** para este dataset

---

### Problema: Cache desatualizado após mudar parâmetros

**Arquivos de cache afetados:**

| Parâmetro alterado | Cache a deletar |
|-------------------|-----------------|
| `EMBEDDING_MODEL` | `cache/movie_embeddings.pkl` |
| `SIMILARITY_CHUNK_SIZE` ou `TOP_K_SIMILAR` | `cache/item_similarity_topk.pkl` |
| `RANDOM_STATE` | `cache/tsne_2d.pkl` |
| Dataset sampling | `cache/dataset_processed.pkl` |

**Comando para limpar tudo:**
```bash
rm -rf cache/*.pkl
```

---

## Resumo de Recomendações

### Para sua máquina (16GB, CPU rápido):

```python
# === CONFIGURAÇÃO OTIMIZADA ===
K = 10
MIN_RATING_THRESHOLD = 4.0
RANDOM_STATE = 42

EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

EMBEDDING_BATCH_SIZE = 64          # ✨ Aproveitar sua CPU
SIMILARITY_CHUNK_SIZE = 2000       # ✨ Aproveitar sua RAM
TOP_K_SIMILAR = 100                # ✅ Sweet spot

ALPHA_MIN_RATINGS = 5
ALPHA_MAX_RATINGS = 50
```

### Para publicação/pesquisa (máxima qualidade):

```python
# === CONFIGURAÇÃO MÁXIMA ===
EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'  # ← Único change crítico
TOP_K_SIMILAR = 200                                          # ← Opcional
```

**Lembre-se:** Deletar `cache/movie_embeddings.pkl` antes de mudar modelo!

---

## Referências

- MovieLens: Harper & Konstan (2015)
- Sentence-Transformers: Reimers & Gurevych (2019)
- Hybrid Systems: Burke (2002)
- Matrix Factorization: Koren et al. (2009)
