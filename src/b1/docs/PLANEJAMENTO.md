# Planejamento do Mini Projeto: Embeddings e Busca Semântica

## Objetivo

Criar um projeto didático em Python que demonstre os conceitos fundamentais do **Bloco A - Embeddings e Busca Semântica**, com exemplos práticos e visuais utilizando modelos pequenos e gratuitos do HuggingFace.

## Estrutura do Projeto

```text
src/b1/
├── exemplos/
│   ├── 01_embeddings_ingenuos.py
│   ├── 02_word_embeddings.py
│   ├── 03_similaridade_embeddings.py
│   ├── 04_busca_semantica.py
│   └── 05_visualizacao_3d.py
├── dados/
│   ├── textos_exemplo.json
│   └── temperaturas_cidades.csv
├── docs/
│   └── PLANEJAMENTO.md
└── visualizacoes/
    └── (gráficos gerados)
```

## Conteúdo do Bloco A (7 Aulas)

### Aula 1: Introdução à busca semântica e classificação

- Conceitos de busca semântica vs busca literal
- Diferença entre busca e classificação
- Exemplos práticos de aplicações

### Aula 2: Introdução a embeddings

- Como representar objetos do mundo real para computadores
- Limitações dos computadores em "entender" dados não estruturados
- Embeddings como vetores numéricos em espaços n-dimensionais

### Aula 3: Embeddings como vetores

- Visualização de embeddings em 2D e 3D
- Analogia com vetores da física (setas no plano cartesiano)
- Relações espaciais entre embeddings

### Aula 4: Características de embeddings

- **Representações densas**: poucas dimensões, todas com informação
- **Preservam semântica**: objetos similares têm embeddings próximos
- **Únicos**: cada objeto tem uma representação única
- Exemplo: embeddings de cidades por temperatura

### Aula 5: Embeddings ingênuos

- **One-hot encoding**: vetores esparsos sem semântica
- **Contagem de entidades**: frequência de palavras/pixels
- **Conversão em bits**: sem significado intrínseco
- Limitações dessas abordagens

### Aula 6: Gerando embeddings com redes neurais

- **Word2Vec**: embeddings baseados em contexto
- **Autoencoders**: compressão de dados preservando características
- Como redes neurais aprendem representações densas e semânticas

### Aula 7: Similaridade de embeddings

- **Distância Euclidiana (L2)**: distância direta entre vetores
- **Similaridade de Cossenos**: ângulo entre vetores (comum em NLP)
- **Produto Escalar**: combinação de magnitude e direção
- Quando usar cada métrica

## Exemplos Didáticos Planejados

### Exemplo 1: Embeddings Ingênuos (`01_embeddings_ingenuos.py`)

**Objetivo**: Demonstrar as limitações das abordagens ingênuas

**Conceitos**:

- One-hot encoding de palavras
- Visualização de vetores esparsos
- Cálculo de distâncias (todas iguais!)

**Dataset**: Lista de palavras simples (animais, objetos)

**Saída**: Matriz one-hot e comparação de distâncias

### Exemplo 2: Word Embeddings (`02_word_embeddings.py`)

**Objetivo**: Gerar embeddings semânticos com modelo pré-treinado

**Modelo HuggingFace**: `sentence-transformers/all-MiniLM-L6-v2`

- Modelo pequeno (~80MB)
- 384 dimensões
- Gratuito e rápido

**Conceitos**:

- Carregar modelo do HuggingFace
- Gerar embeddings de frases
- Embeddings densos vs esparsos

**Dataset**: Frases sobre animais, tecnologia, esportes

**Saída**: Vetores de embeddings e suas características

### Exemplo 3: Similaridade de Embeddings (`03_similaridade_embeddings.py`)

**Objetivo**: Calcular e comparar diferentes métricas de similaridade

**Conceitos**:

- Implementação das 3 métricas principais
- Comparação de resultados
- Visualização em 2D com PCA/t-SNE

**Dataset**: Conjunto de textos relacionados e não relacionados

**Saída**:

- Matriz de similaridade
- Gráfico 2D mostrando clusters
- Comparação das métricas

### Exemplo 4: Busca Semântica Simples (`04_busca_semantica.py`)

**Objetivo**: Implementar sistema básico de busca semântica

**Conceitos**:

- Banco de documentos
- Query (consulta do usuário)
- Ranking por similaridade de cossenos
- Top-K resultados mais relevantes

**Dataset**: Base de artigos/perguntas sobre diferentes tópicos

**Fluxo**:

1. Usuário entra com query
2. Sistema gera embedding da query
3. Calcula similaridade com todos documentos
4. Retorna top-5 documentos mais similares

**Saída**: Resultados ranqueados com scores

### Exemplo 5: Visualização 3D (`05_visualizacao_3d.py`)

**Objetivo**: Visualizar embeddings em 3D interativo

**Conceitos**:

- Redução de dimensionalidade (PCA/t-SNE)
- Visualização de clusters semânticos
- Exploração interativa

**Dataset**: Embeddings do exemplo 2

**Saída**: Gráfico 3D interativo (matplotlib)

### Exemplo Bônus: Temperaturas de Cidades (baseado na Aula 4)

**Objetivo**: Replicar o exemplo das cidades do slide

**Dataset**: `dados/temperaturas_cidades.csv`

- Cidades do mundo
- Temperatura média Janeiro/Julho

**Conceitos**:

- Embeddings manuais (2D)
- Clustering natural por clima
- Visualização geográfica

## Modelos HuggingFace Recomendados

### Para Texto (Sentence Embeddings)

1. **all-MiniLM-L6-v2** (Recomendado)
   - Tamanho: ~80MB
   - Dimensões: 384
   - Velocidade: Rápida
   - Uso: Propósito geral

2. **paraphrase-MiniLM-L3-v2** (Alternativa menor)
   - Tamanho: ~60MB
   - Dimensões: 384
   - Velocidade: Muito rápida
   - Uso: Paráfrase e similaridade

3. **all-mpnet-base-v2** (Melhor qualidade)
   - Tamanho: ~420MB
   - Dimensões: 768
   - Velocidade: Média
   - Uso: Quando precisar de maior qualidade

### Para Multilíngue (Português)

1. **paraphrase-multilingual-MiniLM-L12-v2**
   - Tamanho: ~470MB
   - Dimensões: 384
   - Suporta: 50+ idiomas incluindo português

## Tecnologias e Bibliotecas

- **Python**: 3.10+
- **UV**: Gerenciador de dependências
- **sentence-transformers**: Modelos de embeddings
- **numpy**: Operações com vetores
- **scikit-learn**: PCA, métricas de distância
- **matplotlib**: Visualizações
- **pandas**: Manipulação de dados
- **torch**: Backend para transformers

## Etapas de Implementação

### Fase 1: Setup e Dados

- [x] Configurar pyproject.toml
- [x] Criar estrutura de diretórios
- [ ] Preparar datasets de exemplo
- [ ] Criar arquivo de temperaturas de cidades

### Fase 2: Exemplos Básicos

- [ ] Implementar embeddings ingênuos
- [ ] Implementar word embeddings
- [ ] Implementar cálculos de similaridade

### Fase 3: Aplicação Prática

- [ ] Implementar busca semântica
- [ ] Criar visualizações 2D/3D
- [ ] Exemplo de temperaturas de cidades

### Fase 4: Documentação

- [ ] README completo com instruções
- [ ] Comentários detalhados no código
- [ ] Notebook Jupyter tutorial

### Fase 5: Extras

- [ ] Comparação de modelos
- [ ] Benchmark de velocidade
- [ ] Exemplos com textos em português

## Estrutura dos Arquivos Python

Cada arquivo terá:

```python
"""
Título do Exemplo
=================

Conceitos abordados:
- Conceito 1
- Conceito 2

Referência: Aula X do Bloco A
"""

# Imports
import numpy as np
# ...

# Funções auxiliares
def funcao_exemplo():
    """Docstring clara"""
    pass

# Seção principal com explicações
if __name__ == "__main__":
    print("=== Título da Seção ===")
    # Código didático com prints explicativos
```

## Métricas de Sucesso

- ✅ Código claro e bem comentado
- ✅ Exemplos executam em menos de 30 segundos
- ✅ Visualizações intuitivas e informativas
- ✅ Aborda todos os conceitos do Bloco A
- ✅ Modelos pequenos (<500MB total)
- ✅ Funciona offline após download inicial

## Próximos Passos

1. Instalar dependências com UV
2. Criar datasets de exemplo
3. Implementar exemplo 1 (embeddings ingênuos)
4. Testar e iterar

## Referências

- Apresentação Bloco A: `.github/docs/content/presentations/b1/`
- Resumo: `.github/docs/content/resumes/rb1.md`
- Sentence Transformers: <https://www.sbert.net/>
- HuggingFace Models: <https://huggingface.co/models>