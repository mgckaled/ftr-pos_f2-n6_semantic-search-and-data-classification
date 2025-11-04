<!--markdownlint-disable-->

# Classificação de Texto com IA - Bloco C

Mini-projeto educacional comparando **3 abordagens de classificação** com IA:

1. **Embeddings + KNN** (naive, rápido)
2. **Fine-tuning DistilBERT** (melhor precisão)
3. **LLM com Gemini** (flexível, zero-shot)

Dataset: **emotion** (6 classes: joy, sadness, love, anger, fear, surprise)

---

## 🚀 Quick Start

### 1. Instalação

```bash
# Instalar dependências
pipenv install

# Configurar API key (obtenha em: https://aistudio.google.com/app/apikey)
cp .env.example .env
# Editar .env e adicionar: GEMINI_API_KEY=sua_chave_aqui
```

### 2. Configuração (Opcional)

Edite `config.py` para ajustar:
- `MAX_SAMPLES`: Limitar dataset (None = completo, 1000 = teste rápido)
- `EPOCHS`: Número de épocas de treino (3 = padrão, 1 = rápido)
- `BATCH_SIZE`: Tamanho do batch (16 = padrão, 8 = se OOM)

---

## 📋 Ordem de Execução

### ⚠️ IMPORTANTE: Siga esta ordem!

```
1. main.py            → Treina os 3 modelos e gera comparações
         ↓
2. Notebooks          → Analisa resultados interativamente
```

### Passo 1: Executar o Experimento Completo

```bash
pipenv run python main.py
```

**O que acontece:**
- Carrega dataset emotion (16000 treino + 2000 teste)
- Treina Embedding + KNN (~5-10 min)
- Treina Fine-tuned DistilBERT (~30-60 min)
- Executa LLM Gemini (~10-20 min)
- Gera gráficos de comparação em `results/`

**Tempo total:** ~1-2 horas (CPU) ou ~15-30 min (GPU)

### Passo 2: Analisar Resultados (Notebooks)

Execute **NESTA ORDEM**:

#### 📊 2.1. Análise Geral dos Resultados
```bash
jupyter notebook notebooks/results_analysis.ipynb
```
- Visualiza todas as métricas e gráficos lado a lado
- Compara os 3 modelos
- Decisão: qual modelo usar?

#### 🔍 2.2. Análise de Erros
```bash
jupyter notebook notebooks/error_analysis.ipynb
```
- Onde cada modelo erra?
- Padrões de confusão entre classes
- Casos onde modelo simples bate complexo
- ⚠️ **Requer modificação no `main.py`** (instruções no notebook)

#### 🎮 2.3. Playground Interativo
```bash
jupyter notebook notebooks/interactive_playground.ipynb
```
- Teste com seus próprios textos
- Compare predições dos 3 modelos em tempo real
- Experimente textos ambíguos

---

## 📁 Estrutura do Projeto

```
src/b3/
├── config.py                      # ⚙️ Configurações globais
├── main.py                        # 🎯 EXECUTE PRIMEIRO
│
├── models/                        # 🤖 3 Classificadores
│   ├── embedding_classifier.py    #    1. Embedding + KNN
│   ├── finetuned_classifier.py    #    2. Fine-tuned DistilBERT
│   └── llm_classifier.py          #    3. LLM (Gemini)
│
├── utils/                         # 🛠️ Utilitários
│   ├── data_loader.py             #    Carrega dataset
│   ├── metrics.py                 #    Calcula métricas
│   └── visualization.py           #    Gera gráficos
│
├── notebooks/                     # 📓 EXECUTE DEPOIS (ordem abaixo)
│   ├── results_analysis.ipynb     #    1. Análise geral
│   ├── error_analysis.ipynb       #    2. Análise de erros
│   ├── interactive_playground.ipynb#   3. Testes interativos
│   ├── exploratory_analysis.ipynb #    (extra) Análise do dataset
│   └── test_visualizations.ipynb  #    (extra) Testa gráficos
│
├── docs/                          # 📚 Documentação
│   ├── plan.md                    #    Planejamento completo
│   ├── testing-guide.md           #    Guia de testes detalhado
│   ├── interpretation-guide.md    #    Como interpretar resultados
│   └── storage-locations.md       #    Onde modelos são salvos
│
└── results/                       # 📊 Gerado por main.py
    ├── confusion_matrix_*.png
    ├── metrics_comparison.png
    ├── inference_time_comparison.png
    └── class_distribution_*.png
```

---

## 🎓 O Que Você Vai Aprender

**Bloco C - Conteúdo Coberto:**

- ✅ **Aula 1**: Modelos foundational vs clássicos
- ✅ **Aula 2**: Métricas de avaliação (accuracy, precision, recall, F1, ROC-AUC, confusion matrix)
- ✅ **Aula 3**: 3 abordagens de classificação com IA
- ✅ **Aula 4**: Trade-offs práticos (velocidade vs precisão vs custo)

**Resultados Esperados:**

| Modelo | Accuracy | Velocidade | Precisa Treinar? |
|--------|----------|------------|------------------|
| Embedding + KNN | ~75% | Rápido (0.5s) | Não |
| Fine-tuned DistilBERT | ~88% | Médio (2.3s) | Sim (30-60 min) |
| LLM (Gemini) | ~82% | Lento (5.7s) | Não (API) |

---

## 🧪 Testes Rápidos (Opcional)

Antes de rodar `main.py`, você pode testar módulos individualmente:

```bash
# Teste 1: Carregamento de dados (~30s)
pipenv run python utils/data_loader.py

# Teste 2: Embedding classifier (~1 min)
pipenv run python models/embedding_classifier.py

# Teste 3: LLM classifier (~30s)
pipenv run python models/llm_classifier.py

# Teste 4: Fine-tuned (~2 min, 1 época com 500 amostras)
pipenv run python models/finetuned_classifier.py
```

---

## ⚠️ Troubleshooting

### Erro: `GEMINI_API_KEY não encontrada`
```bash
# 1. Obtenha chave em: https://aistudio.google.com/app/apikey
# 2. Crie arquivo .env:
echo GEMINI_API_KEY=sua_chave_aqui > .env
```

### Erro: `Out of Memory (OOM)`
```python
# Edite config.py:
MAX_SAMPLES = 1000  # Reduzir dataset
BATCH_SIZE = 8      # Reduzir batch size
```

### Fine-tuning muito lento
```python
# Edite config.py:
EPOCHS = 1          # Treinar apenas 1 época
MAX_SAMPLES = 2000  # Usar menos dados
```

### Modelos não salvando/carregando
- Verifique espaço em disco (~1-2 GB necessário)
- Cache em: `~/.cache/huggingface/` (Windows: `C:\Users\<USER>\.cache\`)
- Veja `docs/storage-locations.md` para detalhes

---

## 📚 Documentação Adicional

- **[docs/plan.md](docs/plan.md)**: Planejamento completo do projeto
- **[docs/testing-guide.md](docs/testing-guide.md)**: Guia de testes com cenários pré-configurados
- **[docs/interpretation-guide.md](docs/interpretation-guide.md)**: Como interpretar métricas e gráficos
- **[docs/storage-locations.md](docs/storage-locations.md)**: Onde os modelos são armazenados

---

## 🎯 Fluxo Recomendado

```
📖 Leia plan.md → Configure .env → Execute main.py (1-2h)
                                           ↓
          Analise results_analysis.ipynb → error_analysis.ipynb → interactive_playground.ipynb
                                           ↓
                              Ajuste config.py → Re-execute main.py
```

**Pronto para começar?** Execute `pipenv run python main.py` 🚀
