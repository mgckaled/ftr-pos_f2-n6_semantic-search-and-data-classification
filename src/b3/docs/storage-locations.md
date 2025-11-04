# Locais de Armazenamento - Datasets e Modelos

## Resumo de Espaço Necessário

| Item | Tamanho Aproximado | Localização Padrão |
|------|-------------------|-------------------|
| Dataset (emotion) | ~10 MB | `~/.cache/huggingface/datasets/` |
| Dataset (imdb) | ~80 MB | `~/.cache/huggingface/datasets/` |
| Sentence-Transformers | ~80-120 MB | `~/.cache/torch/sentence_transformers/` |
| DistilBERT base | ~250-300 MB | `~/.cache/huggingface/hub/` |
| Modelo Fine-tuned | ~250-300 MB | `src/b3/models/finetuned_classifier/` |
| **TOTAL** | **~600 MB - 1 GB** | Múltiplos diretórios |

---

## Locais Padrão (Windows)

### 1. Datasets (Hugging Face)

**Localização padrão:**
```
C:\Users\<SEU_USUARIO>\.cache\huggingface\datasets\
```

**Estrutura:**
```
datasets/
├── emotion/
│   └── default/
│       └── 0.0.0/
│           ├── dataset_info.json
│           └── emotion-train.arrow
└── imdb/
    └── plain_text/
        └── 1.0.0/
            └── ...
```

**Exemplo real:**
```
C:\Users\Usuario\.cache\huggingface\datasets\emotion\default\0.0.0\
```

---

### 2. Modelos Hugging Face (Transformers)

**Localização padrão:**
```
C:\Users\<SEU_USUARIO>\.cache\huggingface\hub\
```

**Estrutura:**
```
hub/
├── models--distilbert-base-uncased/
│   ├── snapshots/
│   │   └── <hash>/
│   │       ├── config.json
│   │       ├── model.safetensors
│   │       ├── tokenizer_config.json
│   │       └── vocab.txt
│   └── refs/
└── ...
```

**Exemplo real:**
```
C:\Users\Usuario\.cache\huggingface\hub\models--distilbert-base-uncased\snapshots\abc123\
```

---

### 3. Modelos Sentence-Transformers

**Localização padrão:**
```
C:\Users\<SEU_USUARIO>\.cache\torch\sentence_transformers\
```

**Estrutura:**
```
sentence_transformers/
└── sentence-transformers_all-MiniLM-L6-v2/
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer_config.json
    └── vocab.txt
```

**Exemplo real:**
```
C:\Users\Usuario\.cache\torch\sentence_transformers\sentence-transformers_all-MiniLM-L6-v2\
```

---

### 4. Modelo Fine-tuned (Local no Projeto)

**Localização:** (definida no projeto)
```
src/b3/models/finetuned_classifier/
```

**Estrutura:**
```
finetuned_classifier/
├── config.json               (~1 KB)
├── model.safetensors         (~260 MB)
├── tokenizer_config.json     (~1 KB)
├── vocab.txt                 (~230 KB)
├── special_tokens_map.json   (~1 KB)
└── training_args.bin         (~4 KB)
```

**Caminho completo:**
```
C:\rocketseat\ftr\aulas\f2_n6_semantic-search-and-data-classification\src\b3\models\finetuned_classifier\
```

---

### 5. Resultados e Visualizações (Local no Projeto)

**Localização:**
```
src/b3/results/
```

**Estrutura:**
```
results/
├── confusion_matrix_embedding.png      (~50-100 KB)
├── confusion_matrix_finetuned.png      (~50-100 KB)
├── confusion_matrix_llm.png            (~50-100 KB)
├── metrics_comparison.png              (~50-100 KB)
├── inference_time_comparison.png       (~50-100 KB)
├── class_distribution_train.png        (~50-100 KB)
├── class_distribution_test.png         (~50-100 KB)
└── logs/
    └── events.out.tfevents.*           (~1-5 MB)
```

**Caminho completo:**
```
C:\rocketseat\ftr\aulas\f2_n6_semantic-search-and-data-classification\src\b3\results\
```

---

## Verificar Locais Atuais

### Via Python

```python
import os
from pathlib import Path

# Cache Hugging Face
hf_home = os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")
print(f"Hugging Face cache: {hf_home}")

# Cache Torch
torch_home = os.getenv("TORCH_HOME", Path.home() / ".cache" / "torch")
print(f"Torch cache: {torch_home}")

# Projeto
import config
print(f"Modelo fine-tuned: {config.FINETUNED_MODEL_PATH}")
print(f"Resultados: {config.RESULTS_DIR}")
```

### Via Terminal

```bash
# Windows PowerShell
echo $env:USERPROFILE\.cache\huggingface\datasets
echo $env:USERPROFILE\.cache\huggingface\hub
echo $env:USERPROFILE\.cache\torch\sentence_transformers

# Git Bash / WSL
echo ~/.cache/huggingface/datasets
echo ~/.cache/huggingface/hub
echo ~/.cache/torch/sentence_transformers
```

---

## Mudar Locais de Armazenamento

### Opção 1: Variáveis de Ambiente (Recomendado)

Adicione ao arquivo `.env` do projeto:

```bash
# Cache Hugging Face (datasets + modelos)
HF_HOME=D:\ml_cache\huggingface

# Cache Torch (sentence-transformers)
TORCH_HOME=D:\ml_cache\torch

# Cache geral (fallback)
XDG_CACHE_HOME=D:\ml_cache
```

**Ativar:**

```bash
# Carregar variáveis
source .env  # Linux/Mac
# ou adicionar manualmente no Windows
```

---

### Opção 2: Configuração Global (Sistema)

#### Windows

```powershell
# Configurar permanentemente
[Environment]::SetEnvironmentVariable("HF_HOME", "D:\ml_cache\huggingface", "User")
[Environment]::SetEnvironmentVariable("TORCH_HOME", "D:\ml_cache\torch", "User")
```

#### Linux/Mac

```bash
# Adicionar ao ~/.bashrc ou ~/.zshrc
export HF_HOME="$HOME/ml_cache/huggingface"
export TORCH_HOME="$HOME/ml_cache/torch"
```

---

### Opção 3: Modificar config.py

Para o modelo fine-tuned (local no projeto):

```python
# config.py

# Mudar local do modelo fine-tuned
FINETUNED_MODEL_PATH = Path("D:/ml_models/bloco_c/finetuned")

# Mudar local dos resultados
RESULTS_DIR = Path("D:/ml_results/bloco_c")
```

---

## Limpar Cache

### Limpar cache Hugging Face

```python
# Python
from huggingface_hub import scan_cache_dir

cache_info = scan_cache_dir()
print(f"Total: {cache_info.size_on_disk_str}")

# Deletar específico
# cache_info.delete_revisions("revision_hash").execute()
```

```bash
# Terminal (Windows)
rmdir /s /q %USERPROFILE%\.cache\huggingface

# Terminal (Linux/Mac)
rm -rf ~/.cache/huggingface
```

---

### Limpar cache Torch/Sentence-Transformers

```bash
# Windows
rmdir /s /q %USERPROFILE%\.cache\torch

# Linux/Mac
rm -rf ~/.cache/torch
```

---

### Limpar modelo fine-tuned do projeto

```bash
# Windows
rmdir /s /q src\b3\models\finetuned_classifier

# Linux/Mac
rm -rf src/b3/models/finetuned_classifier
```

---

### Limpar resultados

```bash
# Windows
rmdir /s /q src\b3\results

# Linux/Mac
rm -rf src/b3/results
```

---

## Estrutura Completa de Diretórios

```
Sistema:
├── C:\Users\<USER>\.cache\
│   ├── huggingface\
│   │   ├── datasets\           (~10-100 MB)
│   │   └── hub\                (~250-500 MB)
│   └── torch\
│       └── sentence_transformers\  (~80-120 MB)
│
Projeto:
└── C:\rocketseat\ftr\aulas\f2_n6_semantic-search-and-data-classification\
    └── src\b3\
        ├── models\
        │   └── finetuned_classifier\  (~260 MB) - criado após treinar
        ├── data\
        │   ├── raw\                   (vazio)
        │   └── processed\             (vazio)
        └── results\                   (~1-5 MB) - criado após executar
```

---

## Checklist: Verificar Espaço em Disco

Antes de executar o projeto:

```bash
# Windows PowerShell
Get-PSDrive C | Select-Object Free

# Linux/Mac
df -h ~
```

**Espaço mínimo recomendado:**
- **Disco C:** ~1-2 GB livres (para cache)
- **Disco do projeto:** ~500 MB livres (para modelo + resultados)

---

## Otimizações de Espaço

### 1. Usar apenas um dataset

```python
# config.py
DATASET_NAME = "emotion"  # Menor (~10 MB)
# ao invés de
DATASET_NAME = "imdb"     # Maior (~80 MB)
```

---

### 2. Não salvar modelo fine-tuned

Comente as linhas de salvamento no `main.py`:

```python
# main.py - linha ~140
# finetuned_clf.save_model()  # Comentar esta linha
```

**Economia:** ~260 MB

**Trade-off:** Precisará re-treinar toda vez

---

### 3. Usar modelo menor

```python
# config.py
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # ~80 MB
# ao invés de
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"  # ~420 MB
```

---

### 4. Compartilhar cache entre projetos

Se você tem múltiplos projetos que usam Hugging Face, o cache é compartilhado automaticamente! Não duplica espaço.

---

## FAQ

### Por que não salvar tudo no projeto?

- **Cache compartilhado:** Outros projetos podem reutilizar modelos
- **Padrão das bibliotecas:** Facilita atualizações e versionamento
- **Não versionar binários:** Git ignora cache, mantém repo leve

---

### Como mover cache para outro disco?

1. Copiar diretório `.cache` para novo local
2. Configurar variáveis de ambiente (ver Opção 1 acima)
3. Deletar cache antigo

```bash
# Exemplo: Mover para D:
cp -r ~/.cache/huggingface D:/ml_cache/huggingface
export HF_HOME="D:/ml_cache/huggingface"
rm -rf ~/.cache/huggingface
```

---

### O que acontece se deletar o cache?

- **Datasets:** Serão baixados novamente (~10-80 MB)
- **Modelos:** Serão baixados novamente (~80-300 MB cada)
- **Modelo fine-tuned:** Precisará treinar novamente (~30-60 min)
- **Resultados:** Precisará executar experimento novamente

**Não afeta o código!** Apenas precisará re-baixar/re-treinar.

---

### Como verificar o que está usando espaço?

```python
import os
from pathlib import Path

def get_dir_size(path):
    total = 0
    for entry in Path(path).rglob('*'):
        if entry.is_file():
            total += entry.stat().st_size
    return total / (1024**3)  # GB

# Verificar caches
hf_cache = Path.home() / ".cache" / "huggingface"
torch_cache = Path.home() / ".cache" / "torch"

print(f"HF cache: {get_dir_size(hf_cache):.2f} GB")
print(f"Torch cache: {get_dir_size(torch_cache):.2f} GB")
```

---

## Resumo Final

| O que | Onde | Tamanho | Pode deletar? |
|-------|------|---------|---------------|
| Datasets | `~/.cache/huggingface/datasets/` | ~10-100 MB | ✅ Sim (re-baixa) |
| Modelos HF | `~/.cache/huggingface/hub/` | ~250-500 MB | ✅ Sim (re-baixa) |
| Sentence-T | `~/.cache/torch/sentence_transformers/` | ~80-120 MB | ✅ Sim (re-baixa) |
| Fine-tuned | `src/b3/models/finetuned_classifier/` | ~260 MB | ✅ Sim (re-treina) |
| Resultados | `src/b3/results/` | ~1-5 MB | ✅ Sim (re-executa) |
| **Código fonte** | `src/b3/*.py` | ~50 KB | ❌ **NÃO!** |

**Total necessário:** ~1-2 GB
