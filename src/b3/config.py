"""
Configurações do projeto de classificação de dados com IA.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Carregar variáveis de ambiente do .env (se existir)
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Criar diretórios se não existirem
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset configuration
DATASET_NAME = "emotion"  # Opções: 'emotion', 'imdb'
DATASET_CONFIG = None  # Para 'emotion' é None, para 'imdb' também é None
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_SAMPLES = None  # None para usar todo o dataset, ou número para limitar

# Label mappings (serão preenchidos dinamicamente)
LABEL2ID = {}
ID2LABEL = {}

# Model configurations
# Abordagem 1: Embeddings + KNN
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
K_NEIGHBORS = 5
KNN_METRIC = "cosine"  # 'cosine', 'euclidean', 'manhattan'

# Abordagem 2: Fine-tuning
FINETUNED_MODEL_NAME = "distilbert-base-uncased"
FINETUNED_MODEL_PATH = MODELS_DIR / "finetuned_classifier"
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MAX_LENGTH = 128  # Tamanho máximo de tokens
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
LOGGING_STEPS = 100
SAVE_STEPS = 500
EVAL_STEPS = 500

# Abordagem 3: LLM
LLM_MODEL_NAME = "gemini-2.0-flash-exp"  # Modelo mais recente e gratuito
LLM_BATCH_SIZE = 5  # Processar em lotes para respeitar rate limits (10 req/min)
LLM_MAX_RETRIES = 3
LLM_RETRY_DELAY = 7  # segundos (aumentado para evitar rate limit)

# API Keys (carregadas de variáveis de ambiente)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Evaluation settings
METRICS_TO_CALCULATE = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "confusion_matrix",
]

# Para classificação multiclasse
AVERAGE_METHOD = "weighted"  # 'micro', 'macro', 'weighted'

# Visualization settings
FIGURE_SIZE = (10, 8)
DPI = 100
SAVE_FORMAT = "png"

# Logging
VERBOSE = True
LOG_FILE = RESULTS_DIR / "experiment.log"

# Device configuration (para PyTorch)
DEVICE = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"

# Seed para reprodutibilidade
SEED = 42

# Few-shot examples para LLM (serão preenchidos dinamicamente baseado no dataset)
FEW_SHOT_EXAMPLES = []
NUM_FEW_SHOT_EXAMPLES = 3  # Número de exemplos por classe
