"""
config.py - single place for every constant shared across steps.
Change values here, never in notebooks.
"""
import os

# ---------------------------------------------------------------- paths (Drive)
ROOT      = '/content/drive/MyDrive/graphrag_defense'
CODE_DIR  = f'{ROOT}/grd'
INDEX_DIR = f'{ROOT}/index'
LOG_DIR   = f'{ROOT}/logs'
DATA_DIR  = f'{ROOT}/data'

def index_path(dataset: str, n_docs: int) -> str:
    return f'{INDEX_DIR}/lightrag_{dataset}_{n_docs}'

def ensure_dirs():
    for d in (INDEX_DIR, LOG_DIR, DATA_DIR):
        os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------- models (Ollama, T4)
OLLAMA_HOST = 'http://localhost:11434'
LLM_MODEL   = 'qwen2.5:7b'
EMB_MODEL   = 'nomic-embed-text'
EMB_DIM     = 768
LLM_NUM_CTX = 32768
LLM_MAX_ASYNC = 1        # one extraction at a time on a shared T4
OLLAMA_TIMEOUT = 600    # seconds; LightRAG default is too tight for a busy T4

# ---------------------------------------------------------------- the attack paper settings
CHUNK_TOKENS   = 1200
CHUNK_OVERLAP  = 100
LIGHTRAG_QUERY = dict(mode='hybrid', top_k=60,
                      max_token_for_text_unit=6000,
                      max_token_for_local_context=4000)
N_QUERIES_PER_SETTING = 250
N_QA_UTILITY = 100

# ---------------------------------------------------------------- data
DATASET   = 'hcm'
N_DOCS    = 200          # small-graph study for the 4-day window; report explicitly
SEED      = 42

# ---------------------------------------------------------------- generation
GEN_OPTIONS = dict(temperature=0.0, num_ctx=LLM_NUM_CTX, num_predict=2048)

# ---------------------------------------------------------------- DP defense
DP_CANDIDATE_POOL = 200   # top-N candidates to run DP selection over (N >> top_k)
DP_TOP_K          = 10    # items selected per stage (P7 GraphRAG top_k_entities/relationships)
DP_EPS_ENT        = 1.0   # eps_1^V
DP_EPS_REL        = 1.0   # eps_1^E
DP_EPS_DESC       = 1.0   # eps_2
DP_USER_BUDGET    = None  # per-user epsilon ceiling across queries; None = off
