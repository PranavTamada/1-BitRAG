"""
RAG-Router Configuration
========================
All hyperparameters, paths, and model names centralised here.
Every other module imports from this file — never hardcode values elsewhere.
"""

import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# ── LLM models ────────────────────────────────────────────────────────────────
CHEAP_MODEL = "llama3.2:1b"                     # Ollama model name (locally available)
FULL_MODEL = "llama-3.3-70b-versatile"          # Groq model name
EMBEDDING_MODEL = "all-MiniLM-L6-v2"           # SentenceTransformer model

# ── API keys (loaded from environment / .env) ────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K = 10                                      # Documents to retrieve
RRF_K = 50                                      # RRF constant (Cormack et al.)
RETRIEVAL_HIT_THRESHOLD = 0.035                 # Minimum score to consider retrieval valid

# ── Routing thresholds (swept in pareto_curve.py) ─────────────────────────────
DEFAULT_ROUTING_THRESHOLD = 0.5                 # Pre-router confidence threshold
DEFAULT_CONFIDENCE_THRESHOLD = 0.7              # Post-gen confidence threshold
BERTSCORE_SUCCESS_THRESHOLD = 0.65              # BERTScore F1 above which cheap LLM "succeeds"

# ── Evaluation ────────────────────────────────────────────────────────────────
BERTSCORE_MODEL = "distilbert-base-uncased"     # Use roberta-large for camera-ready
BUDGET_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# ── Datasets ──────────────────────────────────────────────────────────────────
DATASETS = ["natural_questions", "pubmedqa"]  # healthcare_qa removed; use labeled_routing_data.jsonl only

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR      = PROJECT_ROOT / "data"
RESULTS_DIR   = PROJECT_ROOT / "results"
FIGURES_DIR   = RESULTS_DIR  / "figures"
TABLES_DIR    = RESULTS_DIR  / "tables"
MODELS_DIR    = PROJECT_ROOT / "models"
CACHE_DIR     = PROJECT_ROOT / ".cache"
LOG_DIR       = PROJECT_ROOT / "results"

# ── Ensure directories exist on import ────────────────────────────────────────
for _d in [DATA_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR, CACHE_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_STATE = 42
