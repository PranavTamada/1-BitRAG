# RAG-Router

**Retrieval-difficulty-aware routing for cost-sensitive Retrieval-Augmented Generation (RAG).**

This repository studies whether a RAG system can decide **before generation** whether a query should be answered by a small local model or escalated to a larger paid model, using only retrieval-time signals and lightweight query features.

The project is built as a research codebase rather than a demo-only prototype: it includes structured feature extraction, trained routing models, baseline evaluation, ablations, threshold calibration, caching, and artifact generation for tables and figures.

## Table of Contents

- [Why this project exists](#why-this-project-exists)
- [Research question](#research-question)
- [System overview](#system-overview)
- [Architecture](#architecture)
- [Core ideas](#core-ideas)
- [Repository status](#repository-status)
- [Tech stack](#tech-stack)
- [Project layout](#project-layout)
- [Setup](#setup)
- [Data](#data)
- [Experiment workflow](#experiment-workflow)
- [Current checked-in results](#current-checked-in-results)
- [Reproducibility](#reproducibility)
- [Testing](#testing)
- [Known limitations](#known-limitations)

## Why this project exists

In most cascaded LLM systems, routing happens **after** a first model has already generated an answer. That still incurs latency and often wastes a cheap-model call on hard queries.

This project asks a narrower engineering question:

> Can retrieval itself tell us enough about query difficulty to make a good routing decision before any LLM call is made?

If the answer is yes, a RAG system can avoid unnecessary expensive generations and reduce end-to-end latency on hard queries by escalating immediately.

## Research question

The central hypothesis is:

> The geometry of retrieval scores, combined with lightweight query-complexity signals, contains enough predictive signal to estimate whether a cheap model will succeed on a given query.

Concretely, the system predicts:

- `P(cheap LLM succeeds | retrieval features, query features)`

and uses that probability to route each query to either:

- a **cheap local model** via Ollama, or
- a **full-capability remote model** via Groq.

## System overview

The repository implements three routing regimes:

1. `always_cheap`
   Always answer with the local small model.
2. `pre_only`
   Route before generation using a trained pre-router.
3. `rag_router`
   Use the pre-router first, then optionally apply a learned post-generation confidence gate.

It also includes comparison baselines:

- `always_full`
- `random_routing`
- `frugal_gpt` / post-generation cascade baseline
- `oracle_routing`

## Architecture

```text
Query
  |
  v
Hybrid Retrieval
  |- Dense retrieval (SentenceTransformers + FAISS)
  |- Sparse retrieval (BM25)
  `- Reciprocal Rank Fusion (RRF)
  |
  v
Feature Construction
  |- 10 retrieval-geometry features
  `- 8 query-complexity features
  |
  v
Pre-Router
  |- Logistic regression (primary, interpretable)
  `- Gradient boosted trees (comparison)
  |
  +--> route to cheap model --------------------+
  |                                             |
  |                                      Cheap LLM (Ollama)
  |                                             |
  |                                      Post-Router
  |                                             |
  |                        accept cheap answer <-+-> escalate to full model
  |
  `--> route directly to full model
                                                |
                                                v
                                      Full LLM (Groq)
                                                |
                                                v
                                              Answer
```

## Core ideas

### 1. Pre-generation routing

The main research contribution is the **pre-router** in [router/pre_router.py](/C:/Users/prana/OneDrive/Desktop/Projects/1-BitRAG/rag_router/router/pre_router.py), which predicts cheap-model success without making any LLM call.

This is materially different from post-generation cascades because it can:

- skip the cheap model entirely on hard queries
- avoid wasted generations
- make routing latency effectively negligible compared with generation latency

### 2. Retrieval geometry as signal

The retrieval feature extractor in [features/retrieval_features.py](/C:/Users/prana/OneDrive/Desktop/Projects/1-BitRAG/rag_router/features/retrieval_features.py) models the **shape** of the ranked score distribution, not just top-1 relevance.

The canonical retrieval features are:

- `score_gap`
- `score_mean`
- `score_variance`
- `score_entropy`
- `top_score`
- `score_ratio`
- `low_score_fraction`
- `retrieval_hit`
- `bm25_dense_agreement`
- `context_density`

These are paired with query-level signals from [features/query_features.py](/C:/Users/prana/OneDrive/Desktop/Projects/1-BitRAG/rag_router/features/query_features.py), including:

- query length
- query entropy
- negation
- conditional structure
- comparison structure
- named entity count

### 3. Calibrated decision thresholds

Training does not stop at fitting a classifier. The pre-router threshold is calibrated in [experiments/train_router.py](/C:/Users/prana/OneDrive/Desktop/Projects/1-BitRAG/rag_router/experiments/train_router.py) using a held-out split and an F1 objective on the minority class, then serialized to:

- [models/pre_router_threshold.json](/C:/Users/prana/OneDrive/Desktop/Projects/1-BitRAG/rag_router/models/pre_router_threshold.json)

This matters because routing quality depends as much on thresholding as on classifier AUC.

### 4. Cost-aware evaluation

The evaluation layer in [evaluation/metrics.py](/C:/Users/prana/OneDrive/Desktop/Projects/1-BitRAG/rag_router/evaluation/metrics.py) reports more than accuracy:

- BERTScore F1
- ROUGE-L
- full-model usage fraction
- estimated dollar cost
- bootstrap confidence intervals
- paired significance tests
- latency

## Repository status

This repository contains both:

- the **intended research design**, and
- the **current checked-in experiment state**.

Important current-state notes:

- The code still contains loaders and legacy references for `healthcare_qa` and `natural_questions`.
- The active checked-in experiment flow is now mostly centered on **PubMedQA**.
- [experiments/run_all.py](/C:/Users/prana/OneDrive/Desktop/Projects/1-BitRAG/rag_router/experiments/run_all.py) currently assumes labeled `pubmedqa` data and skips cross-domain evaluation unless more labeled data is added.
- The present checked-in results do **not** show the full routed system outperforming simple baselines on PubMedQA.

That is not a README flaw; it is the actual current empirical state of the repository.

## Tech stack

### Languages and runtime

- Python 3.12-style codebase

### Retrieval and representation

- `sentence-transformers`
- `faiss-cpu`
- `rank-bm25`
- `numpy`
- `scipy`

### Modeling and calibration

- `scikit-learn`
- `joblib`

### Evaluation and analysis

- `bert-score`
- `rouge-score`
- `pandas`
- `matplotlib`
- `seaborn`
- `tqdm`

### Data access

- `datasets` from Hugging Face

### Inference backends

- `ollama` for the cheap local model
- `groq` for the full remote model

### Configuration and environment

- `python-dotenv`

The pinned dependency surface is listed in [requirements.txt](/C:/Users/prana/OneDrive/Desktop/Projects/1-BitRAG/rag_router/requirements.txt).

## Project layout

```text
rag_router/
+- config.py                 # Central configuration and paths
+- data/
¦  +- loaders.py             # Dataset loading layer
¦  +- labeled_routing_data.jsonl
+- retriever/
¦  +- dense.py               # Dense retrieval
¦  +- sparse.py              # BM25 retrieval
¦  +- fusion.py              # RRF fusion
¦  +- retrieve.py            # Unified retrieval entry point
+- features/
¦  +- retrieval_features.py  # Retrieval-geometry features
¦  +- query_features.py      # Query-complexity features
+- router/
¦  +- pre_router.py          # Main routing classifier
¦  +- post_router.py         # Post-generation confidence model
¦  +- budget_optimizer.py    # Threshold / budget utilities
+- llm/
¦  +- cheap_llm.py           # Ollama wrapper
¦  +- full_llm.py            # Groq wrapper + key rotation
+- evaluation/
¦  +- baselines.py           # Baseline registry
¦  +- metrics.py             # Accuracy, cost, significance
¦  +- evaluate.py            # Main evaluation loop
+- experiments/
¦  +- collect_labels.py      # Build labeled routing dataset
¦  +- train_router.py        # Train and calibrate routers
¦  +- run_ablation.py        # Ablation comparisons
¦  +- feature_ablation.py    # Feature-group studies
¦  +- pareto_curve.py        # Cost/quality frontier
¦  +- cross_domain.py        # Cross-domain evaluation
¦  +- run_all.py             # End-to-end experiment driver
+- models/                   # Serialized routers and thresholds
+- results/
¦  +- figures/               # Generated figures
¦  +- tables/                # CSV + LaTeX artifacts
¦  +- training_log.jsonl     # Research log
+- tests/
¦  +- test_phase1.py
+- utils/
   +- cache.py
   +- logger.py
   +- normalize.py
   +- prompt.py
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file in the repository root or export variables in your shell:

```env
GROQ_API_KEY=your_primary_groq_key
GROQ_API_KEY_2=optional_secondary_key
GROQ_API_KEY_3=optional_secondary_key
GROQ_API_KEY_4=optional_secondary_key
GROQ_API_KEY_5=optional_secondary_key
```

The full-model wrapper in [llm/full_llm.py](/C:/Users/prana/OneDrive/Desktop/Projects/1-BitRAG/rag_router/llm/full_llm.py) rotates across multiple Groq keys when rate limits are hit.

### 3. Start the local cheap model

The configured cheap model in [config.py](/C:/Users/prana/OneDrive/Desktop/Projects/1-BitRAG/rag_router/config.py) is:

- `llama3.2:1b`

Example:

```bash
ollama pull llama3.2:1b
ollama serve
```

### 4. Sanity-check imports

```bash
python -c "from retriever.retrieve import retrieve; print('retrieval OK')"
python -c "from router.pre_router import PreRouter; print('pre-router OK')"
python -c "from router.post_router import PostRouter; print('post-router OK')"
```

## Data

The repository supports three datasets in code:

- `healthcare_qa`
- `natural_questions`
- `pubmedqa`

See [data/loaders.py](/C:/Users/prana/OneDrive/Desktop/Projects/1-BitRAG/rag_router/data/loaders.py) and [data/README.md](/C:/Users/prana/OneDrive/Desktop/Projects/1-BitRAG/rag_router/data/README.md).

Current practical status:

- The checked-in labeled routing set is [data/labeled_routing_data.jsonl](/C:/Users/prana/OneDrive/Desktop/Projects/1-BitRAG/rag_router/data/labeled_routing_data.jsonl).
- The current training/evaluation flow primarily uses **PubMedQA** labels.
- Healthcare and cross-domain paths remain partially available, but they are not the current default path.

## Experiment workflow

### 1. Collect labels

This step creates supervision for routing by comparing cheap-model and full-model answers against ground truth.

```bash
python experiments/collect_labels.py --dataset pubmedqa
```

The output is appended to:

- [data/labeled_routing_data.jsonl](/C:/Users/prana/OneDrive/Desktop/Projects/1-BitRAG/rag_router/data/labeled_routing_data.jsonl)

### 2. Train routers

```bash
python experiments/train_router.py --dataset pubmedqa
```

This produces:

- [models/pre_router_logistic.pkl](/C:/Users/prana/OneDrive/Desktop/Projects/1-BitRAG/rag_router/models/pre_router_logistic.pkl)
- [models/pre_router_gbt.pkl](/C:/Users/prana/OneDrive/Desktop/Projects/1-BitRAG/rag_router/models/pre_router_gbt.pkl)
- [models/post_router.pkl](/C:/Users/prana/OneDrive/Desktop/Projects/1-BitRAG/rag_router/models/post_router.pkl)
- [models/pre_router_threshold.json](/C:/Users/prana/OneDrive/Desktop/Projects/1-BitRAG/rag_router/models/pre_router_threshold.json)

### 3. Run evaluation

```bash
python evaluation/evaluate.py --dataset pubmedqa
```

This writes:

- [results/tables/main_results_pubmedqa.csv](/C:/Users/prana/OneDrive/Desktop/Projects/1-BitRAG/rag_router/results/tables/main_results_pubmedqa.csv)
- [results/tables/main_results_pubmedqa.tex](/C:/Users/prana/OneDrive/Desktop/Projects/1-BitRAG/rag_router/results/tables/main_results_pubmedqa.tex)

### 4. Run ablations

```bash
python experiments/run_ablation.py --dataset pubmedqa
python experiments/feature_ablation.py
```

### 5. Generate Pareto curves

```bash
python experiments/pareto_curve.py --dataset pubmedqa
```

### 6. Run the current end-to-end driver

```bash
python experiments/run_all.py
```

## Current checked-in results

The checked-in main evaluation for PubMedQA is in [results/tables/main_results_pubmedqa.csv](/C:/Users/prana/OneDrive/Desktop/Projects/1-BitRAG/rag_router/results/tables/main_results_pubmedqa.csv).

Headline numbers from that artifact:

| System | BERTScore F1 | Full LLM Fraction | Mean Latency (ms) |
|---|---:|---:|---:|
| `always_cheap` | 0.8172 | 0.0000 | 5003.67 |
| `always_full` | 0.7971 | 1.0000 | 652.52 |
| `random_routing` | 0.8069 | 0.4909 | 2866.97 |
| `frugal_gpt` | 0.8171 | 0.0318 | 5024.69 |
| `pre_only` | 0.7982 | 0.9636 | 805.92 |
| `rag_router` | 0.7976 | 0.9864 | 825.53 |

Interpretation:

- The current checked-in PubMedQA run does **not** show a strong advantage for the full routed system.
- `always_cheap` is currently stronger than `pre_only` and `rag_router` in the stored table.
- The current pre-router is routing most queries to the full model anyway.

That makes the present repository valuable as a real research artifact rather than a polished success-story snapshot: it contains negative findings, failed hypotheses, and intermediate results worth debugging.

Two particularly important artifacts:

- [results/tables/feature_coefficients.csv](/C:/Users/prana/OneDrive/Desktop/Projects/1-BitRAG/rag_router/results/tables/feature_coefficients.csv)
- [results/tables/negative_findings.txt](/C:/Users/prana/OneDrive/Desktop/Projects/1-BitRAG/rag_router/results/tables/negative_findings.txt)

The negative-findings note explicitly records that an earlier PubMedQA setup yielded a degenerate post-router with no signal; later training logs show the post-router improving after the label distribution changed. This is the kind of detail that matters in a serious research README.

## Reproducibility

The codebase already includes several reproducibility-oriented decisions:

- central configuration in [config.py](/C:/Users/prana/OneDrive/Desktop/Projects/1-BitRAG/rag_router/config.py)
- fixed random seed via `RANDOM_STATE = 42`
- serialized models and thresholds under `models/`
- JSONL experiment logging in [results/training_log.jsonl](/C:/Users/prana/OneDrive/Desktop/Projects/1-BitRAG/rag_router/results/training_log.jsonl)
- LLM-call caching in [utils/cache.py](/C:/Users/prana/OneDrive/Desktop/Projects/1-BitRAG/rag_router/utils/cache.py)
- LaTeX and CSV output generation for paper-ready tables

If you want reproducible comparisons, keep these fixed across runs:

- retrieval corpus
- label-generation criterion (`LABEL_MODE`, `GAP_RATIO`, `BERTSCORE_SUCCESS_THRESHOLD`)
- cheap/full model identities
- threshold calibration procedure

## Testing

The repository currently includes a focused foundation-layer test file:

- [tests/test_phase1.py](/C:/Users/prana/OneDrive/Desktop/Projects/1-BitRAG/rag_router/tests/test_phase1.py)

Run it with:

```bash
python -m pytest tests/test_phase1.py -v
```

Coverage emphasis:

- config import chain
- retrieval feature extraction
- query feature extraction
- RRF fusion behavior
- combined feature dimensionality

## Known limitations

- The README now reflects the **actual current scope**, which is narrower than the original broader three-dataset story.
- Dataset support in code and active experiment flow are not perfectly aligned.
- The current pre-router performance is modest, and the checked-in routed systems do not yet demonstrate a convincing win on PubMedQA.
- The cheap model can outperform the full model on the stored PubMedQA evaluation, which complicates the original routing premise.
- Several legacy comments and historical logs still refer to earlier paths, thresholds, and dataset balances.
- The repository has limited automated test coverage beyond the feature/retrieval foundation layer.

## Practical next steps

If you are continuing this project as research engineering work, the highest-value next steps are:

1. Reconcile the documented research story with the current active experiment path.
2. Freeze one label-generation policy and regenerate a clean labeled routing dataset.
3. Re-run training, ablations, and evaluation from scratch into a fresh artifact directory.
4. Audit why `always_cheap` is currently stronger than routed systems on PubMedQA.
5. Decide whether the project should optimize for cost, latency, or quality retention, then recalibrate thresholds accordingly.
