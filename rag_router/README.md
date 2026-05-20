# RAG-Router: Retrieval-Difficulty-Aware Adaptive Routing for Cost-Efficient RAG

> **Research System** targeting EMNLP Findings / NAACL 2026

## Research Hypothesis

> *The geometric shape of retrieval score distributions -- computed purely from
> the retrieval step, before any LLM is invoked -- is a reliable predictor of
> query answering difficulty, and can be used to route queries between cheap and
> expensive LLMs with minimal accuracy loss and significant cost reduction.*

## Key Contributions

1. **Pre-generation routing** using retrieval geometry features (10 signals
   extracted from the score distribution of hybrid BM25+dense retrieval)
2. **Calibrated confidence scoring** via Platt-scaled logistic regression
   (replacing hand-weighted heuristics)
3. **Budget-constrained threshold optimization** producing Pareto-optimal
   accuracy-cost tradeoffs
4. **Cross-domain evaluation** on 3 datasets showing feature generalization

## Novelty vs Prior Work

| System | Routes *before* LLM? | Needs human labels? | Learned router? |
|--------|----------------------|--------------------|--------------------|
| FrugalGPT (Chen 2023) | No (post-gen) | No | Threshold only |
| RouteLLM (Ong 2024) | Yes | Yes (preferences) | Yes (Bradley-Terry) |
| 1-BitRAG v1 | No (post-gen) | No | No (hand-weighted) |
| **RAG-Router (Ours)** | **Yes** | **No** | **Yes (calibrated)** |

> **Baseline note:** Our *Post-Gen Cascade* baseline adapts FrugalGPT's
> cascade idea to the RAG setting (all baselines share the same retrieval
> pipeline; only the routing decision differs). Original FrugalGPT has no
> retrieval component and is therefore not directly comparable as a system.

## Architecture

```
Query --> [Hybrid Retriever] --> [Feature Extractor] --> [Pre-Router]
             |                    10 retrieval geometry     |
             |                    + 8 query features        |
             |                                              |
             |                    confidence >= t --------> [Cheap LLM]
             |                                                |
             |                                          [Post-Router]
             |                                           conf >= 0.7 --> Answer
             |                                           conf < 0.7 -+
             |                    confidence < t ------------------> [Full LLM] --> Answer
```

### Three Routing Modes (Ablation)
- **Mode A:** Pre-routing only (zero LLM calls for routing decision)
- **Mode B:** Post-routing only (calibrated post-gen confidence gate)
- **Mode C:** Hybrid (pre-routing + post-gen secondary gate) -- *full system*

## Retrieval Geometry Features (Core Contribution)

| # | Feature | Intuition |
|---|---------|-----------|
| 1 | `score_gap` | Rank-1 vs rank-2 gap. Sharp peak = easy query |
| 2 | `score_mean` | Mean of top-k scores. Low = poor retrieval |
| 3 | `score_variance` | Score spread. High variance = ambiguous |
| 4 | `score_entropy` | Shannon entropy. Low = peaked = confident |
| 5 | `top_score` | Best match quality |
| 6 | `score_ratio` | Rank-1 dominance over rank-2 |
| 7 | `low_score_fraction` | Fraction of docs below mean |
| 8 | `retrieval_hit` | Binary: is top score above threshold? |
| 9 | `bm25_dense_agreement` | Spearman rho between BM25 and dense ranks |
| 10 | `context_density` | Mean pairwise cosine sim of retrieved docs |

## Datasets

1. **Healthcare QA** -- local JSONL, primary development dataset
2. **Natural Questions** -- HuggingFace, open-domain generalization test
3. **PubMedQA** -- HuggingFace, biomedical domain transfer test

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
export GROQ_API_KEY=your_key_here
ollama pull tinyllama && ollama serve

# 3. Verify setup
python -c "from retriever.retrieve import retrieve; print('retriever OK')"
python -c "from features.retrieval_features import extract_retrieval_features; print('features OK')"
python -c "from router.pre_router import PreRouter; print('router OK')"

# 4. Collect labels (slow -- ~4-8 hours, run once)
python experiments/collect_labels.py --dataset healthcare_qa

# 5. Train routers (~1 min)
python experiments/train_router.py

# 6. Run ablation study
python experiments/run_ablation.py --dataset healthcare_qa

# 7. Generate Pareto frontier (Figure 2)
python experiments/pareto_curve.py --dataset healthcare_qa

# 8. Cross-domain evaluation (Table 1)
python experiments/cross_domain.py
```

## Expected Results

The main claim is validated if:
- RAG-Router achieves **>= 95%** of always-full BERTScore F1
- While using **<= 50%** of full LLM calls
- And beats the Post-Gen Cascade baseline at matched cost on **>= 2 of 3** datasets

```
Dataset         | System              | BERTScore F1 | Full LLM % | Latency
----------------|---------------------|--------------|------------|--------
Healthcare QA   | Always-Cheap        | 0.XX         | 0%         | XX ms
Healthcare QA   | Always-Full         | 0.XX         | 100%       | XX ms
Healthcare QA   | Post-Gen Cascade    | 0.XX         | XX%        | XX ms
Healthcare QA   | RAG-Router          | 0.XX         | XX%        | XX ms
```

## Project Structure

```
rag_router/
  config.py              # All hyperparameters
  retriever/             # Hybrid BM25 + FAISS retrieval
  features/              # Retrieval geometry + query feature extractors
  router/                # Pre-router, post-router, budget optimizer
  llm/                   # Cheap (Ollama) + Full (Groq) LLM wrappers
  evaluation/            # Metrics (BERTScore, ROUGE-L), baselines, eval loop
  experiments/           # 5-step experiment pipeline
  data/                  # Dataset loaders + raw data
  utils/                 # Caching, logging, prompts
  results/figures/       # Generated plots
  results/tables/        # Generated CSV + LaTeX tables
  tests/                 # Unit tests
```

## Reproducibility

- All random seeds fixed to `42`
- All LLM calls cached to `.cache/` (SHA-256 keyed)
- All results saved to `results/` (CSV + LaTeX + JSON logs)
- Feature ordering is canonical across all modules

## License

Research code -- see LICENSE file.
