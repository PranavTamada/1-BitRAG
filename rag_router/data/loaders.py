"""
Dataset Loaders
================
Unified loading interface for all three experimental datasets.

Each loader returns a list of dicts with keys:
    {"query": str, "answer": str, "dataset": str}

Datasets:
    1. Healthcare QA   — local JSONL file (primary development dataset)
    2. Natural Questions — HuggingFace datasets (open-domain, short answers)
    3. PubMedQA         — HuggingFace datasets (biomedical, expert-labeled)

Research purpose:
    Cross-domain evaluation on three datasets tests whether retrieval
    geometry features generalise beyond the training domain (healthcare).
    This is a key experiment (Section 5.3 of the paper).
"""

import json
from pathlib import Path
from config import DATA_DIR


def load_healthcare_qa(max_samples: int | None = None) -> list[dict]:
    """Load the local Healthcare QA dataset.

    Format: JSONL with {\"prompt\": str, \"completion\": str} per line.

    Args:
        max_samples: optional cap on number of samples to load.

    Returns:
        list of {"query", "answer", "dataset"} dicts.
    """
    path = DATA_DIR / "healthcare_qa_dataset.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Healthcare QA dataset not found at {path}. "
            f"Copy it from rag_project/data/healthcare_qa_dataset.jsonl"
        )

    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            if "prompt" not in obj or "completion" not in obj:
                raise ValueError(
                    f"Missing prompt/completion keys at line {line_no} in {path}"
                )
            dataset.append({
                "query": obj["prompt"],
                "answer": obj["completion"],
                "dataset": "healthcare_qa",
            })
            if max_samples and len(dataset) >= max_samples:
                break
    return dataset


def load_natural_questions(max_samples: int | None = None) -> list[dict]:
    """Load Natural Questions (validation split, short answers only).

    Uses HuggingFace ``datasets`` library.  Downloads on first call.

    Research note:
        We filter to questions that have at least one short answer to
        ensure meaningful BERTScore evaluation.  Questions with only
        long answers or no answers are excluded.

    Args:
        max_samples: optional cap on number of samples.

    Returns:
        list of {"query", "answer", "dataset"} dicts.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install 'datasets' package: pip install datasets")

    ds = load_dataset("natural_questions", "default", split="validation",
                      trust_remote_code=True)

    dataset = []
    for example in ds:
        question = example["question"]["text"]
        short_answers = example["annotations"]["short_answers"]

        # Collect all non-empty short answer texts
        answer_texts = []
        for sa_list in short_answers:
            for sa in sa_list:
                start = sa["start_token"]
                end = sa["end_token"]
                if start >= 0 and end > start:
                    # Extract answer text from document tokens
                    tokens = example["document"]["tokens"]["token"]
                    answer_text = " ".join(tokens[start:end])
                    if answer_text.strip():
                        answer_texts.append(answer_text.strip())

        if not answer_texts:
            continue

        # Use the first short answer as ground truth
        dataset.append({
            "query": question,
            "answer": answer_texts[0],
            "dataset": "natural_questions",
        })
        if max_samples and len(dataset) >= max_samples:
            break

    return dataset


def load_pubmedqa(max_samples: int | None = None) -> list[dict]:
    """Load PubMedQA (expert-labeled split).

    Uses HuggingFace ``datasets`` library.  Downloads on first call.

    Research note:
        Ground truth is the ``long_answer`` field (the detailed explanation),
        NOT the yes/no/maybe label.  We use the long answer for BERTScore
        evaluation because it provides richer semantic content for scoring.

    Args:
        max_samples: optional cap on number of samples.

    Returns:
        list of {"query", "answer", "dataset"} dicts.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install 'datasets' package: pip install datasets")

    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train",
                      trust_remote_code=True)

    dataset = []
    for example in ds:
        question = example["question"]
        long_answer = example["long_answer"]

        if not question or not long_answer:
            continue

        dataset.append({
            "query": question,
            "answer": long_answer,
            "dataset": "pubmedqa",
        })
        if max_samples and len(dataset) >= max_samples:
            break

    return dataset


def load_dataset_by_name(
    name: str, max_samples: int | None = None
) -> list[dict]:
    """Load a dataset by its config name.

    Args:
        name: one of "healthcare_qa", "natural_questions", "pubmedqa"
        max_samples: optional cap.

    Returns:
        list of {"query", "answer", "dataset"} dicts.
    """
    loaders = {
        "healthcare_qa": load_healthcare_qa,
        "natural_questions": load_natural_questions,
        "pubmedqa": load_pubmedqa,
    }
    if name not in loaders:
        raise ValueError(
            f"Unknown dataset '{name}'. Choose from: {list(loaders.keys())}"
        )
    return loaders[name](max_samples=max_samples)
