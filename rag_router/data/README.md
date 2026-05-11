# Dataset Sources

## 1. Healthcare QA (Primary)
- **File:** `healthcare_qa_dataset.jsonl`
- **Source:** Copied from `rag_project/data/healthcare_qa_dataset.jsonl`
- **Format:** JSONL with `{"prompt": str, "completion": str}` per line
- **Size:** ~200 Q&A pairs

## 2. Natural Questions (Open Domain)
- **Source:** `datasets.load_dataset("natural_questions", "default", split="validation")`
- **Format:** HuggingFace dataset — downloaded automatically on first use
- **Size:** ~3,610 examples (filtered to short-answer subset)
- **Purpose:** Tests domain generalisation beyond healthcare

## 3. PubMedQA (Biomedical)
- **Source:** `datasets.load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")`
- **Format:** HuggingFace dataset — downloaded automatically on first use
- **Size:** 1,000 expert-labeled biomedical Q&A pairs
- **Purpose:** Tests routing generalisation to a different biomedical domain
- **Note:** We use the `long_answer` field (not yes/no label) for BERTScore evaluation
