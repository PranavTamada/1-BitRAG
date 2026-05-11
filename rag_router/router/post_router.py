"""
Calibrated Post-Generation Confidence Estimator
=================================================
Replaces the old 1-BitRAG v1 hand-weighted heuristic system
(40% relevance / 30% uncertainty / 15% hedging / 15% assertiveness).

Instead: trains a logistic regression on measurable output features,
with BERTScore success as the supervision signal.

Research role:
    - Mode B in the ablation: always call cheap LLM, calibrated post-router
      gates escalation.
    - Mode C (full system): secondary gate that catches cheap-LLM failures
      that the pre-router missed.

Prior work contrast:
    v1 hand-weighted heuristics have no theoretical grounding and require
    manual weight tuning.  This module learns the weights from data.

Features extracted from cheap LLM output (7 total):
    1. answer_length_norm   : character length / 500 (clipped to 1.0)
    2. query_answer_overlap : Jaccard token overlap between query and answer
    3. uncertainty_count    : normalised count of uncertainty phrases
    4. hedge_ratio          : hedging words / content word count
    5. assertion_score      : (strong verbs + quantities) / answer length
    6. answer_entropy       : Shannon entropy of answer token distribution
    7. self_echo_penalty    : 1.0 if answer mostly echoes the query
"""

import re
import numpy as np
import joblib
from scipy.stats import entropy as scipy_entropy
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from pathlib import Path

from config import MODELS_DIR, RANDOM_STATE

# ── Signal lexicons ──────────────────────────────────────────────────────────
UNCERTAINTY_PHRASES = [
    "i don't know", "cannot determine", "not provided", "unclear",
    "i'm not sure", "i am not sure", "cannot say", "unknown",
    "not enough information", "cannot answer", "no information",
]
HEDGE_WORDS = frozenset({
    "maybe", "possibly", "perhaps", "might", "could", "seems",
    "appears", "likely", "unlikely", "speculate",
})
STRONG_VERBS = frozenset({
    "causes", "prevents", "increases", "decreases", "requires",
    "produces", "results", "leads", "enables", "inhibits",
    "reduces", "triggers", "treats", "diagnoses",
})
QUANTITY_PATTERN = re.compile(
    r"\b\d+\.?\d*\s*(%|mg|ml|kg|g|years?|days?|hours?|minutes?)\b"
)

# Feature names for logging and interpretability
POST_FEATURE_NAMES = [
    "answer_length_norm", "query_answer_overlap", "uncertainty_count",
    "hedge_ratio", "assertion_score", "answer_entropy", "self_echo_penalty",
]


class PostRouter:
    """Calibrated post-generation confidence estimator.

    Replaces v1's hand-weighted heuristics with a learned, calibrated
    logistic regression over output features.
    """

    def __init__(self):
        self.model = None  # set in train()
        self.scaler = StandardScaler()
        self.threshold = 0.7
        self.is_trained = False

    def extract_features(self, query: str, answer: str) -> np.ndarray:
        """Extract features from the cheap LLM's answer for confidence estimation.

        Args:
            query:  the original user query.
            answer: the cheap LLM's generated response.

        Returns:
            (7,) float32 feature vector.
        """
        q_tokens = set(query.lower().split())
        a_tokens = answer.lower().split()
        a_content = [t for t in a_tokens if len(t) > 3]

        eps = 1e-9
        answer_len = max(len(answer), 1)

        # 1. Normalised answer length
        length_norm = min(len(answer) / 500.0, 1.0)

        # 2. Query-answer token overlap (Jaccard)
        a_set = set(a_tokens)
        union = q_tokens | a_set
        overlap = len(q_tokens & a_set) / (len(union) + eps) if union else 0.0

        # 3. Uncertainty count (normalised by answer length)
        answer_lower = answer.lower()
        unc_count = sum(
            1 for p in UNCERTAINTY_PHRASES if p in answer_lower
        ) / (answer_len / 100.0 + eps)

        # 4. Hedge ratio (hedge words / content words)
        hedge_ratio = sum(
            1 for t in a_tokens if t in HEDGE_WORDS
        ) / (len(a_content) + eps)

        # 5. Assertion score (strong verbs + quantities / answer length)
        strong_verb_count = sum(1 for t in a_tokens if t in STRONG_VERBS)
        quantity_count = len(QUANTITY_PATTERN.findall(answer))
        assertion_score = (strong_verb_count + quantity_count) / (
            answer_len / 50.0 + eps
        )

        # 6. Answer entropy (token distribution)
        token_counts: dict[str, int] = {}
        for t in a_tokens:
            token_counts[t] = token_counts.get(t, 0) + 1
        if token_counts:
            probs = np.array(list(token_counts.values()), dtype=float)
            probs /= probs.sum()
            answer_entropy = float(scipy_entropy(probs + eps))
        else:
            answer_entropy = 0.0

        # 7. Self-echo penalty
        echo = len(q_tokens & a_set) / (len(q_tokens) + eps) if q_tokens else 0.0
        self_echo = 1.0 if echo > 0.85 else 0.0

        return np.array([
            length_norm, overlap, unc_count, hedge_ratio,
            assertion_score, answer_entropy, self_echo,
        ], dtype=np.float32)

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Train the post-router on extracted output features.

        Args:
            X: (n_samples, 7) feature matrix from extract_features().
            y: (n_samples,) binary labels (1 = cheap answer was good).

        Returns:
            Training report dict.
        """
        X_scaled = self.scaler.fit_transform(X)

        # Adaptive: use fewer folds / skip calibration for small datasets
        n_min_class = int(min(y.sum(), len(y) - y.sum()))
        cv_folds = max(2, min(5, n_min_class))

        base = LogisticRegression(
            C=0.1, max_iter=2000, random_state=RANDOM_STATE, penalty="l2",
        )

        if n_min_class >= 10:
            self.model = CalibratedClassifierCV(
                base, method="sigmoid", cv=cv_folds
            )
        else:
            self.model = base

        self.model.fit(X_scaled, y)
        self.is_trained = True

        y_prob = self.model.predict_proba(X_scaled)[:, 1]
        report = {
            "post_router_train_auc": float(roc_auc_score(y, y_prob)),
            "n_samples": int(len(y)),
            "positive_rate": float(y.mean()),
        }

        # Extract coefficients for interpretability
        try:
            if hasattr(self.model, 'calibrated_classifiers_'):
                base_lr = self.model.calibrated_classifiers_[0].estimator
            else:
                base_lr = self.model
            report["feature_coefficients"] = dict(zip(
                POST_FEATURE_NAMES, base_lr.coef_[0].tolist()
            ))
        except (AttributeError, IndexError):
            pass

        return report

    def predict_confidence(self, query: str, answer: str) -> float:
        """Return P(cheap LLM answer is correct) -- calibrated.

        Args:
            query:  original user query.
            answer: cheap LLM's response.

        Returns:
            Calibrated probability in [0, 1].
        """
        assert self.is_trained, "Call train() before predict_confidence()"
        x = self.extract_features(query, answer)
        x_scaled = self.scaler.transform(x.reshape(1, -1))
        return float(self.model.predict_proba(x_scaled)[0, 1])

    def should_escalate(self, query: str, answer: str) -> tuple[bool, float]:
        """Decide whether to escalate to the full LLM.

        Returns:
            (escalate: bool, confidence: float)
        """
        conf = self.predict_confidence(query, answer)
        return conf < self.threshold, conf

    def save(self) -> None:
        """Persist the trained model to disk."""
        Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
        path = Path(MODELS_DIR) / "post_router.pkl"
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "threshold": self.threshold,
        }, path)
        print(f"  Saved post-router to {path}")

    def load(self) -> None:
        """Load a previously trained model from disk."""
        path = Path(MODELS_DIR) / "post_router.pkl"
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.threshold = data["threshold"]
        self.is_trained = True
        print(f"  Loaded post-router from {path}")
