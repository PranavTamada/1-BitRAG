"""
Pre-Generation Routing Classifier
===================================
Trained on: retrieval geometry features + query complexity features
Label: 1 if cheap LLM would succeed (BERTScore F1 >= threshold), 0 otherwise
Model: Logistic Regression (interpretable) + GBT (performance comparison)

Research design decisions:
    - Logistic Regression is the paper's primary model because it produces
      interpretable feature coefficients that directly support the narrative:
      "score_gap has coefficient X, confirming retrieval geometry predicts
      difficulty."
    - GBT is trained for comparison: if LR and GBT perform similarly, it
      validates that the feature space is sufficient without complex
      nonlinearities.
    - Both models are calibrated with Platt scaling (CalibratedClassifierCV)
      to produce true probability outputs for threshold sweeping.

Prior work contrast:
    Post-Gen Cascade (our baseline): adapts the cascade strategy from
        FrugalGPT (Chen et al., 2023) to the RAG setting -- routes based
        on *post-generation* signals (must call cheap LLM first).
    RouteLLM: requires human preference labels (expensive annotation).
    Our pre-router: routes using only retrieval features + query text.
    Zero LLM calls. Zero human annotation. Sub-millisecond inference.
"""

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from pathlib import Path

from config import MODELS_DIR, RANDOM_STATE


class PreRouter:
    """Pre-generation routing classifier.

    Predicts P(cheap LLM will succeed) from retrieval geometry + query
    features *before* any LLM is called.  This is the paper's core
    contribution.
    """

    def __init__(self, model_type: str = "logistic"):
        """
        Args:
            model_type: "logistic" (interpretable, paper primary) or
                        "gbt" (gradient boosted trees, for comparison)
        """
        assert model_type in ("logistic", "gbt"), (
            f"model_type must be 'logistic' or 'gbt', got '{model_type}'"
        )
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.threshold = 0.5  # Optimised by budget_optimizer.py
        self._base_model = None  # set in train()
        self.model = None        # set in train()
        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Train the pre-router on feature matrix X and binary labels y.

        Research outputs:
            - Cross-validated AUC (5-fold) for unbiased performance estimate
            - Feature coefficients (logistic only) for Table 3 in the paper
            - Positive rate for label balance verification

        Args:
            X: (n_samples, 18) feature matrix — 10 retrieval + 8 query features
            y: (n_samples,) binary labels — 1 if cheap LLM succeeds

        Returns:
            Training report dict for logging.
        """
        X_scaled = self.scaler.fit_transform(X)

        # Adaptive CV: use fewer folds for small datasets
        n_min_class = int(min(y.sum(), len(y) - y.sum()))
        cv_folds = max(2, min(5, n_min_class))

        # Build model: use calibration only if we have enough samples
        if self.model_type == "logistic":
            base = LogisticRegression(
                C=0.1, max_iter=2000, random_state=RANDOM_STATE,
                penalty="l2",
            )
        else:
            base = GradientBoostingClassifier(
                n_estimators=50, max_depth=2, learning_rate=0.05,
                random_state=RANDOM_STATE,
            )
        self._base_model = base

        if n_min_class >= 10:
            self.model = CalibratedClassifierCV(
                base, method="sigmoid", cv=cv_folds
            )
        else:
            # Too few samples for calibration; use raw model
            self.model = base

        self.model.fit(X_scaled, y)
        self.is_trained = True

        # Cross-validated AUC with stratified folds
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                              random_state=RANDOM_STATE)
        cv_scores = cross_val_score(
            self.model, X_scaled, y, cv=skf, scoring="roc_auc"
        )
        y_prob = self.model.predict_proba(X_scaled)[:, 1]

        report = {
            "model_type": self.model_type,
            "cv_auc_mean": float(cv_scores.mean()),
            "cv_auc_std": float(cv_scores.std()),
            "train_auc": float(roc_auc_score(y, y_prob)),
            "n_samples": int(len(y)),
            "positive_rate": float(y.mean()),
        }

        # Extract feature coefficients for the paper (logistic only)
        if self.model_type == "logistic":
            try:
                # Try calibrated wrapper first, then raw model
                if hasattr(self.model, 'calibrated_classifiers_'):
                    base_lr = self.model.calibrated_classifiers_[0].estimator
                else:
                    base_lr = self.model
                report["feature_coefficients"] = dict(zip(
                    self._feature_names(), base_lr.coef_[0].tolist()
                ))
                report["intercept"] = float(base_lr.intercept_[0])
            except (AttributeError, IndexError):
                report["feature_coefficients"] = {}

        return report

    def predict_proba(self, x: np.ndarray) -> float:
        """Return P(cheap LLM will succeed) for a single feature vector.

        Args:
            x: (18,) feature vector — 10 retrieval + 8 query features

        Returns:
            float in [0, 1] — calibrated probability.
        """
        assert self.is_trained, "Call train() before predict_proba()"
        x_scaled = self.scaler.transform(x.reshape(1, -1))
        return float(self.model.predict_proba(x_scaled)[0, 1])

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        """Return P(cheap succeeds) for a batch of feature vectors.

        Args:
            X: (n, 18) feature matrix

        Returns:
            (n,) array of calibrated probabilities.
        """
        assert self.is_trained, "Call train() before predict_proba_batch()"
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def route(self, x: np.ndarray) -> tuple[str, float]:
        """Make a routing decision for a single query.

        Returns:
            ("cheap", confidence) if routing_confidence >= threshold
            ("full", confidence)  otherwise

        Uses self.threshold, which is set by budget_optimizer.py.
        """
        p = self.predict_proba(x)
        decision = "cheap" if p >= self.threshold else "full"
        return decision, p

    def save(self, name: str = "pre_router") -> None:
        """Persist the trained model to disk."""
        Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
        path = Path(MODELS_DIR) / f"{name}_{self.model_type}.pkl"
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "threshold": self.threshold,
            "model_type": self.model_type,
        }, path)
        print(f"  Saved pre-router to {path}")

    def load(self, name: str = "pre_router") -> None:
        """Load a previously trained model from disk."""
        path = Path(MODELS_DIR) / f"{name}_{self.model_type}.pkl"
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.threshold = data["threshold"]
        self.model_type = data["model_type"]
        self.is_trained = True
        print(f"  Loaded pre-router from {path}")

    @staticmethod
    def _feature_names() -> list[str]:
        """Canonical feature names in the same order as the feature vector."""
        return [
            # Retrieval geometry features (10)
            "score_gap", "score_mean", "score_variance", "score_entropy",
            "top_score", "score_ratio", "low_score_fraction", "retrieval_hit",
            "bm25_dense_agreement", "context_density",
            # Query complexity features (8)
            "query_length", "query_entropy", "has_negation", "has_conditional",
            "question_count", "has_comparison", "avg_word_length",
            "named_entity_count",
        ]
