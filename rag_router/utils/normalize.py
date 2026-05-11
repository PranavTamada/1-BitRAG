"""
Normalisation Utilities
========================
Min-max score normalisation used in retrieval and evaluation.
"""


def min_max_normalize(scores: list[float]) -> list[float]:
    """Normalise a list of scores to [0, 1] via min-max scaling."""
    if len(scores) == 0:
        return scores

    min_s = min(scores)
    max_s = max(scores)

    if max_s - min_s == 0:
        return [0.0 for _ in scores]

    return [(s - min_s) / (max_s - min_s + 1e-8) for s in scores]
