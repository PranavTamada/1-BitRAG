"""
Reciprocal Rank Fusion (RRF)
============================
Fuses dense and sparse retrieval results using the RRF formula from
Cormack et al. (2009):  ``score(d) = Σ  1 / (k + rank_i(d))``

Enhanced from v1 to return per-system ranking metadata that the
retrieval-feature extractor needs for the ``bm25_dense_agreement``
signal (Spearman rank correlation between BM25 and dense rankings).
"""

from config import RRF_K


def fuse_scores(
    dense_scores_dict: dict[int, float],
    sparse_scores_dict: dict[int, float],
    k: int = 5,
) -> tuple[list[tuple[int, float]], dict[int, int], dict[int, int]]:
    """Fuse dense and sparse scores using Reciprocal Rank Fusion.

    Args:
        dense_scores_dict:  {doc_index: dense_score}
        sparse_scores_dict: {doc_index: sparse_score}
        k: number of top results to return

    Returns:
        fused_results: list of (doc_index, fused_rrf_score) in descending order
        dense_ranks:   {doc_index: dense_rank}   (1-based)
        sparse_ranks:  {doc_index: sparse_rank}  (1-based)

    Research note:
        Returning per-system ranks is the key enhancement over v1.
        The feature extractor computes Spearman correlation between these
        ranks to measure retriever agreement — a novel difficulty signal.
    """
    # Build 1-based rank lookups sorted by score descending
    dense_ranked = sorted(dense_scores_dict.items(), key=lambda x: x[1], reverse=True)
    sparse_ranked = sorted(sparse_scores_dict.items(), key=lambda x: x[1], reverse=True)

    dense_ranks = {idx: rank + 1 for rank, (idx, _) in enumerate(dense_ranked)}
    sparse_ranks = {idx: rank + 1 for rank, (idx, _) in enumerate(sparse_ranked)}

    all_indices = set(dense_scores_dict.keys()) | set(sparse_scores_dict.keys())

    fused = []
    for idx in all_indices:
        d_rank = dense_ranks.get(idx)
        s_rank = sparse_ranks.get(idx)

        # RRF formula: 1 / (constant + rank)
        d_rrf = 1.0 / (RRF_K + d_rank) if d_rank else 0.0
        s_rrf = 1.0 / (RRF_K + s_rank) if s_rank else 0.0

        fused.append((idx, d_rrf + s_rrf))

    fused.sort(key=lambda x: x[1], reverse=True)
    return fused[:k], dense_ranks, sparse_ranks
