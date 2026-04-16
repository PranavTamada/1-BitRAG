def fuse_scores(dense_scores_dict, sparse_scores_dict, k=5):
    # Step 1: Create rank lookups (1-based indexing)
    # We sort each dictionary by its score (descending) to find the rank of each index
    dense_ranked = sorted(dense_scores_dict.items(), key=lambda x: x[1], reverse=True)
    sparse_ranked = sorted(sparse_scores_dict.items(), key=lambda x: x[1], reverse=True)

    dense_ranks = {idx: rank + 1 for rank, (idx, _) in enumerate(dense_ranked)}
    sparse_ranks = {idx: rank + 1 for rank, (idx, _) in enumerate(sparse_ranked)}

    all_indices = set(dense_scores_dict.keys()) | set(sparse_scores_dict.keys())

    fused = []
    # RRF constant (k=50 is the gold standard used in research papers like Cormack et al.)
    rrf_constant = 50

    for idx in all_indices:
        # Get ranks; if an index is missing from a ranker, it contributes 0 to the score
        d_rank = dense_ranks.get(idx)
        s_rank = sparse_ranks.get(idx)

        # Apply RRF formula: 1 / (constant + rank)
        d_rrf_score = 1.0 / (rrf_constant + d_rank) if d_rank else 0.0
        s_rrf_score = 1.0 / (rrf_constant + s_rank) if s_rank else 0.0

        fused_score = d_rrf_score + s_rrf_score
        
        # Updated print for RRF debugging
        print(f"Index: {idx} | Dense Rank: {d_rank} | Sparse Rank: {s_rank} | Fused: {fused_score:.5f}")

        fused.append((idx, fused_score))

    # Sort by the new fused RRF score
    fused.sort(key=lambda x: x[1], reverse=True)

    return fused[:k]