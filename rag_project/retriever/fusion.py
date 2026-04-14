from utils.normalize import min_max_normalize

def fuse_scores(docs_dense, scores_dense, docs_sparse, scores_sparse, k=5):
    # Normalize scores
    scores_dense = min_max_normalize(scores_dense)
    scores_sparse = min_max_normalize(scores_sparse)

    score_dict = {}

    # Add dense scores
    for doc, score in zip(docs_dense, scores_dense):
        score_dict[doc] = score_dict.get(doc, 0) + 0.5 * score

    # Add sparse scores
    for doc, score in zip(docs_sparse, scores_sparse):
        score_dict[doc] = score_dict.get(doc, 0) + 0.5 * score

    # Sort
    ranked = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)

    docs = [doc for doc, _ in ranked[:k]]
    scores = [score for _, score in ranked[:k]]

    return docs, scores