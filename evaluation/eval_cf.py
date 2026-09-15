"""
Person A owns this file (Phase 2/6).

RMSE/MAE via surprise cross-validation, plus Precision@K/Recall@K/NDCG@K
on the ranked output.
"""


def evaluate_cf_accuracy():
    """TODO: surprise cross_validate(SVD(), data, measures=['RMSE','MAE'], cv=5)"""
    raise NotImplementedError


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """TODO: standard surprise-cookbook precision/recall@k implementation."""
    raise NotImplementedError
