"""
Person A owns this file (Phase 2/6).

RMSE/MAE via surprise cross-validation, plus Precision@K/Recall@K/NDCG@K
on the ranked output.
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate, train_test_split

RATINGS_PATH = "data/ml-latest-small/ratings.csv"


def evaluate_cf_accuracy():
    """RMSE/MAE cross-validation - same settings as the trained model in
    train_cf.py, so this number is directly comparable to it."""
    df = pd.read_csv(RATINGS_PATH)
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(df[["userId", "movieId", "rating"]], reader)

    algo = SVD(n_factors=20, reg_all=0.08, random_state=42)
    results = cross_validate(algo, data, measures=["RMSE", "MAE"], cv=5, verbose=True)
    print(f"\nMean RMSE: {results['test_rmse'].mean():.4f}")
    print(f"Mean MAE:  {results['test_mae'].mean():.4f}")
    return results


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """
    Standard precision/recall@k: for each user, take their top-k predicted
    items, check how many are actually "relevant" (real rating >= threshold).

    precision@k = (relevant items in top-k) / k
    recall@k    = (relevant items in top-k) / (total relevant items for user)

    Returns (mean_precision, mean_recall) averaged across all users.
    """
    user_predictions = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_predictions[uid].append((est, true_r))

    precisions, recalls = {}, {}
    for uid, user_ratings in user_predictions.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)  # sort by predicted score desc

        n_relevant = sum(true_r >= threshold for (_, true_r) in user_ratings)
        top_k = user_ratings[:k]
        n_relevant_in_top_k = sum(true_r >= threshold for (_, true_r) in top_k)

        precisions[uid] = n_relevant_in_top_k / k if k != 0 else 0
        recalls[uid] = n_relevant_in_top_k / n_relevant if n_relevant != 0 else 0

    mean_precision = sum(precisions.values()) / len(precisions)
    mean_recall = sum(recalls.values()) / len(recalls)
    return mean_precision, mean_recall


def ndcg_at_k(predictions, k=10):
    """
    NDCG@k: rewards putting truly-high-rated items near the TOP of the
    list, not just anywhere in the top-k (unlike precision/recall, which
    don't care about order within the top-k).
    """
    user_predictions = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_predictions[uid].append((est, true_r))

    ndcgs = []
    for uid, user_ratings in user_predictions.items():
        # DCG: rank by our predicted order, discount relevance by log2(position+1)
        by_prediction = sorted(user_ratings, key=lambda x: x[0], reverse=True)[:k]
        dcg = sum(true_r / np.log2(i + 2) for i, (_, true_r) in enumerate(by_prediction))

        # Ideal DCG: same items, but in the BEST possible order (sorted by true rating)
        by_ideal = sorted(user_ratings, key=lambda x: x[1], reverse=True)[:k]
        idcg = sum(true_r / np.log2(i + 2) for i, (_, true_r) in enumerate(by_ideal))

        ndcgs.append(dcg / idcg if idcg > 0 else 0)

    return sum(ndcgs) / len(ndcgs)


def evaluate_ranking_quality(k=10, threshold=3.5):
    """Trains on 80%, tests on the held-out 20%, reports Precision/Recall/NDCG@k."""
    df = pd.read_csv(RATINGS_PATH)
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(df[["userId", "movieId", "rating"]], reader)

    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    algo = SVD(n_factors=20, reg_all=0.08, random_state=42)
    algo.fit(trainset)
    predictions = algo.test(testset)

    precision, recall = precision_recall_at_k(predictions, k=k, threshold=threshold)
    ndcg = ndcg_at_k(predictions, k=k)

    print(f"\nPrecision@{k}: {precision:.4f}")
    print(f"Recall@{k}:    {recall:.4f}")
    print(f"NDCG@{k}:      {ndcg:.4f}")
    return precision, recall, ndcg


if __name__ == "__main__":
    print("=== RMSE/MAE (cross-validated) ===")
    evaluate_cf_accuracy()

    print("\n=== Ranking quality (80/20 split) ===")
    evaluate_ranking_quality(k=10)