"""
Person A owns this file.

Trains an SVD collaborative filtering model on ratings.csv using the
`surprise` library, and implements predict_rating() / top_cf_candidates()
from interfaces.py.

Usage: python cf_model/train_cf.py
"""
import pickle
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

RATINGS_PATH = "data/ml-latest-small/ratings.csv"
MODEL_OUT = "data/cf_model.pkl"


def load_ratings():
    return pd.read_csv(RATINGS_PATH)


def train():
    """
    Loads ratings, cross-validates an SVD model (prints RMSE/MAE), then
    retrains on the full dataset and pickles the final model to MODEL_OUT.
    """
    df = load_ratings()

    # surprise needs exactly these three columns, in this order, and a
    # Reader telling it the valid rating range.
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(df[["userId", "movieId", "rating"]], reader)

    # Cross-validate first so we have a real accuracy number before we
    # commit to using this model — this is the RMSE/MAE check.
    algo = SVD(n_factors=20, reg_all=0.08, random_state=42)
    results = cross_validate(algo, data, measures=["RMSE", "MAE"], cv=5, verbose=True)
    print(f"\nMean RMSE: {results['test_rmse'].mean():.4f}")
    print(f"Mean MAE:  {results['test_mae'].mean():.4f}")

    # Now retrain on ALL the data (no held-out fold) — this is the final
    # model we'll actually serve predictions from.
    full_trainset = data.build_full_trainset()
    final_model = SVD(n_factors=20, reg_all=0.08, random_state=42)
    final_model.fit(full_trainset)

    with open(MODEL_OUT, "wb") as f:
        pickle.dump(final_model, f)
    print(f"\nSaved trained model -> {MODEL_OUT}")


def _load_model():
    with open(MODEL_OUT, "rb") as f:
        return pickle.load(f)


def predict_rating(user_id: int, movie_id: int) -> float:
    """Predict what rating user_id would give movie_id."""
    model = _load_model()
    return model.predict(user_id, movie_id).est


def top_cf_candidates(user_id: int, k: int = 200) -> list[tuple[int, float]]:
    """
    Return the top-k movies (movie_id, predicted_rating) this user hasn't
    rated yet, sorted descending by predicted rating.
    """
    model = _load_model()
    df = load_ratings()

    all_movie_ids = df["movieId"].unique()
    already_rated = set(df[df["userId"] == user_id]["movieId"])
    unrated = [m for m in all_movie_ids if m not in already_rated]

    predictions = [(m, model.predict(user_id, m).est) for m in unrated]
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:k]


if __name__ == "__main__":
    train()

    # quick sanity check after training
    test_user = 1
    print(f"\nTop 5 candidates for user {test_user}:")
    for movie_id, score in top_cf_candidates(test_user, k=5):
        print(f"  movieId={movie_id}  predicted_rating={score:.2f}")