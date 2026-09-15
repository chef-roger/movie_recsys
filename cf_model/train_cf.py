"""
Person A owns this file.

Trains an SVD collaborative filtering model on ratings.csv using the
`surprise` library, and implements predict_rating() / top_cf_candidates()
from interfaces.py.

Usage: python cf_model/train_cf.py
"""
import pandas as pd
# from surprise import SVD, Dataset, Reader
# from surprise.model_selection import cross_validate

RATINGS_PATH = "data/ml-latest-small/ratings.csv"
MODEL_OUT = "data/cf_model.pkl"


def load_ratings():
    return pd.read_csv(RATINGS_PATH)


def train():
    """
    TODO (Person A):
    1. Load ratings with load_ratings()
    2. Wrap in surprise's Reader/Dataset (rating_scale=(0.5, 5.0))
    3. Train SVD, run cross_validate() for RMSE/MAE
    4. Pickle the trained model to MODEL_OUT
    """
    raise NotImplementedError


def predict_rating(user_id: int, movie_id: int) -> float:
    """TODO: load MODEL_OUT, call model.predict(user_id, movie_id).est"""
    raise NotImplementedError


def top_cf_candidates(user_id: int, k: int = 200) -> list[tuple[int, float]]:
    """TODO: predict_rating() over all unrated movies for this user, sort, take top k."""
    raise NotImplementedError


if __name__ == "__main__":
    train()
