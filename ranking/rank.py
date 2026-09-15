"""
Person A owns this file (Phase 4).

Blends CF score + content similarity into a single ranked list.
Starts simple (weighted sum), can graduate to LightGBM later.
"""
from cf_model.train_cf import predict_rating


def rank(user_id: int, candidate_movie_ids: list[int]) -> list[int]:
    """
    TODO (Person A):
    1. For each candidate, get predict_rating(user_id, movie_id)
    2. (Optional) blend in content similarity / popularity as extra signal
    3. Sort descending, return ordered list of movie_ids
    """
    raise NotImplementedError
