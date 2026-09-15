"""
JOINT FILE — built together once cf_model and content_model both work.

Combines top_cf_candidates() (Person A) and get_similar_movies() (Person B)
into a single deduplicated candidate shortlist.
"""
from cf_model.train_cf import top_cf_candidates
from content_model.build_embeddings import get_similar_movies


def get_candidates(user_id: int, k: int = 200) -> list[int]:
    """
    TODO (joint):
    1. Get CF candidates: top_cf_candidates(user_id, k)
    2. Find the user's highest-rated movie, get_similar_movies() on it
    3. Merge both lists, dedupe by movie_id, return up to k movie_ids
    """
    raise NotImplementedError
