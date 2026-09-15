"""
INTERFACE CONTRACT — read this before writing any real logic.

This file defines the function signatures that let two people work in
parallel without breaking each other's code. Person A implements the CF
functions, Person B implements the content functions. Nobody should need
to read the *inside* of the other person's code to use it — just these
signatures.

Do not put real logic in this file. Each stub below lives in its real
module (cf_model/, content_model/, etc.) and should match this exact
signature. This file is just the map.
"""

# ---- Owned by Person A (CF / prediction track) ----

def predict_rating(user_id: int, movie_id: int) -> float:
    """
    Predict what rating (0.5-5.0) a user would give a movie they haven't
    rated. Backed by cf_model/train_cf.py's trained model.
    Returns: predicted rating as a float.
    """
    raise NotImplementedError

def top_cf_candidates(user_id: int, k: int = 200) -> list[tuple[int, float]]:
    """
    Return the top-k movies (movie_id, predicted_rating) this user hasn't
    rated yet, sorted descending by predicted rating.
    """
    raise NotImplementedError


# ---- Owned by Person B (content / exploration track) ----

def get_similar_movies(movie_id: int, k: int = 10) -> list[tuple[int, float]]:
    """
    Return the k most similar movies to movie_id (movie_id, similarity_score),
    based on content embeddings (genres/tags). Backed by
    content_model/build_embeddings.py + a FAISS index.
    """
    raise NotImplementedError

def choose_with_bandit(user_id: int, candidates: list[int]) -> int:
    """
    Given a list of candidate movie_ids (already retrieved/ranked), use the
    bandit to pick which one to actually show — balancing exploration vs
    exploitation. Backed by bandit/bandit.py.
    Returns: a single movie_id from the candidates list.
    """
    raise NotImplementedError


# ---- Joint (built together at the Phase 3 checkpoint) ----

def get_candidates(user_id: int, k: int = 200) -> list[int]:
    """
    Combine top_cf_candidates() and get_similar_movies() (seeded from the
    user's highest-rated movie) into a single deduplicated candidate list.
    This is the retrieval layer — the first place A's and B's code meet.
    """
    raise NotImplementedError

def recommend(user_id: int, mode: str = "ranked", k: int = 10) -> list[int]:
    """
    The top-level function the Streamlit app calls.
    mode="ranked"  -> plain best-guess ranked list (no bandit)
    mode="explore" -> candidates run through choose_with_bandit()
    Returns: list of movie_ids in the order to display.
    """
    raise NotImplementedError
