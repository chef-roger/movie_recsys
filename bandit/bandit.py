"""
Person B owns this file (Phase 5).

LinUCB contextual bandit: picks which candidate to actually show, balancing
exploration vs exploitation. Context = the user's latent taste vector,
reused directly from the trained SVD model (cf_model.pkl) so we don't need
to build a separate feature system.
"""
import pickle
import numpy as np
from collections import defaultdict

CF_MODEL_PATH = "data/cf_model.pkl"


def _load_cf_model():
    with open(CF_MODEL_PATH, "rb") as f:
        return pickle.load(f)


def get_user_context(user_id: int) -> np.ndarray:
    """
    Returns the user's latent taste vector from the trained SVD model
    (shape: n_factors,). This is what LinUCB uses as its context.
    Falls back to a zero vector for users the CF model has never seen
    (cold-start users) — LinUCB will just start with no prior belief
    about them, which is the correct behavior.
    """
    model = _load_cf_model()
    try:
        inner_uid = model.trainset.to_inner_uid(user_id)
        return model.pu[inner_uid].copy()
    except ValueError:
        return np.zeros(model.pu.shape[1])


class LinUCBBandit:
    """
    One ridge-regression model per arm (movie). At selection time, picks
    the arm with the highest predicted reward plus a confidence bonus that
    shrinks as more data comes in for that arm - this is what balances
    exploration (uncertain arms) against exploitation (arms we're
    confident about).
    """

    def __init__(self, context_dim: int, alpha: float = 1.0):
        self.context_dim = context_dim
        self.alpha = alpha
        # A: per-arm d x d matrix (starts as identity - ridge regression prior)
        # b: per-arm d-length vector (starts at zero)
        self.A = defaultdict(lambda: np.identity(context_dim))
        self.b = defaultdict(lambda: np.zeros(context_dim))

    def select(self, candidate_movie_ids: list[int], context: np.ndarray) -> int:
        """
        Pick the candidate with the highest UCB score for this context.
        Ties are broken randomly - not by list order - since arms that
        haven't been updated yet (including brand-new, never-seen movies)
        can genuinely tie exactly, and always favoring whichever arm
        appears first in the list would silently starve everything after it.
        """
        scores = {}
        for movie_id in candidate_movie_ids:
            A_inv = np.linalg.inv(self.A[movie_id])
            theta = A_inv @ self.b[movie_id]  # ridge regression estimate

            predicted_reward = theta @ context
            confidence_bonus = self.alpha * np.sqrt(context @ A_inv @ context)
            scores[movie_id] = predicted_reward + confidence_bonus

        best_score = max(scores.values())
        tied_best = [mid for mid, s in scores.items() if s == best_score]
        return np.random.default_rng().choice(tied_best)

    def update(self, movie_id: int, context: np.ndarray, reward: float):
        """After observing a real reward for this arm, update its ridge regression."""
        self.A[movie_id] += np.outer(context, context)
        self.b[movie_id] += reward * context


# Module-level bandit instance so it persists state across calls within
# one process (needed for the simulator to actually learn over rounds).
_bandit_instance = None


def _get_bandit():
    global _bandit_instance
    if _bandit_instance is None:
        model = _load_cf_model()
        context_dim = model.pu.shape[1]
        _bandit_instance = LinUCBBandit(context_dim=context_dim, alpha=1.0)
    return _bandit_instance


def choose_with_bandit(user_id: int, candidates: list[int]) -> int:
    """Wraps a persistent LinUCBBandit instance and picks from candidates."""
    context = get_user_context(user_id)
    return _get_bandit().select(candidates, context)


if __name__ == "__main__":
    import pandas as pd

    ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
    movies = pd.read_csv("data/ml-latest-small/movies.csv").set_index("movieId")["title"]

    bandit = _get_bandit()
    rng = np.random.default_rng(42)

    # Fix a small pool of movies up front and reuse it across every round -
    # LinUCB can only get more confident about an arm if it sees that same
    # arm repeatedly. Resampling random distractors every round (the first
    # version of this test) meant almost no arm was ever seen twice.
    pool = ratings["movieId"].drop_duplicates().sample(30, random_state=42).tolist()
    pool_ratings = ratings[ratings["movieId"].isin(pool)]

    n_rounds = 1000
    hits = 0
    for i in range(n_rounds):
        row = pool_ratings.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        user_id = int(row["userId"])
        real_movie_id = int(row["movieId"])
        real_rating = float(row["rating"])

        context = get_user_context(user_id)
        chosen = bandit.select(pool, context)

        if chosen == real_movie_id:
            hits += 1
            bandit.update(chosen, context, real_rating)

    print(f"Bandit picked the real rated movie in {hits}/{n_rounds} rounds "
          f"({100*hits/n_rounds:.1f}%)  [baseline random: {100/len(pool):.1f}%]")
    print("(Smoke test only - the real evaluation belongs in eval_bandit.py)")