import pickle
import numpy as np

CF_MODEL_PATH = "data/cf_model.pkl"


def _load_cf_model():
    with open(CF_MODEL_PATH, "rb") as f:
        return pickle.load(f)


def get_user_context(user_id: int) -> np.ndarray:
    model = _load_cf_model()
    try:
        inner_uid = model.trainset.to_inner_uid(user_id)
        return model.pu[inner_uid].copy()
    except ValueError:
        return np.zeros(model.pu.shape[1])


class LinUCBBandit:
    def __init__(self, context_dim: int, alpha: float = 1.0):
        self.context_dim = context_dim
        self.alpha = alpha
        self.A = {}
        self.b = {}

    def _ensure_arm(self, movie_id):
        if movie_id not in self.A:
            self.A[movie_id] = np.identity(self.context_dim)
            self.b[movie_id] = np.zeros(self.context_dim)

    def select(self, candidate_movie_ids: list[int], context: np.ndarray) -> int:
        scores = {}
        for movie_id in candidate_movie_ids:
            self._ensure_arm(movie_id)
            A_inv = np.linalg.inv(self.A[movie_id])
            theta = A_inv @ self.b[movie_id]
            predicted_reward = theta @ context
            confidence_bonus = self.alpha * np.sqrt(context @ A_inv @ context)
            scores[movie_id] = predicted_reward + confidence_bonus

        best_score = max(scores.values())
        tied_best = [mid for mid, s in scores.items() if s == best_score]
        return np.random.default_rng().choice(tied_best)

    def update(self, movie_id: int, context: np.ndarray, reward: float):
        self._ensure_arm(movie_id)
        self.A[movie_id] += np.outer(context, context)
        self.b[movie_id] += reward * context


_bandit_instance = None


def _get_bandit():
    global _bandit_instance
    if _bandit_instance is None:
        model = _load_cf_model()
        context_dim = model.pu.shape[1]
        _bandit_instance = LinUCBBandit(context_dim=context_dim, alpha=1.0)
    return _bandit_instance


def choose_with_bandit(user_id: int, candidates: list[int]) -> int:
    context = get_user_context(user_id)
    return _get_bandit().select(candidates, context)


if __name__ == "__main__":
    import pandas as pd

    ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
    movies = pd.read_csv("data/ml-latest-small/movies.csv").set_index("movieId")["title"]

    bandit = _get_bandit()
    rng = np.random.default_rng(42)

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