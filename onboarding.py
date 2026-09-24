import os
import pickle
import numpy as np
import pandas as pd

from content_model.build_embeddings import get_similar_movies
from bandit.bandit import LinUCBBandit

CF_MODEL_PATH = "data/cf_model.pkl"
MOVIES_PATH = "data/ml-latest-small/movies.csv"
RATINGS_PATH = "data/ml-latest-small/ratings.csv"
USER_BANDITS_PATH = "data/user_bandits.pkl"

ONBOARDING_THRESHOLD = 5
POPULAR_POOL_SIZE = 50


def load_cf_model():
    with open(CF_MODEL_PATH, "rb") as f:
        return pickle.load(f)


def load_user_bandits():
    if os.path.exists(USER_BANDITS_PATH):
        with open(USER_BANDITS_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def save_user_bandits(user_bandits):
    with open(USER_BANDITS_PATH, "wb") as f:
        pickle.dump(user_bandits, f)


def get_bandit_for_user(user_id, context_dim):
    user_bandits = load_user_bandits()
    if user_id not in user_bandits:
        user_bandits[user_id] = LinUCBBandit(context_dim=context_dim, alpha=1.0)
        save_user_bandits(user_bandits)
    return user_bandits[user_id]


def record_feedback(user_id, movie_id, pu, reward):
    """
    Call this once the user actually rates a movie the bandit recommended.
    Updates that user's persisted bandit state so future picks reflect it.
    """
    user_bandits = load_user_bandits()
    if user_id not in user_bandits:
        user_bandits[user_id] = LinUCBBandit(context_dim=len(pu), alpha=1.0)
    user_bandits[user_id].update(movie_id, pu, reward)
    save_user_bandits(user_bandits)


def get_popular_movies(n=POPULAR_POOL_SIZE, min_ratings=20):
    ratings = pd.read_csv(RATINGS_PATH)
    stats = ratings.groupby("movieId")["rating"].agg(["count", "mean"])
    stats = stats[stats["count"] >= min_ratings]
    stats = stats.sort_values(["mean", "count"], ascending=False)
    return stats.head(n).index.tolist()


def fold_in_user(model, rated_movies):
    """
    rated_movies: list of (movieId, rating).
    Estimates a latent vector for a user NOT in the trained model, using
    ridge regression against the item vectors of movies they've rated.
    Movies the trained model has never seen are skipped (no item vector
    exists for them to regress against).
    """
    mu = model.trainset.global_mean
    rows, targets = [], []
    for movie_id, rating in rated_movies:
        try:
            inner_iid = model.trainset.to_inner_iid(movie_id)
        except ValueError:
            continue
        qi = model.qi[inner_iid]
        bi = model.bi[inner_iid]
        rows.append(qi)
        targets.append(rating - mu - bi)

    if not rows:
        return np.zeros(model.qi.shape[1])

    Q = np.array(rows)
    y = np.array(targets)
    reg = 0.1
    d = Q.shape[1]
    pu = np.linalg.solve(Q.T @ Q + reg * np.identity(d), Q.T @ y)
    return pu


def predict_with_folded_in(model, pu, movie_id):
    mu = model.trainset.global_mean
    try:
        inner_iid = model.trainset.to_inner_iid(movie_id)
    except ValueError:
        return mu
    qi = model.qi[inner_iid]
    bi = model.bi[inner_iid]
    return mu + bi + pu @ qi


def get_candidates(model, pu, rated_movie_ids, top_rated_movie_id, k=30):
    all_movie_ids = [model.trainset.to_raw_iid(iid) for iid in model.trainset.all_items()]
    unrated = [m for m in all_movie_ids if m not in rated_movie_ids]

    cf_scores = [(m, predict_with_folded_in(model, pu, m)) for m in unrated]
    cf_scores.sort(key=lambda x: x[1], reverse=True)
    cf_ids = [m for m, _ in cf_scores[:k // 2]]

    content_results = get_similar_movies(top_rated_movie_id, k=k // 2)
    content_ids = [m for m, _ in content_results]
    similarity_lookup = {m: s for m, s in content_results}

    combined = []
    seen = set()
    for m in cf_ids + content_ids:
        if m not in seen:
            seen.add(m)
            combined.append(m)
    return combined, similarity_lookup


def rank(model, pu, candidates, similarity_lookup):
    scored = []
    for movie_id in candidates:
        cf_score = predict_with_folded_in(model, pu, movie_id)
        if movie_id in similarity_lookup:
            content_score = similarity_lookup[movie_id] * 4 + 1
            score = 0.7 * cf_score + 0.3 * content_score
        else:
            score = cf_score
        scored.append((movie_id, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [m for m, _ in scored]


def onboard_and_recommend(user_id, rated_movies):
    """
    rated_movies: list of (movieId, rating) the new user has given so far.
    Returns either a popularity list (not enough ratings yet) or a single
    bandit-picked recommendation (enough context to personalize). The
    bandit picking is persisted per user_id, so it actually learns across
    repeated calls once record_feedback() is called after each pick.
    """
    if len(rated_movies) < ONBOARDING_THRESHOLD:
        return {"mode": "popularity", "movies": get_popular_movies(n=10)}

    model = load_cf_model()
    pu = fold_in_user(model, rated_movies)

    rated_ids = set(m for m, _ in rated_movies)
    top_rated_movie_id = max(rated_movies, key=lambda x: x[1])[0]

    candidates, similarity_lookup = get_candidates(model, pu, rated_ids, top_rated_movie_id)
    ranked = rank(model, pu, candidates, similarity_lookup)
    top_set = ranked[:10]

    bandit = get_bandit_for_user(user_id, context_dim=model.qi.shape[1])
    picked = bandit.select(top_set, pu)

    return {"mode": "personalized", "movie": picked, "candidate_set": top_set, "context": pu}


if __name__ == "__main__":
    movies = pd.read_csv(MOVIES_PATH).set_index("movieId")["title"]
    user_id = 99999

    print("New user, 0 ratings:")
    result = onboard_and_recommend(user_id, [])
    print(f"  mode={result['mode']}")
    for m in result["movies"]:
        print(f"    {movies.loc[m]}")

    popular = get_popular_movies(n=10)
    simulated_ratings = [(popular[0], 5.0), (popular[1], 4.5), (popular[2], 2.0),
                          (popular[3], 4.5), (popular[4], 5.0)]

    print(f"\nSame user after rating {len(simulated_ratings)} movies - 5 recommendation rounds,")
    print("with simulated feedback fed back after each pick:")
    for round_num in range(5):
        result = onboard_and_recommend(user_id, simulated_ratings)
        picked_title = movies.loc[result["movie"]]
        print(f"  round {round_num + 1}: bandit picked {picked_title}")

        # Simulate the user's reaction and feed it back so the bandit
        # actually learns before the next round.
        simulated_reward = np.random.default_rng(round_num).uniform(2.0, 5.0)
        record_feedback(user_id, result["movie"], result["context"], simulated_reward)
        print(f"    (simulated reward: {simulated_reward:.2f})")