"""
Cold-start test for the bandit.

Adds a handful of brand-new movies (zero rating history, never seen by CF
or content model at training time) into the catalog, then runs repeated
simulation rounds to check whether LinUCB's confidence bonus actually
gives these unseen arms a fair chance to be explored - and whether its
reward estimate for them converges toward a "true" quality we define
ourselves (since real ground truth can't exist for a movie we invented).

Run this AFTER cf_model and content_model have already been trained once
on the real catalog - it adds to that catalog, it doesn't replace it.

Usage: python -m bandit.cold_start_test
"""
import numpy as np
import pandas as pd

from bandit.bandit import LinUCBBandit, get_user_context, _load_cf_model
import content_model.build_embeddings as content_model

MOVIES_PATH = "data/ml-latest-small/movies.csv"

# Synthetic new movies: zero ratings, but we secretly know their "true"
# quality for simulation purposes. A real system would never have this -
# we're using it purely to check whether the bandit's estimate converges
# toward the right answer as it gets more (simulated) exposure.
NEW_MOVIES = [
    {"movieId": 900001, "title": "The Silent Horizon (2026)", "genres": "Drama|Sci-Fi", "true_reward": 4.7},
    {"movieId": 900002, "title": "Neon Static (2026)", "genres": "Action|Thriller", "true_reward": 2.0},
    {"movieId": 900003, "title": "Last Light Diner (2026)", "genres": "Drama|Romance", "true_reward": 4.2},
]


def inject_new_movies():
    """Appends the synthetic movies to movies.csv, then rebuilds content
    embeddings so they're included in the FAISS index."""
    movies = pd.read_csv(MOVIES_PATH)

    already_present = movies["movieId"].isin([m["movieId"] for m in NEW_MOVIES])
    if already_present.any():
        print("New movies already present, skipping injection.")
    else:
        new_rows = pd.DataFrame(
            [{"movieId": m["movieId"], "title": m["title"], "genres": m["genres"]} for m in NEW_MOVIES]
        )
        movies = pd.concat([movies, new_rows], ignore_index=True)
        movies.to_csv(MOVIES_PATH, index=False)
        print(f"Added {len(NEW_MOVIES)} new movies to {MOVIES_PATH}")

    print("Rebuilding content embeddings to include the new movies...")
    content_model.build()


def run_cold_start_simulation(n_rounds=300, pool_size=20):
    """
    Mixes a fixed pool of established movies with the brand-new ones,
    runs repeated rounds, and tracks how often each new movie gets
    selected plus how the bandit's estimate for it evolves.
    """
    ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
    model = _load_cf_model()
    context_dim = model.pu.shape[1]
    bandit = LinUCBBandit(context_dim=context_dim, alpha=1.0)

    established_pool = ratings["movieId"].drop_duplicates().sample(pool_size, random_state=42).tolist()
    new_movie_ids = [m["movieId"] for m in NEW_MOVIES]
    true_rewards = {m["movieId"]: m["true_reward"] for m in NEW_MOVIES}
    full_pool = established_pool + new_movie_ids

    pool_ratings = ratings[ratings["movieId"].isin(established_pool)]

    selection_counts = {mid: 0 for mid in full_pool}

    rng = np.random.default_rng(7)
    for i in range(n_rounds):
        user_id = int(pool_ratings.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]["userId"])
        context = get_user_context(user_id)

        chosen = bandit.select(full_pool, context)
        selection_counts[chosen] += 1

        if chosen in new_movie_ids:
            # We defined this reward ourselves - this is the simulation's
            # stand-in for "the user actually watched and rated it."
            reward = true_rewards[chosen] + rng.normal(0, 0.2)  # small noise
            bandit.update(chosen, context, reward)
        else:
            user_rating_row = ratings[(ratings["userId"] == user_id) & (ratings["movieId"] == chosen)]
            if not user_rating_row.empty:
                reward = float(user_rating_row.iloc[0]["rating"])
                bandit.update(chosen, context, reward)
            # else: no ground truth for this (user, movie) pair, skip

    return bandit, selection_counts, true_rewards


if __name__ == "__main__":
    inject_new_movies()

    print("\nRunning cold-start simulation...")
    bandit, counts, true_rewards = run_cold_start_simulation()

    print("\nSelection counts for the new (zero-history) movies:")
    for m in NEW_MOVIES:
        mid = m["movieId"]
        print(f"  {m['title']:35s} selected {counts[mid]:3d} times  (true quality: {m['true_reward']})")

    established_selected = sum(v for k, v in counts.items() if k not in true_rewards)
    new_selected = sum(v for k, v in counts.items() if k in true_rewards)
    print(f"\nTotal: {established_selected} picks went to established movies, "
          f"{new_selected} picks went to the 3 brand-new movies "
          f"(out of {established_selected + new_selected} rounds)")