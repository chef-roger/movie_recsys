"""
Person B owns this file (Phase 6).

Simulates the bandit against held-out ratings and tracks cumulative regret.
This is the formal version of the hit-rate smoke test in bandit.py's
__main__ block - same idea, but measures REGRET (how much worse than the
best possible choice) instead of just a hit/miss percentage, and plots
how it evolves over time.

Usage: python -m evaluation.eval_bandit
"""
import numpy as np
import pandas as pd

from bandit.bandit import LinUCBBandit, get_user_context, _load_cf_model

RATINGS_PATH = "data/ml-latest-small/ratings.csv"


def simulate(pool_size=30, n_rounds=1000, seed=42):
    """
    Fixes a pool of movies, runs repeated rounds, and tracks regret per
    round: (best possible reward available this round) - (reward actually
    received). Only rounds where the bandit's chosen movie has a real
    logged rating for that user count toward the curve - we have no way
    to know the "true" reward otherwise (the honest limitation we've
    discussed: no impression/skip log exists in this dataset).

    Returns a list of cumulative regret values, one per resolved round -
    this is what you plot to see whether the bandit is actually learning
    (a learning bandit's curve should flatten out over time, not keep
    climbing linearly).
    """
    ratings = pd.read_csv(RATINGS_PATH)
    model = _load_cf_model()
    context_dim = model.pu.shape[1]
    bandit = LinUCBBandit(context_dim=context_dim, alpha=1.0)

    pool = ratings["movieId"].drop_duplicates().sample(pool_size, random_state=seed).tolist()
    pool_ratings = ratings[ratings["movieId"].isin(pool)]

    rng = np.random.default_rng(seed)
    cumulative_regret = []
    running_total = 0.0

    for i in range(n_rounds):
        user_id = int(pool_ratings.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]["userId"])
        user_pool_ratings = pool_ratings[pool_ratings["userId"] == user_id]
        if user_pool_ratings.empty:
            continue

        # "Best possible" = the highest real rating this user gave to
        # anything in the pool - this is the ground-truth optimal choice
        # we compare the bandit's actual choice against.
        best_possible = user_pool_ratings["rating"].max()

        context = get_user_context(user_id)
        chosen = bandit.select(pool, context)

        chosen_rating_row = user_pool_ratings[user_pool_ratings["movieId"] == chosen]
        if chosen_rating_row.empty:
            continue  # no ground truth for this choice, can't score this round

        reward = float(chosen_rating_row.iloc[0]["rating"])
        bandit.update(chosen, context, reward)

        regret = best_possible - reward
        running_total += regret
        cumulative_regret.append(running_total)

    return cumulative_regret


def plot_regret(cumulative_regret, out_path="data/bandit_regret.png"):
    import matplotlib
    matplotlib.use("Agg")  # no display needed, just save to file
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(cumulative_regret)
    plt.xlabel("Round")
    plt.ylabel("Cumulative regret")
    plt.title("LinUCB cumulative regret over time")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved regret plot -> {out_path}")


if __name__ == "__main__":
    cumulative_regret = simulate(pool_size=15, n_rounds=5000)

    print(f"Resolved rounds: {len(cumulative_regret)}")
    print(f"Final cumulative regret: {cumulative_regret[-1]:.2f}")
    print(f"Average regret per round: {cumulative_regret[-1] / len(cumulative_regret):.4f}")

    # A learning bandit's average regret per round should trend DOWN over
    # time - check the first half vs second half of the curve as a rough
    # signal of whether it's actually improving.
    midpoint = len(cumulative_regret) // 2
    first_half_avg = cumulative_regret[midpoint - 1] / midpoint
    second_half_total = cumulative_regret[-1] - cumulative_regret[midpoint - 1]
    second_half_avg = second_half_total / (len(cumulative_regret) - midpoint)
    print(f"\nAvg regret/round, first half:  {first_half_avg:.4f}")
    print(f"Avg regret/round, second half: {second_half_avg:.4f}")
    print("(second half should be LOWER than first half if the bandit is learning)")

    plot_regret(cumulative_regret)