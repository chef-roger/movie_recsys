"""
Person B owns this file (Phase 6).

Simulates the bandit against held-out ratings and tracks cumulative regret.
"""
import pandas as pd


def simulate(bandit, held_out_ratings: pd.DataFrame, n_rounds: int = 1000):
    """
    TODO (Person B):
    For each round: pick a random user, get candidates, bandit.select(),
    check if held_out_ratings has a real rating for (user, chosen movie) —
    if yes, that's the reward, call bandit.update(). Track regret per round
    (best possible reward in that round minus reward received).
    Returns: list of cumulative regret values for plotting.
    """
    raise NotImplementedError
