"""
Person B owns this file (Phase 5).

Hand-rolled multi-armed / contextual bandit that picks which candidate to
actually show, balancing exploration vs exploitation. Trains/evaluates
against a simulator using held-out ratings as reward.
"""
import numpy as np


class UCBBandit:
    """Simple UCB1 bandit — treats each movie_id as an arm."""

    def __init__(self):
        self.counts = {}   # movie_id -> times shown
        self.values = {}   # movie_id -> running average reward

    def select(self, candidate_movie_ids: list[int]) -> int:
        """
        TODO (Person B):
        1. For any candidate never shown, show it first (counts == 0)
        2. Otherwise compute UCB score per candidate:
           value + sqrt(2 * ln(total_counts) / counts[movie_id])
        3. Return the movie_id with the highest UCB score
        """
        raise NotImplementedError

    def update(self, movie_id: int, reward: float):
        """TODO: update self.counts and self.values (running average) for movie_id."""
        raise NotImplementedError


def choose_with_bandit(user_id: int, candidates: list[int]) -> int:
    """TODO: wraps a per-user UCBBandit instance and calls .select()."""
    raise NotImplementedError
