"""
Player entity module for DiscGolfStats Staurbyskov.

Defines the Player class, representing an individual disc golf player,
their recorded rounds, and derived statistics such as average scores,
hole performance, and momentum (autocorrelation of relative scores).
"""

# ---------------- Standard library ----------------
from typing import Dict, List

# ---------------- Third-party ----------------
import numpy as np
import pandas as pd

# ---------------- Entities ----------------
from Entities.round import Round


class Player:
    """Represents a disc golf player and their performance across rounds.

    Attributes:
        name (str): Player's name.
        rounds (List[Round]): List of Round objects for the player.
        nr_rounds (int): Number of rounds played.
        avg_score (float): Average total score across rounds.
        avg_relative_score (float): Average score relative to par.
        sd_score (float): Standard deviation of total scores.
        best_round (Round): Round with the lowest total score.
        worst_round (Round): Round with the highest total score.
        hole_avg (Dict[str, float]): Average score per hole across all rounds.
        momentum (float): Autocorrelation of relative scores between consecutive rounds.
    """

    def __init__(self, name: str, df: pd.DataFrame) -> None:
        """Initializes a Player object with their rounds and computed statistics.

        Args:
            name (str): The player's name.
            df (pd.DataFrame): DataFrame containing round data for the player.

        Side Effects:
            Instantiates Round objects, computes and assigns average scores,
            standard deviation, best/worst rounds, per-hole averages, and momentum.
        """
        self.name = name
        self.rounds = [Round(row) for _, row in df.iterrows()]

        self.nr_rounds = len(self.rounds)
        self.avg_score = np.mean([r.total_score for r in self.rounds])
        self.avg_relative_score = np.mean([r.relative_score for r in self.rounds])
        self.sd_score = np.std([r.total_score for r in self.rounds], ddof=1) if self.nr_rounds > 1 else np.nan
        self.best_round = min(self.rounds, key=lambda r: r.total_score)
        self.worst_round = max(self.rounds, key=lambda r: r.total_score)

        hole_scores = {}
        for r in self.rounds:
            for h, s in r.scores.items():
                hole_scores.setdefault(h, []).append(s)
        self.hole_avg = {h: np.mean(vals) for h, vals in hole_scores.items()}
        self.momentum = 0
        self._compute_momentum()

    def __repr__(self) -> str:
        """Returns a concise string representation of the player.

        Returns:
            str: The player's name and number of rounds.

        Side Effects:
            None
        """
        return f"Player({self.name}, rounds={len(self.rounds)})"

    def _compute_momentum(self) -> None:
        """Computes the player's score momentum.

        Momentum is calculated as the autocorrelation between consecutive
        relative scores. It reflects whether a player tends to improve or
        regress between rounds.

        Side Effects:
            Updates the instance attribute `momentum`.
            Sets NaN if <3 rounds, 0.0 if denominator is zero.

        Returns:
            None
        """
        scores = [r.relative_score for r in self.rounds]
        if len(scores) < 3:
            self.momentum = np.nan
            return

        mu = float(np.mean(scores))
        x: np.ndarray = np.array(scores[:-1], dtype=float) - mu
        y: np.ndarray = np.array(scores[1:], dtype=float) - mu

        numerator: float = float(np.sum(x * y))
        sum_x2: float = float(np.sum(x ** 2))
        sum_y2: float = float(np.sum(y ** 2))
        denominator = np.sqrt(sum_x2 * sum_y2)
        self.momentum = numerator / denominator if denominator != 0 else 0.0
