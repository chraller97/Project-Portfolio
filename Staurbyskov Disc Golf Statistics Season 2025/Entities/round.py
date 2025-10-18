"""
round.py
--------

Defines the `Round` class for representing a single disc golf round played by a player.
Integrates with global constants for course information and par values.
"""

# ---------------- Standard library ----------------
from __future__ import annotations

# ---------------- Third-party ----------------
import numpy as np
import pandas as pd

# ---------------- Project imports ----------------
from Entities.constants import Constants

class Round:
    """Represents a single disc golf round played by a player.

    Class Attributes:
        course_name (str): Name of the course (from Constants).
        par (int): Standard par for each hole (from Constants).
        holes (int): Number of holes on the course (from Constants).
        layout (str): Tee layout used (from Constants).

    Instance Attributes:
        division (str): Player's division.
        date (Any): Date of the round.
        player (str): Player's name.
        card (Any): Raw scorecard data.
        total_score (float): Total score for the round.
        relative_score (float): Score relative to par.
        handicap_relative_score (Optional[float]): Handicap-adjusted relative score.
        handicap_starting_adjustment (Optional[float]): Handicap starting adjustment.
        scores (Dict[str, float]): Dictionary mapping hole identifiers to scores.
        sd (Optional[float]): Standard deviation of hole scores.
        mean_score (Optional[float]): Mean score across holes.
        min_score (Optional[int]): Lowest hole score.
        max_score (Optional[int]): Highest hole score.
        birdies (int): Number of holes scored under par.
        pars (int): Number of holes scored at par.
        bogeys (int): Number of holes scored over par.
    """

    course_name = Constants.COURSE_NAME
    par = Constants.PAR
    holes = Constants.COURSE_HOLES
    layout = Constants.COURSE_LAYOUT

    def __init__(self, row: pd.Series) -> None:
        """Initializes a Round object from a pandas Series representing one round.

        Args:
            row (pd.Series): A pandas Series containing round data for a single player.

        Side Effects:
            Sets multiple attributes on the instance, including statistical summaries
            (mean, SD, min, max) and counts of birdies, pars, and bogeys.
        """
        self.division = row["division"]
        self.date = row["date"]
        self.player = row["name"]
        self.card = row["card"]
        self.total_score = row["round_total_score"]
        self.relative_score = row["round_relative_score"]
        self.handicap_relative_score = row.get("handicap_relative_round_score")
        self.handicap_starting_adjustment = row.get("handicap_starting_score_adjustment")
        self.scores = {}
        for col in row.index:
            if col.startswith("hole_"):
                value = row[col]
                if isinstance(value, pd.Series):
                    value = value.squeeze()
                self.scores[col] = float(value) if pd.notna(value) else np.nan

        values = [float(v) for v in self.scores.values() if pd.notna(v)]
        self.sd = np.std(values, ddof=1) if values else None
        self.mean_score = np.mean(values) if values else None
        self.min_score = min(values) if values else None
        self.max_score = max(values) if values else None
        self.birdies = sum(1 for s in self.scores.values() if s < Constants.PAR)
        self.pars = sum(1 for s in self.scores.values() if s == Constants.PAR)
        self.bogeys = sum(1 for s in self.scores.values() if s > Constants.PAR)

    def __repr__(self) -> str:
        """Returns a concise string representation of the round.

        Returns:
            str: A string showing the player, date, and total score.

        Side Effects:
            None
        """
        return f"Round(player={self.player}, date={self.date}, total={self.total_score})"

    def print_holes(self) -> None:
        """Prints each hole and the corresponding score for the round.

        Side Effects:
            Outputs to standard output (prints hole scores line by line).

        Returns:
            None
        """
        for hole, score in self.scores.items():
            print(f"{hole}: {score}")
