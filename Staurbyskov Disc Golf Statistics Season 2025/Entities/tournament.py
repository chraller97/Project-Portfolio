"""
Tournament module for DiscGolfStatsStarubyskov.

Defines the Tournament class which represents a disc golf tournament,
its rounds, players, and computed statistics.
"""

# ---------------- Standard library ----------------
from typing import Dict, List, Optional, Any

# ---------------- Third-party ----------------
import numpy as np

# ---------------- Entities ----------------
from Entities.round import Round
from Entities.player import Player


class Tournament:
    """Represents a disc golf tournament, including rounds, players, and statistics.

    Attributes:
        date (str): Tournament date.
        rounds (Dict[str, Round]): Mapping of player names to their Round objects.
        players (Dict[str, Player]): Mapping of player names to Player objects.
        nr_players (int): Number of players in the tournament.
        avg_score (Optional[float]): Average total score across all rounds.
        sd_score (Optional[float]): Standard deviation of total scores.
        best_round (Optional[Round]): Round object with the lowest total score.
        worst_round (Optional[Round]): Round object with the highest total score.
        cards (Dict[Any, List[str]]): Mapping of card/group identifiers to player names.
        players_played_with (Dict[str, List[str]]): Players that each player played with on the same card.
    """

    def __init__(self, date: str) -> None:
        """Initializes a Tournament object for a given date.

        Args:
            date (str): The date of the tournament.

        Side Effects:
            Creates empty dictionaries for rounds, players, cards, and co-players.
        """
        self.date = date
        self.rounds: Dict[str, Round] = {}
        self.players: Dict[str, Player] = {}
        self.nr_players = 0
        self.avg_score: Optional[float] = None
        self.sd_score: Optional[float] = None
        self.best_round: Optional[Round] = None
        self.worst_round: Optional[Round] = None
        self.cards: Dict[Any, list[str]] = {}
        self.players_played_with: Dict[str, list[str]] = {}

    def add_round(self, player: Player, round: Round) -> None:
        """Adds a round and associated player to the tournament.

        Args:
            player (Player): The player object.
            round (Round): The round played by the player.

        Side Effects:
            Updates the `rounds` and `players` dictionaries with the new data.
        """
        self.rounds[player.name] = round
        self.players[player.name] = player

    def compute_tournament_metrics(self) -> None:
        """Computes tournament metrics including averages, standard deviations,
        best/worst rounds, card groupings, and co-players.

        Side Effects:
            Updates `nr_players`, `avg_score`, `sd_score`, `best_round`, `worst_round`,
            `cards`, and `players_played_with`.
        """
        if not self.rounds:
            return

        self.nr_players = len(self.players)
        scores = [r.total_score for r in self.rounds.values()]
        self.avg_score = np.mean(scores)
        self.sd_score = np.std(scores, ddof=1)
        self.best_round = min(self.rounds.values(), key=lambda r: r.total_score)
        self.worst_round = max(self.rounds.values(), key=lambda r: r.total_score)

        for r in self.rounds.values():
            if r.card is not None:
                self.cards.setdefault(r.card, []).append(r.player)

        for player_name, round in self.rounds.items():
            if round.card is not None:
                co_players = [p for p in self.cards[round.card] if p != player_name]
                self.players_played_with[player_name] = co_players
            else:
                self.players_played_with[player_name] = []

    def rank_rounds(self, metric: str) -> list[tuple[int, str, float]]:
        """Ranks rounds by a specified metric.

        Args:
            metric (str): Attribute of Round objects to rank by (e.g., 'total_score').

        Returns:
            list[tuple[int, str, float]]: List of tuples containing
                (rank, player name, metric value).

        Side Effects:
            None
        """
        sorted_rounds = sorted(self.rounds.values(), key=lambda r: getattr(r, metric))
        return [(i + 1, r.player, getattr(r, metric)) for i, r in enumerate(sorted_rounds)]
