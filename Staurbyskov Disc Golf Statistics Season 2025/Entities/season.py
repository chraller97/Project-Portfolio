"""
Season module for DiscGolfStats Staurbyskov.

Defines the Season class which represents a disc golf season,
its rounds, players, tournaments, and computed statistics.
"""

# ---------------- Standard library ----------------
from typing import Dict, Optional
import re

# ---------------- Third-party ----------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression

# ---------------- Entities ----------------
from Entities.constants import Constants
from Entities.round import Round
from Entities.player import Player
from Entities.tournament import Tournament

class Season:
    """Represents a disc golf season, containing all rounds, players, and tournaments.

    Instance Attributes:
        raw_data (pd.DataFrame): Original raw DataFrame filtered by season/division.
        data (pd.DataFrame): Cleaned DataFrame excluding DNF results.
        name (str): Name of the season.
        total_rounds (Optional[int]): Total number of rounds in the season.
        total_players (Optional[int]): Total number of players.
        tournament_count (Optional[int]): Number of tournaments in the season.
        avg_score (Optional[float]): Average total score across all rounds.
        sd_score (Optional[float]): Standard deviation of total scores.
        avg_relative_score (Optional[float]): Average relative score.
        best_round (Optional[Round]): Round with the lowest total score.
        worst_round (Optional[Round]): Round with the highest total score.
        hole_avg (Dict[str, float]): Average score per hole.
        hole_sd (Dict[str, float]): Standard deviation per hole.
        rounds (List[Round]): List of all Round objects.
        players (Dict[str, Player]): Mapping of player names to Player objects.
        tournaments (Dict[str, Tournament]): Mapping of dates to Tournament objects.
    """

    def __init__(self, df: pd.DataFrame, season_name: str) -> None:
        """Initializes a Season object.

        Args:
            df (pd.DataFrame): Full dataset containing all rounds.
            season_name (str): Division/season name to filter the dataset.

        Side Effects:
            Constructs Player and Tournament objects.
            Computes season-level and hole-level metrics.
        """
        self.raw_data = df[df["division"] == season_name]
        self.data = self.raw_data.loc[self.raw_data["position"] != "DNF"].copy()
        self.data["name"] = self.data["name"].str.strip()
        self.name = season_name

        self.total_rounds: Optional[int] = None
        self.total_players: Optional[int] = None
        self.tournament_count: Optional[int] = None
        self.avg_score: Optional[float] = None
        self.sd_score: Optional[float] = None
        self.avg_relative_score: Optional[float] = None
        self.best_round: Optional[Round] = None
        self.worst_round: Optional[Round] = None

        self.hole_avg: Dict[str, Optional[float]] = {}
        self.hole_sd: Dict[str, Optional[float]] = {}

        self.rounds = [Round(row) for _, row in self.data.iterrows()]
        self.players: Dict[str, Player] = {}
        self.tournaments: Dict[str, Tournament] = {}
        self._construct_players_and_tournaments()
        self._compute_season_metrics()
        self._compute_hole_metrics()

    def _construct_players_and_tournaments(self) -> None:
        """Builds Player and Tournament objects from the dataset.

        Side Effects:
            Populates `players` and `tournaments` dictionaries with Player and Tournament objects.
        """
        for player, player_data in self.data.groupby("name"):
            self.players[player] = Player(player, player_data)

        for date, date_data in self.data.groupby("date"):
            tournament: Tournament = Tournament(date)
            for _, row in date_data.iterrows():
                player = self.players[str(row["name"])]
                player_round = Round(row)
                tournament.add_round(player, player_round)
            tournament.compute_tournament_metrics()
            self.tournaments[date] = tournament

    def _compute_season_metrics(self) -> None:
        """Computes overall season statistics including averages, standard deviations, and best/worst rounds.

        Side Effects:
            Updates `total_rounds`, `total_players`, `tournament_count`,
            `avg_score`, `sd_score`, `avg_relative_score`, `best_round`, and `worst_round`.
        """
        all_rounds = [r for player in self.players.values() for r in player.rounds]
        self.total_rounds = len(all_rounds)
        self.total_players = len(self.players)
        self.tournament_count = len(self.tournaments)
        self.avg_score = np.mean([r.total_score for r in all_rounds])
        self.sd_score = np.std([r.total_score for r in all_rounds], ddof=1)
        self.avg_relative_score = np.mean([r.relative_score for r in all_rounds])
        self.best_round = min(all_rounds, key=lambda r: r.total_score)
        self.worst_round = max(all_rounds, key=lambda r: r.total_score)

    def _compute_hole_metrics(self) -> None:
        """Computes average and standard deviation per hole.

        Side Effects:
            Updates `hole_avg` and `hole_sd` dictionaries.
        """
        hole_cols = [col for col in self.data.columns if col.startswith("hole_")]
        for hole in hole_cols:
            scores = self.data[hole].dropna().astype(int)
            if not scores.empty:
                self.hole_avg[hole] = float(np.mean(scores))
                self.hole_sd[hole] = float(np.std(scores, ddof=1))
            else:
                self.hole_avg[hole] = None
                self.hole_sd[hole] = None

    def rank_players(self, metric: str, min_rounds: int = Constants.MIN_PLAYER_ROUNDS) -> list:
        """Ranks players based on a specified metric.

        Args:
            metric (str): Player attribute to rank by (e.g., 'avg_score').
            min_rounds (int, optional): Minimum number of rounds required to include a player.

        Returns:
            list: List of tuples containing (rank, player name, metric value).

        Side Effects:
            None
        """
        eligible = [p for p in self.players.values() if p.nr_rounds >= min_rounds]
        sorted_players = sorted(eligible, key=lambda p: getattr(p, metric))
        return [(i + 1, p.name, round(getattr(p, metric), 3)) for i, p in enumerate(sorted_players)]

    def rank_tournaments(self, metric: str, min_players: int = Constants.MIN_TOURNAMENT_PLAYERS) -> pd.DataFrame:
        """Ranks tournaments by a specified metric.

        Args:
            metric (str): Tournament attribute to rank by (e.g., 'avg_score').
            min_players (int, optional): Minimum number of players required to include a tournament.

        Returns:
            pd.DataFrame: Ranked tournament table with columns ['Dato', metric, '#Spillere'].

        Side Effects:
            None
        """
        eligible = [t for t in self.tournaments.values() if t.nr_players >= min_players]
        sorted_tournaments = sorted(eligible, key=lambda t: getattr(t, metric))
        df = pd.DataFrame(
            [(t.date, round(getattr(t, metric), 3), t.nr_players) for t in sorted_tournaments],
            columns=['Dato', metric, '#Spillere']
        )
        df.index = range(1, len(df) + 1)
        df.index.name = 'Pos.'
        return df

    def plot_avg_vs_sd(self, cmap=Constants.DEFAULT_CMAP, min_rounds: int = Constants.MIN_PLAYER_ROUNDS) -> plt.Figure | None:
        """Plots a scatter of average score vs. standard deviation for eligible players.

        Args:
            cmap (str, optional): Colormap to use for points.
            min_rounds (str): Minimum number of rounds required to include a player.

        Returns:
            plt.Figure | None: Matplotlib figure of the scatter plot, or None if no eligible players.

        Side Effects:
            Creates a matplotlib figure.
        """
        avg_rank = self.rank_players("avg_score", min_rounds=min_rounds)
        sd_rank = self.rank_players("sd_score", min_rounds=min_rounds)

        names = [p[1] for p in avg_rank]
        avg_values = [p[2] for p in avg_rank]
        sd_values = [next(sd[2] for sd in sd_rank if sd[1] == name) for name in names]

        fig, ax = plt.subplots(figsize=Constants.FIGURE_SIZE_SMALL)
        scatter = ax.scatter(avg_values, sd_values, c=avg_values, cmap=cmap)
        for i, name in enumerate(names):
            ax.text(avg_values[i], sd_values[i], name, fontsize=8)

        ax.set_xlabel("Gennemsnitlig score")
        ax.set_ylabel("Standardafvigelse")
        ax.grid(axis="both", linestyle="--", alpha=0.5)
        plt.colorbar(scatter, ax=ax, label="Gennemsnitlig score", fraction=0.046, pad=0.15)
        plt.tight_layout()
        return fig

    def summary_lines(self) -> list[str]:
        """Generates summary lines of season metrics.

        Returns:
            list[str]: List of formatted strings summarizing the season.

        Side Effects:
            None
        """
        return [
            f"Sæson: {self.name}",
            f"Antal spillere: {self.total_players}",
            f"Antal runder: {self.total_rounds}",
            f"Turneringer: {self.tournament_count}",
            f"Gennemsnitlig score: {self.avg_score:.2f} ({self.avg_relative_score:+.2f})",
            f"Score SD: {self.sd_score:.2f}",
            f"Bedste runde: {self.best_round.player} med {self.best_round.relative_score} den {self.best_round.date}",
        ]

    def season_summary_stats(self) -> str:
        """Generates a detailed summary of season and hole statistics.

        Returns:
            str: Multi-line string with season stats and per-hole averages and standard deviations.

        Side Effects:
            None
        """
        lines = self.summary_lines() + ["", "Hole stats:"]
        for hole, avg in self.hole_avg.items():
            sd = self.hole_sd.get(hole, float("nan"))
            lines.append(f"{hole}: avg={avg:.2f}  sd={sd:.2f}")
        return "\n".join(lines)

    def get_hole_rankings(self, n: int = Constants.COURSE_HOLES / 2) -> tuple[list[tuple], list[tuple]]:
        """Returns the hardest and easiest holes based on average score.

        Args:
            n (int, optional): Number of holes to include in hardest/easiest lists.

        Returns:
            tuple[list[tuple], list[tuple]]: Two lists of tuples (hole name, avg, sd), first is hardest, second is easiest.

        Side Effects:
            None
        """
        n = int(n)
        hole_avgs = [(h, self.hole_avg[h], self.hole_sd[h]) for h in self.hole_avg.keys() if
                     self.hole_avg[h] is not None]
        hole_avgs.sort(key=lambda x: -x[1])
        hardest = hole_avgs[:n]
        easiest = hole_avgs[-n:]
        return hardest, easiest

    def get_hole_stats(self) -> pd.DataFrame:
        """Creates a DataFrame of holes with their average score and standard deviation.

        Returns:
            pd.DataFrame: Columns are ['hole', 'avg', 'sd'], sorted by hole number.

        Side Effects:
            None
        """
        holes = sorted(self.hole_avg.keys(), key=lambda h: int(re.search(r"\d+", h).group()))
        df = pd.DataFrame({
            "hole": holes,
            "avg": [self.hole_avg[h] for h in holes],
            "sd": [self.hole_sd[h] for h in holes],
        })
        return df

    def plot_hole_stats(self, color_palette: dict) -> plt.Figure:
        """Plots hole averages with error bars representing standard deviations.

        Args:
            color_palette (dict): Dictionary with colors for plot styling.

        Returns:
            plt.Figure: Matplotlib figure showing hole statistics.

        Side Effects:
            Creates and returns a matplotlib figure.
        """
        df = self.get_hole_stats()
        fig, ax = plt.subplots(figsize=Constants.FIGURE_SIZE_LARGE)
        ax.errorbar(
            df["hole"],
            df["avg"],
            yerr=df["sd"],
            capsize=5,
            color=color_palette["main"],
            fmt="o",
            ecolor=color_palette["highlight"],
            elinewidth=1.5,
        )
        ax.set_ylabel("Gennemsnit", color=color_palette["black"])
        plt.xticks(rotation=45)
        ax.grid(axis="y", linestyle="--", color=color_palette["table_bg"], alpha=0.5)
        plt.tight_layout()
        return fig

    def get_round_scores(self) -> np.ndarray:
        """Returns all relative round scores from the season.

        Returns:
            np.ndarray: Array of relative round scores.

        Side Effects:
            None
        """
        return self.data["round_relative_score"].dropna().to_numpy()

    def plot_score_histogram(self, color_palette: dict) -> plt.Figure:
        """Plots histogram and kernel density estimate of relative round scores.

        Args:
            color_palette (dict): Dictionary with color styling.

        Returns:
            plt.Figure: Matplotlib figure containing histogram and KDE line.

        Side Effects:
            Creates a matplotlib figure.
        """
        scores = self.get_round_scores()
        min_score, max_score = int(scores.min()), int(scores.max())
        bins_range = range(min_score - 1, max_score + 2)
        fig, ax = plt.subplots(figsize=Constants.FIGURE_SIZE_MEDIUM)
        ax.hist(
            scores,
            bins=bins_range,
            color=color_palette["highlight"],
            edgecolor=color_palette["accent"],
            align="left",
        )
        kde = gaussian_kde(scores, bw_method="silverman")
        x = np.linspace(min_score - 1, max_score + 1, 500)
        ax.plot(x, kde(x) * len(scores), color=color_palette["main"], linewidth=2)
        ax.set_xlabel("Relativ runde-score", color=color_palette["black"])
        ax.set_ylabel("Frekvens", color=color_palette["black"])
        ax.grid(axis="y", alpha=0.75)
        plt.tight_layout()
        return fig

    def plot_player_hole_heatmap(self, min_rounds: int = Constants.MIN_PLAYER_ROUNDS, cmap=None) -> plt.Figure | None:
        """Plots heatmap of average player performance per hole relative to par.

        Args:
            min_rounds (int, optional): Minimum number of rounds required per player.
            cmap (str, optional): Colormap for the heatmap.

        Returns:
            plt.Figure | None: Matplotlib heatmap figure, or None if no eligible players.

        Side Effects:
            Creates a seaborn heatmap figure.
        """
        players: list[Player] = [p for p in self.players.values() if p.nr_rounds >= min_rounds]
        if not players:
            return None

        data = {p.name: {h: avg - Constants.PAR for h, avg in p.hole_avg.items()} for p in players}
        df = pd.DataFrame.from_dict(data, orient="index")
        if df.empty:
            return None

        hole_cols = sorted(df.columns, key=lambda x: float(x.split("_")[1].replace("½", ".5")))
        df = df[hole_cols]

        if cmap is None:
            cmap = Constants.DEFAULT_CMAP

        height = max(4.0, len(players) * 0.3)
        fig, ax = plt.subplots(figsize=(Constants.FIGURE_WIDTH, height))
        sns.heatmap(df, cmap=cmap, annot=True, fmt=".2f", cbar=True, ax=ax)
        ax.set_title("Spiller vs. Hul: Avg score i forhold til par", color="black")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return fig

    def plot_tournament_scores_over_time(self, color_palette: dict,
                                         min_players: int = Constants.MIN_TOURNAMENT_PLAYERS) -> plt.Figure | None:
        """Plots average tournament scores over time with rolling average.

        Args:
            color_palette (dict): Dictionary with plot colors.
            min_players (int, optional): Minimum number of players required for a tournament to be included.

        Returns:
            plt.Figure | None: Matplotlib figure, or None if insufficient data.

        Side Effects:
            Creates and returns a matplotlib figure.
        """
        df = self.rank_tournaments("avg_score", min_players)
        if df.empty:
            return None

        df_plot = df.sort_values("Dato").reset_index(drop=True)
        df_plot["Dato"] = pd.to_datetime(df_plot["Dato"])
        df_plot["rolling_avg"] = df_plot["avg_score"].rolling(window=Constants.ROLLING_AVG_WINDOW, center=False).mean()
        season_avg = df_plot["avg_score"].mean()

        fig, ax = plt.subplots(figsize=Constants.FIGURE_SIZE_MEDIUM)
        ax.plot(df_plot["Dato"], df_plot["avg_score"], linestyle="-", color=color_palette["main"], label="Avg")
        ax.scatter(df_plot["Dato"], df_plot["avg_score"], s=df_plot["#Spillere"] * Constants.SCATTER_SCALE,
                   color=color_palette["main"], alpha=0.7)
        ax.plot(df_plot["Dato"], df_plot["rolling_avg"], color=color_palette["highlight"], linewidth=1,
                label="Moving Avg")
        ax.axhline(season_avg, color=color_palette["table_bg"], linestyle="--", linewidth=1, label="Sæson Avg")
        ax.xaxis.set_major_formatter(DateFormatter("%d %b"))
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Gennemsnitlig score", color=color_palette["black"])
        plt.legend(frameon=False)
        plt.tight_layout()
        return fig

    def plot_players_relative_scores(self, player_names: list[str]) -> plt.Figure:
        """Plots relative scores for specified players over time with trend lines.

        Args:
            player_names (list[str]): List of player names to include.

        Returns:
            plt.Figure: Matplotlib figure of relative scores over time.

        Side Effects:
            Creates a matplotlib line plot.
        """
        fig, ax = plt.subplots(figsize=Constants.FIGURE_SIZE_LARGE)
        colors = plt.cm.get_cmap("tab10", len(player_names))

        for i, name in enumerate(player_names):
            if name not in self.players:
                continue
            player = self.players[name]
            if not player.rounds:
                continue

            dates = pd.to_datetime([r.date for r in player.rounds])
            rel_scores = np.array([r.relative_score for r in player.rounds])
            weeks_raw = np.array([(d - dates.min()).days / 7 for d in dates])
            weeks_centered = (weeks_raw - weeks_raw.mean()).reshape(-1, 1)

            model = LinearRegression()
            model.fit(weeks_centered, rel_scores)
            slope_per_week: float = float(model.coef_[0])

            color = colors(i)
            ax.plot(dates, rel_scores, marker='o', linestyle='-', label=f"{name} (slope={slope_per_week:.2f}/wk)",
                    color=color)
            y_pred = model.predict(weeks_centered)
            ax.plot(dates, y_pred, linestyle='--', color=color)

        ax.set_title("Relative Score per Round")
        ax.set_xlabel("Date")
        ax.set_ylabel("Relative Score")
        ax.grid(True)
        ax.legend()
        fig.autofmt_xdate()
        plt.tight_layout()
        return fig