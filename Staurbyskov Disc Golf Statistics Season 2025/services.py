"""Services for preprocessing and ranking disc golf season data.

This module provides utility classes for filtering players, preparing hole score
and date data, and ranking players by average round scores. It is intended to
support the analysis pipelines in the main modeling classes.

Classes:
    FilteringService:
        Filters players by minimum number of rounds played.
    DataPreparationService:
        Extracts player names, hole score columns, and centers dates for modeling.
    RankingService:
        Computes average round scores per player and ranks them.

Each class consists of static methods to allow easy integration into analysis
pipelines without requiring instantiation.
"""

# pylint: disable=too-few-public-methods

import pandas as pd
import numpy as np

class FilteringService:
    """Utility class to filter players by minimum rounds played.

    Provides static methods to select only players with sufficient
    participation for analysis pipelines.
    """

    @staticmethod
    def filter_players_by_rounds(
            df: pd.DataFrame, min_rounds: int = 10, col: str = "name"
        ) -> pd.DataFrame:
        """
        Filters players in `df` who have at least `min_rounds` rounds.

        Args:
            df (pd.DataFrame): DataFrame containing player data.
            min_rounds (int): Minimum number of rounds required to keep a player.
            col (str): The name of the column from which to sort players.

        Returns:
            pd.DataFrame: Filtered DataFrame with eligible players only.
        """
        counts = df[col].value_counts()
        eligible_players = counts[counts >= min_rounds].index
        filtered_df = df[df[col].isin(eligible_players)].copy()
        return filtered_df

class DataPreparationService:
    """Utility class for preparing player and hole score data.

    Provides static methods to extract player names, hole score columns,
    and to center date columns for downstream modeling.
    """

    @staticmethod
    def get_player_names(df: pd.DataFrame) -> np.ndarray:
        """
        Extracts player names as a NumPy array.

        Args:
            df (pd.DataFrame): DataFrame with a 'name' column.

        Returns:
            np.ndarray: Array of player names, aligned row-by-row with df.
        """
        return df['name'].values

    @staticmethod
    def get_hole_score(df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts columns corresponding to hole scores from the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing player and hole data 
                (columns with hole data must start with 'hole_').

        Returns:
            pd.DataFrame: DataFrame with only hole score columns.
        """
        hole_df = df[[col for col in df.columns if col.startswith('hole_')]].copy()

        return hole_df

    @staticmethod
    def center_date(df: pd.DataFrame, col: str = "date", unit: str = "week") -> pd.Series:
        """
        Center a datetime column around its mean and return numeric values.

        Args:
            df (pd.DataFrame): DataFrame containing a datetime column.
            col (str): Column name with datetime values.
            unit (str): One of 'day', 'week', or 'month' for scaling.

        Returns:
            pd.Series: Centered numeric values.

        Raises:
            ValueError: If `unit` is not 'day', 'week', or 'month'.
        """
        if unit not in {"day", "week", "month"}:
            raise ValueError(f"Invalid unit '{unit}', must be 'day', 'week', or 'month'.")

        delta = (df[col] - df[col].mean()).dt.days.astype(float)

        if unit == "day":
            return delta
        if unit == "week":
            return delta / 7
        return delta / 30.4375  # average days per month

class RankingService:
    """Utility class for ranking players by average round scores.

    Provides static methods to compute and rank average round scores per player,
    typically along a specified principal component or factor.
    """

    @staticmethod
    def rank_players_by_avg_score(player_names: np.ndarray, round_scores: np.ndarray) -> pd.Series:
        """
        Computes average round score per player along a specified PC and ranks them.

        Args:
            player_names (np.ndarray): Array of player names, aligned row-by-row with round_scores.
            round_scores (np.ndarray): Array of total round scores along a PC.

        Returns:
            pd.Series: Player names as index, average PC scores as values, sorted ascending.
        """
        round_scores_df = pd.DataFrame({
            'name': player_names,
            'round_total_pc': round_scores
        })
        player_avg_by_pc = round_scores_df.groupby('name')['round_total_pc'].mean()
        player_avg_by_pc_ranked = player_avg_by_pc.sort_values()

        return player_avg_by_pc_ranked
