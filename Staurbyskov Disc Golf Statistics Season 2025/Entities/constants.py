"""
constants.py
------------

Defines the `Constants` class used across the DiscGolfStatsStaurbyskov project.

Holds global constants for course information, ranking thresholds, and plotting
defaults used in player, round, and tournament analysis.
"""

# ---------------- Standard library ----------------
from __future__ import annotations

class Constants:
    """Global constants for course info, ranking thresholds, and plotting defaults.

    Attributes:
        COURSE_NAME (str): Name of the disc golf course.
        COURSE_LAYOUT (str): Tee layout used on the course.
        COURSE_HOLES (int): Number of holes on the course.
        PAR (int): Standard par per hole.

        MIN_PLAYER_ROUNDS (int): Minimum number of rounds required for player ranking.
        MIN_TOURNAMENT_PLAYERS (int): Minimum number of players for a tournament to count.
        ROLLING_AVG_WINDOW (int): Window size for rolling average calculations.

        DEFAULT_CMAP (str): Default colormap for plots.
        FIGURE_WIDTH (int): Default width for large figures.
        FIGURE_HEIGHT_MIN (int): Minimum height for heatmaps.
        FIGURE_SIZE_SMALL (tuple[int, int]): Standard small figure size.
        FIGURE_SIZE_MEDIUM (tuple[int, int]): Standard medium figure size.
        FIGURE_SIZE_LARGE (tuple[int, int]): Standard large figure size.
        SCATTER_SCALE (int): Scaling factor for scatter plot marker sizes.
    """

    # Course info
    COURSE_NAME = "Staurbyskov Discgolf"
    COURSE_LAYOUT = "Yellow tees"
    COURSE_HOLES = 14
    PAR = 3

    # Ranking & filtering
    MIN_PLAYER_ROUNDS = 10
    MIN_TOURNAMENT_PLAYERS = 6
    ROLLING_AVG_WINDOW = 5

    # Plotting defaults
    DEFAULT_CMAP = "viridis"
    FIGURE_WIDTH = 10
    FIGURE_HEIGHT_MIN = 4
    FIGURE_SIZE_SMALL = (6, 4)
    FIGURE_SIZE_MEDIUM = (8, 4)
    FIGURE_SIZE_LARGE = (10, 5)
    SCATTER_SCALE = 10
