"""Models for analyzing disc golf season data.

This module provides classes to perform various statistical analyzes on player and round data
from a Season object. It includes functionality for mixed effects modeling, and player similarity metrics.

Classes:
    MixedEffectsModel: 
        Fits mixed effects models with random effects for players and co-players,
        supports flexible random effect formulas, and provides fixed/random effect summaries.
    SimilarityModel: 
        Computes similarity between players using mean-vector or distribution-based methods,
        supports cosine, Minkowski, and Chebyshev metrics, and visualizes similarity matrices.

Each class handles filtering players by minimum rounds, preparing hole score data, 
and storing results in attributes for downstream analysis. Visualization methods return 
matplotlib Figure objects for integration with reporting pipelines.
"""

from __future__ import annotations  # future imports first

# Standard library
from typing import Dict, List, Optional

# Third-party imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
import statsmodels.formula.api as smf

# Local application imports
from Entities.season import Season
import services as svcs

class MixedEffectsModel:
    """Fit and analyze mixed effects models for player round scores.

    This class provides a pipeline to prepare raw season data, filter players,
    construct co-player indicators, build formulas, fit mixed effects models,
    and extract fixed and random effects.

    Attributes:
        season (Season): 
            Season object with raw player data.
        filtered_df (Optional[pd.DataFrame]): 
            Filtered DataFrame after preprocessing.
        player_map (Optional[Dict[str, str]]): 
            Mapping from original co-player names to safe names.
        predictor_cols (Optional[List[str]]): 
            Column names of co-player indicators.
        group_col (str): 
            Column used for grouping in mixed model, defaults to 'primary'.
        player_name_map (Optional[Dict[str, str]]): 
            Reverse mapping of safe names to original player names.
        formula (Optional[str]): 
            Mixed model formula.
        reml (Optional[bool]): 
            Whether to fit using REML.
        min_rounds (Optional[int]): 
            Minimum rounds for filtering players.
        re_formula_mode (Optional[str]): 
            Specifies which random effects formula to use.
        model: 
            MixedLM object (not fitted yet).
        model_fit: 
            Fitted mixed model results object.
    """

    def __init__(self, season: Season):
        """Initialize with season data and internal attributes.

        Side Effects:
            Initializes all internal attributes to None or defaults.
        """
        self.season: Season = season
        self.filtered_df: Optional[pd.DataFrame] = None
        self.player_map: Optional[Dict[str, str]] = None
        self.predictor_cols: Optional[List[str]] = None
        self.group_col: str = "primary"
        self.player_name_map: Optional[Dict[str, str]] = None
        self.formula: Optional[str] = None
        self.reml: Optional[bool] = None
        self.min_rounds: Optional[int] = None
        self.re_formula_mode: Optional[str] = None
        self.model = None
        self.model_fit = None

    def _prepare_raw_df(self) -> None:
        """Prepare raw DataFrame with necessary columns and rename player column to 'primary'.

        Side Effects:
            Sets self.filtered_df to a cleaned DataFrame.
        """
        df = self.season.data.copy()
        df = df.loc[df['name'].notna(), ['date', 'name', 'card', 'round_total_score']].copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.rename(columns={'name': 'primary'})
        self.filtered_df = df

    def _filter(self) -> None:
        """Filter players by minimum rounds.

        Side Effects:
            Updates self.filtered_df with only players meeting min_rounds.

        Raises:
            AttributeError: If self.filtered_df or self.min_rounds is not set.
        """
        if self.filtered_df is None:
            raise AttributeError("filtered_df must be set before filtering")
        if self.min_rounds is None:
            raise AttributeError("min_rounds must be set before filtering")
        self.filtered_df = svcs.FilteringService.filter_players_by_rounds(
            self.filtered_df, self.min_rounds, self.group_col
        )

    def _add_centered_date_col(self) -> None:
        """Add a centered date column for modeling purposes.

        Side Effects:
            Adds column 'date_centered' to self.filtered_df.

        Raises:
            AttributeError: If self.filtered_df is None.
        """
        if self.filtered_df is None:
            raise AttributeError("filtered_df must be set before adding centered date column")
        self.filtered_df['date_centered'] = svcs.DataPreparationService.center_date(
            self.filtered_df, col="date", unit="week"
        )

    def _create_coplayer_indicators(self) -> None:
        """Add binary indicators for presence of each co-player in the round.

        Side Effects:
            Adds multiple 'with_*' columns to self.filtered_df.
            Updates self.predictor_cols with columns having >1 unique value.

        Raises:
            AttributeError: If self.filtered_df is None.
        """
        if self.filtered_df is None:
            raise AttributeError("filtered_df must be set before creating co-player indicators")

        df = self.filtered_df.copy()
        eligible_players = df['primary'].unique()
        presence_df = df[['primary', 'date', 'card']].copy()

        for coplayer in eligible_players:
            col = f"with_{coplayer}"
            df[col] = df.apply(
                lambda r, cp=coplayer: int(
                    (
                            (presence_df['primary'] == cp) &
                            (presence_df['date'] == r['date']) &
                            (presence_df['card'] == r['card'])
                    ).any() and r['primary'] != cp
                ),
                axis=1
            )
        self.filtered_df = df
        self.predictor_cols = [c for c in df.columns if c.startswith('with_') and df[c].nunique()]

    def _map_safe_names(self) -> None:
        """Map co-player columns to safe names for modeling.

        Side Effects:
            Updates self.player_map and self.player_name_map.
            Renames columns in self.filtered_df.
            Updates self.predictor_cols with safe names.

        Raises:
            AttributeError: If self.predictor_cols is None.
        """
        if self.predictor_cols is None:
            raise AttributeError("predictor_cols must be set before mapping names")
        self.player_map = {
            col.replace("with_", ""): f"P{i + 1}"
            for i, col in enumerate(self.predictor_cols.copy())
        }
        rename_map = {f"with_{k}": v for k, v in self.player_map.items()}
        self.player_name_map = rename_map
        self.filtered_df.rename(columns=rename_map, inplace=True)
        self.predictor_cols = list(rename_map.values())

    def _build_formula(self, several_predictors: bool = True) -> None:
        """Build the mixed model formula including co-player indicators.

        Side Effects:
            Sets self.formula.
        """
        if several_predictors:
            self.formula = "round_total_score ~ date_centered + " + " + ".join(self.predictor_cols)
        else:
            self.formula = "round_total_score ~ date_centered"

    def _fit_mixed_model(self) -> None:
        """Fit a mixed effects model with optional random slopes/intercepts.

        Side Effects:
            Sets self.model and self.model_fit.

        Raises:
            AttributeError: If self.formula or self.filtered_df is None.
            ValueError: If self.re_formula_mode is invalid.
        """
        # Determine re_formula based on mode
        mode_map = {
            "none": None,
            "date": "~date_centered",
            "predictors": ("~" + " + ".join(self.predictor_cols) if self.predictor_cols else None),
            "both": (
                    "~date_centered"
                    + (" + " + " + ".join(self.predictor_cols) if self.predictor_cols else "")
            ),
        }
        if self.re_formula_mode not in mode_map:
            raise ValueError(f"Invalid re_formula_mode: {self.re_formula_mode}")
        re_formula = mode_map[self.re_formula_mode]

        # Build MixedLM
        self.model = smf.mixedlm(
            self.formula,
            data=self.filtered_df,
            groups=self.filtered_df[self.group_col],
            re_formula=re_formula,
            use_sparse=True,
        )

        # Fit model (let statsmodels handle initialization)
        self.model_fit = self.model.fit(
            reml=self.reml,
            method="nm",
            maxiter=50000,
            ftol=1e-12
        )

    def analyze(
            self,
            min_rounds: int = 10,
            reml: bool = True,
            re_formula_mode: str = "date",
            several_predictors: bool = True,
    ) -> MixedEffectsModel:
        """Run full pipeline: prepare data, filter, add predictors, build formula, and fit model.

        Args:
            several_predictors: Boolean on whether or not to have predictor columns for every co-player.
            min_rounds (int): Minimum rounds required to include a player.
            reml (bool): Whether to fit the model using REML.
            re_formula_mode (str): 
                Random effects formula mode ('none', 'date', 'predictors', 'both').

        Returns:
            MixedEffectsModel: self, with fitted model and all preprocessing attributes updated.

        Side Effects:
            Updates self.min_rounds, self.reml, self.re_formula_mode,
            and all attributes set during preprocessing 
            and model fitting (filtered_df, predictor_cols, formula, model, model_fit, etc.).
        """
        self.min_rounds = min_rounds
        self.reml = reml
        self.re_formula_mode = re_formula_mode

        self._prepare_raw_df()
        self._filter()
        self._add_centered_date_col()
        self._create_coplayer_indicators()
        self._map_safe_names()
        self._build_formula(several_predictors=several_predictors)
        self._fit_mixed_model()
        return self

    @property
    def get_fixed_effects(self) -> pd.DataFrame:
        """Return all fixed effects coefficients and p-values, including co-player effects.

        Returns:
            pd.DataFrame: Columns ['term', 'coef', 'p_value'], 
                one row per fixed effect (intercept, date_centered, co-players).
        """
        rows = []
        # intercept
        rows.append({
            "term": "Intercept",
            "coef": float(self.model_fit.params.get("Intercept", np.nan)),
            "p_value": float(self.model_fit.pvalues.get("Intercept", np.nan))
        })
        # date_centered
        if "date_centered" in self.model_fit.params:
            rows.append({
                "term": "date_centered",
                "coef": float(self.model_fit.params["date_centered"]),
                "p_value": float(self.model_fit.pvalues.get("date_centered", np.nan))
            })
        # co-player effects
        for orig_name, safe_name in (self.player_name_map or {}).items():
            coef = self.model_fit.params.get(safe_name, np.nan)
            pval = self.model_fit.pvalues.get(safe_name, np.nan)
            rows.append({"term": orig_name, "coef": float(coef), "p_value": float(pval)})
        return pd.DataFrame(rows)

    @property
    def get_random_effects(self) -> pd.DataFrame:
        """Return random intercepts/slopes per primary player with original player names.

        Returns:
            pd.DataFrame: Columns include 'primary', 'baseline_diff', 'individual_improvement', 
                and co-player random effects.
        """
        random_rows = []
        for primary, re_series in self.model_fit.random_effects.items():
            row = {"primary": primary}
            row.update(re_series.to_dict())
            random_rows.append(row)
        random_effects_df = pd.DataFrame(random_rows)
        # rename columns
        rename_map = {}
        if "Group" in random_effects_df.columns:
            rename_map["Group"] = "baseline_diff"
        if "date_centered" in random_effects_df.columns:
            rename_map["date_centered"] = "individual_improvement"
        for orig_name, safe_name in (self.player_name_map or {}).items():
            if safe_name in random_effects_df.columns:
                rename_map[safe_name] = orig_name
        random_effects_df.rename(columns=rename_map, inplace=True)
        return random_effects_df

    def get_pair_effects(self) -> pd.DataFrame:
        """Return pair-specific effects: how much each co-player affects each primary player."""
        fixed_df = self.get_fixed_effects.set_index("term")
        random_df = self.get_random_effects.set_index("primary")
        effects = []

        for primary in random_df.index:
            for col in random_df.columns:
                if col.startswith("with_") or col in self.player_name_map.values():
                    fixed_val = fixed_df.loc[col, "coef"] if col in fixed_df.index else np.nan
                    random_val = random_df.loc[primary, col]
                    coplayer = col.replace("with_", "")

                    if primary == coplayer:
                        effect_on_primary = 0
                    else:
                        effect_on_primary = fixed_val + random_val

                    effects.append({
                        "primary": primary,
                        "coplayer": coplayer,
                        "effect_on_primary": effect_on_primary
                    })

        return pd.DataFrame(effects)

    def plot_pair_effect_heatmap(
            self,
            cmap,
            figsize: tuple = (10, 8)
    ) -> plt.Figure | None:
        """
        Plot a heatmap of pairwise effects: how much each co-player affects each primary player.

        Args:
            cmap: Matplotlib colormap to use.
            figsize (tuple): Figure size.

        Returns:
            matplotlib.figure.Figure: Heatmap figure, or None if no data.
        """
        df_pairs = self.get_pair_effects()
        if df_pairs.empty:
            return None

        # Pivot to matrix and round values
        matrix = df_pairs.pivot(index="primary", columns="coplayer", values="effect_on_primary").fillna(0).round(2)

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(matrix, cmap=cmap, xticklabels=True, yticklabels=True, square=True, annot=True, cbar=True, ax=ax)
        ax.set_title("Parvis effekt: Hvor meget en spiller på x-aksen påvirker en på y-aksen", color="black")
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        return fig

    def get_summary_stats(self) -> dict:
        resid = self.get_residuals()
        ss_total = ((self.filtered_df['round_total_score'] - self.filtered_df['round_total_score'].mean()) ** 2).sum()
        ss_resid = (resid ** 2).sum()
        r2 = 1 - ss_resid / ss_total if ss_total > 0 else np.nan

        # MixedLMResults might have aic/bic/llf only if not reml or if available
        aic = getattr(self.model_fit, "aic", "not existent")
        bic = getattr(self.model_fit, "bic", "not existent")
        llf = getattr(self.model_fit, "llf", "not existent")
        converged = getattr(self.model_fit, "converged", "not existent")
        nobs = getattr(self.model_fit, "nobs", "not existent")

        return {
            "aic": aic,
            "bic": bic,
            "llf": llf,
            "converged": converged,
            "nobs": nobs,
            "r2": r2
        }

    def get_residuals(self) -> pd.Series:
        """Return residuals from the fitted model.

        Returns:
            pd.Series: Residuals aligned with rows in self.filtered_df.
        """
        return self.model_fit.resid

    def get_fitted_values(self) -> pd.Series:
        """Return fitted values from the model.

        Returns:
            pd.Series: Fitted values aligned with rows in self.filtered_df.
        """
        return self.model_fit.fittedvalues

    def plot_residuals_vs_fitted(self, figsize=(8, 6)) -> plt.Figure:
        """Return a figure showing residuals vs fitted values."""
        resid = self.get_residuals()
        fitted = self.get_fitted_values()
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(fitted, resid, alpha=0.7)
        ax.axhline(0, color='red', linestyle='--')
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs Fitted")
        plt.tight_layout()
        return fig

    def compute_baseline_ranking(self) -> pd.DataFrame:
        re_df = self.get_random_effects.copy()
        fixed_df = self.get_fixed_effects
        try:
            fixed_intercept = fixed_df.query("term == 'Intercept'")["coef"].iloc[0]
        except Exception:
            fixed_intercept = 0.0
        re_df["Færdighed"] = re_df.get("baseline_diff", 0) + fixed_intercept
        df_baseline = re_df[["primary", "Færdighed"]].copy()
        df_baseline = df_baseline.sort_values("Færdighed", ascending=True).reset_index(drop=True)
        df_baseline["Pos."] = df_baseline.index + 1
        df_baseline.rename(columns={"primary": "Spiller"}, inplace=True)
        return df_baseline[["Pos.", "Spiller", "Færdighed"]].round(3)

    def compute_individual_improvement(self) -> pd.DataFrame:
        re_df = self.get_random_effects.copy()
        fixed_df = self.get_fixed_effects
        try:
            fixed_slope = fixed_df.query("term == 'date_centered'")["coef"].iloc[0]
        except Exception:
            fixed_slope = 0.0
        if "individual_improvement" in re_df.columns:
            re_df["Forbedring"] = re_df["individual_improvement"] + fixed_slope
            df_improve = re_df[["primary", "Forbedring"]].copy()
            df_improve = df_improve.sort_values("Forbedring", ascending=True).reset_index(drop=True)
            df_improve["Pos."] = df_improve.index + 1
            df_improve.rename(columns={"primary": "Spiller"}, inplace=True)
            return df_improve[["Pos.", "Spiller", "Forbedring"]].round(3)
        else:
            return pd.DataFrame(columns=["Pos.", "Spiller", "Forbedring"])

    def compute_effects_on_others(self) -> pd.DataFrame:
        fixed_df = self.get_fixed_effects
        effect_cols = [term for term in fixed_df["term"] if term.startswith("with_")]
        df_effects = pd.DataFrame({
            "Spiller": [col.replace("with_", "") for col in effect_cols],
            "Effekt": fixed_df.loc[fixed_df["term"].isin(effect_cols), "coef"].values,
            "p-værdi": fixed_df.loc[fixed_df["term"].isin(effect_cols), "p_value"].values,
        })
        df_effects = df_effects.sort_values("Effekt", ascending=True).reset_index(drop=True)
        df_effects["Pos."] = df_effects.index + 1
        return df_effects[["Pos.", "Spiller", "Effekt", "p-værdi"]].round(3)

    def plot_mixedlm_pairplot(self, figsize=(10, 10)) -> plt.Figure:
        """Create a 3x3 scatter/histogram grid of baseline, improvement, and effect."""

        # Compute data (Danish-labeled columns)
        df_baseline = self.compute_baseline_ranking()[["Spiller", "Færdighed"]]
        df_improve = self.compute_individual_improvement()[["Spiller", "Forbedring"]]
        df_effects = self.compute_effects_on_others()[["Spiller", "Effekt på andre"]]

        # Merge on player name
        df_merge = df_baseline.merge(df_improve, on="Spiller", how="outer")
        df_merge = df_merge.merge(df_effects.rename(columns={"Effekt på andre": "Effekt"}), on="Spiller", how="outer")

        # Pairplot (diagonal = KDE, off-diagonal = scatter)
        g = sns.pairplot(
            df_merge,
            vars=["Færdighed", "Forbedring", "Effekt"],
            diag_kind="kde",
            kind="scatter",
            plot_kws={"s": 40, "alpha": 0.7},
        )

        g.figure.subplots_adjust(top=0.95)
        g.figure.suptitle("Spillerfærdigheder, forbedring og effekt på andre", fontsize=16)

        # Convert PairGrid figure to plt.Figure
        fig = g.figure
        plt.tight_layout()
        return fig


class SimilarityModel:
    """Computes similarity between players based on hole scores using different approaches.

    Attributes:
        season (Season): 
            Season object containing raw player data.
        min_rounds (Optional[int]): 
            Minimum rounds required to include a player.
        filtered_df (Optional[pd.DataFrame]): 
            Filtered DataFrame after applying min_rounds.
        mean_vector (Optional[pd.DataFrame]): 
            Player-by-hole mean vectors after skill-centering.
        mean_vector_similarity (Optional[pd.DataFrame]): 
            Similarity or distance matrix between mean vectors.
        distributions (Optional[Dict[str, List[np.ndarray]]]): 
            Player-wise per-hole score distributions after skill-centering.
        distributions_similarity (Optional[pd.DataFrame]): 
            Player×player distance matrix using distributions (EMD).
        metric (Optional[str]): 
            Distance metric for mean-vector similarity ('cosine', 'minkowski', 'chebyshev').
    """

    error_msg = "method must be 'mean_vector' or 'distribution'"

    def __init__(self, season: Season) -> None:
        """Initialize SimilarityModel with a Season object.

        Side Effects:
            Sets all attributes to None or defaults.
        """
        self.season: Season = season
        self.min_rounds: Optional[int] = None
        self.filtered_df: Optional[pd.DataFrame] = None
        self.mean_vector: Optional[pd.DataFrame] = None
        self.mean_vector_similarity: Optional[pd.DataFrame] = None
        self.distributions: Optional[dict[str, list[np.ndarray]]] = None
        self.distributions_similarity: Optional[pd.DataFrame] = None
        self.metric: Optional[str] = None
        self.method: Optional[str] = None

    # --- Data filtering ---
    def _filter(self) -> None:
        """Filter players by minimum rounds and keep only hole columns.

        Side Effects:
            Updates self.filtered_df with players meeting min_rounds and only hole columns.

        Raises:
            AttributeError: If min_rounds is not set.
        """
        if self.min_rounds is None:
            raise AttributeError("min_rounds must be set before filtering")
        df = svcs.FilteringService.filter_players_by_rounds(
            self.season.data.copy(), self.min_rounds, "name"
        )
        hole_cols = [c for c in df.columns if c.startswith("hole_")]
        self.filtered_df = df[["name"] + hole_cols]

    # --- Mean-vector approach ---
    def _build_mean_vectors(self, zscore: bool = True) -> None:
        """Compute skill-centered mean vectors for each player.

        Args:
            zscore (bool): If True, normalize each player's vector by its standard deviation.

        Side Effects:
            Sets self.mean_vector.
        """
        df = self.filtered_df.copy()
        mean_per_hole = df.groupby("name").mean()
        mean_vector = mean_per_hole.sub(mean_per_hole.mean(axis=1), axis=0)
        if zscore:
            mean_vector = mean_vector.div(mean_vector.std(axis=1), axis=0)
        self.mean_vector = mean_vector

    # --- Distribution approach ---
    def _build_distributions(self, zscore: bool = False) -> None:
        """Compute skill-centered distributions for each player per hole.

        Args:
            zscore (bool): If True, normalize scores by per-hole SD after centering.

        Side Effects:
            Sets self.distributions.
        """
        df = self.filtered_df.copy()
        player_distributions: dict[str, list[np.ndarray]] = {}
        for player, group in df.groupby("name"):
            mu_player = group.drop(columns="name").values.flatten().mean()
            distributions = []
            for col in df.columns.drop("name"):
                scores = group[col].values - mu_player
                if zscore:
                    sigma = scores.std(ddof=0)
                    if sigma > 0:
                        scores = scores / sigma
                distributions.append(scores)
            player_distributions[player] = distributions
        self.distributions = player_distributions

    # --- Mean-vector similarity metrics ---
    def _cosine_similarity(self) -> None:
        """Compute cosine similarity between player mean vectors.

        Side Effects:
            Sets self.mean_vector_similarity.
        """
        sims = cdist(self.mean_vector.values, self.mean_vector.values, metric="cosine")
        self.mean_vector_similarity = pd.DataFrame(
            sims,
            index=self.mean_vector.index,
            columns=self.mean_vector.index
        )

    def _minkowski_distance(self, p: int = 3) -> None:
        """Compute Minkowski distances (order p) between player mean vectors.

        Args:
            p (int): Minkowski order (default 3).

        Side Effects:
            Sets self.mean_vector_similarity.
        """
        dists = cdist(self.mean_vector.values, self.mean_vector.values, metric='minkowski', p=p)
        self.mean_vector_similarity = pd.DataFrame(
            dists,
            index=self.mean_vector.index,
            columns=self.mean_vector.index
        )

    def _chebyshev_distance(self) -> None:
        """Compute Chebyshev distances between player mean vectors.

        Side Effects:
            Sets self.mean_vector_similarity.
        """
        dists = cdist(self.mean_vector.values, self.mean_vector.values, metric='chebyshev')
        self.mean_vector_similarity = pd.DataFrame(
            dists,
            index=self.mean_vector.index,
            columns=self.mean_vector.index
        )

    # --- Combined computation functions ---
    def _compute_mean_vector_similarity(self, **kwargs) -> None:
        """Compute similarity or distance between mean vectors using the specified metric.

        Side Effects:
            Updates self.mean_vector_similarity.

        Raises:
            ValueError: If mean vector not built or metric unknown.
        """
        if self.mean_vector is None:
            raise ValueError("Mean vector not built. Call _build_mean_vectors first.")
        if self.metric == "cosine":
            self._cosine_similarity()
        elif self.metric == "minkowski":
            p = kwargs.get("p", 3)
            self._minkowski_distance(p=p)
        elif self.metric == "chebyshev":
            self._chebyshev_distance()
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def _compute_distribution_similarity(self) -> None:
        """Compute average Earth Mover's Distance (EMD) between player distributions per hole.

        Side Effects:
            Updates self.distributions_similarity.
        """
        players = list(self.distributions.keys())
        n = len(players)
        distance_matrix = np.zeros((n, n))
        for i, pi in enumerate(players):
            for j, pj in enumerate(players):
                if j < i:
                    continue
                emd_sum = 0
                for h in range(14):
                    scores_i = self.distributions[pi][h]
                    scores_j = self.distributions[pj][h]
                    emd_sum += wasserstein_distance(scores_i, scores_j)
                distance_matrix[i, j] = distance_matrix[j, i] = emd_sum / 14
        self.distributions_similarity = pd.DataFrame(
            distance_matrix,
            index=players,
            columns=players
        )

    # --- High-level analysis ---
    def analyze(
            self,
            method: str = "mean_vector",
            metric: str = "cosine",
            min_rounds: int = 10,
            **kwargs
    ) -> SimilarityModel:
        """Run full similarity analysis pipeline.

        Args:
            method (str): "mean_vector" or "distribution".
            metric (str): 
                Distance metric for mean-vector method ('cosine', 'minkowski', 'chebyshev').
            min_rounds (int): Minimum rounds to include players.
            **kwargs: Additional arguments for distance metrics (e.g., p for Minkowski).

        Side Effects:
            Updates self.min_rounds, self.metric, self.filtered_df,
            self.mean_vector, self.distributions, and similarity matrices.

        Raises:
            ValueError: If method is not recognized.
        """
        self.min_rounds = min_rounds
        self.metric = metric
        self.method = method
        self._filter()
        self._build_mean_vectors()
        self._build_distributions()
        if self.method == "mean_vector":
            self._compute_mean_vector_similarity(**kwargs)
        elif self.method == "distribution":
            self._compute_distribution_similarity()
        else:
            raise ValueError(self.error_msg)
        return self

    # --- Player ranking ---
    def rank_by_similarity(self) -> pd.DataFrame:
        if self.method == "mean_vector":
            if self.mean_vector_similarity is None:
                raise ValueError("Mean vector similarity not computed yet.")
            matrix = self.mean_vector_similarity.copy()
        elif self.method == "distribution":
            if self.distributions_similarity is None:
                raise ValueError("Distribution similarity not computed yet.")
            matrix = self.distributions_similarity.copy()
        else:
            raise ValueError(self.error_msg)

        total_distance = matrix.sum(axis=1)
        ranking = total_distance.rank(method="min")

        df = pd.DataFrame({
            "Pos.": ranking,
            "Spiller": matrix.index,
            "Total distance": total_distance.round(2)
        }).sort_values("Pos.").reset_index(drop=True)

        return df

    # --- Plotting ---
    def plot_similarity_heatmap(self, cmap: LinearSegmentedColormap | None = "viridis") -> plt.Figure:
        """Create a heatmap of the player×player similarity/distance matrix and return the figure.

        Args:
            cmap (str | None): Colormap name for the heatmap.

        Returns:
            plt.Figure: Matplotlib figure object containing the heatmap.

        Raises:
            ValueError: If similarity matrix for the chosen method has not been computed.
        """
        if self.method == "mean_vector":
            matrix = self.mean_vector_similarity
            title = f"{self.method} baseret lighed"
            if matrix is None:
                raise ValueError("Mean-vector lighed ikke beregnet endnu.")
        elif self.method == "distribution":
            matrix = self.distributions_similarity
            title = "Distribution-baseret lighed (EMD)"
            if matrix is None:
                raise ValueError("Distribution-lighed ikke beregnet endnu.")
        else:
            raise ValueError(self.error_msg)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(matrix, cmap=cmap, xticklabels=True, yticklabels=True, square=True, ax=ax, annot=True, fmt=".2f")
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return fig


if __name__ == "__main__":
    raw_data = pd.read_csv("combined_all.csv")
    MySeason = Season(raw_data, "25seas")
    MixedLM = MixedEffectsModel(MySeason).analyze(min_rounds=10, reml=True, re_formula_mode="both")
    Similarity = SimilarityModel(MySeason)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    print(MixedLM.get_fixed_effects)
    print(MixedLM.get_random_effects)
