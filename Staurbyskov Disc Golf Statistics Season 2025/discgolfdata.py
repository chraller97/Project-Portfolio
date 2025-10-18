"""Service for loading and preprocessing disc golf season data.

Supports both CSV and Excel input files. Cleans, merges, and filters tournament data.

Classes:
    DiscGolfData:
        Loads tournament files, merges, cleans, and saves processed data.
"""

from typing import Optional, List
import os
import re
import pandas as pd


class DiscGolfData:
    """Loader and manager for Disc Golf data (CSV and Excel).

    This class handles loading tournament data from files, cleaning,
    merging, and exporting it to a standardized CSV file.
    """

    handicap_cols: List[str] = [
        "handicap_position",
        "handicap_position_raw",
        "handicap_relative_round_score",
        "handicap_starting_score_adjustment",
    ]

    def __init__(self, folder: str, date_regex: str) -> None:
        """Initialize the DiscGolfData object.

        Args:
            folder (str): Path to the folder containing data files (.csv or .xlsx).
            date_regex (str): Regular expression to extract a date from filenames.
        """
        self.folder = folder
        self.date_pattern = re.compile(date_regex)
        self.raw_data: Optional[pd.DataFrame] = None
        self.combined_all: Optional[pd.DataFrame] = None

    def _load_folder(self) -> None:
        """Load all CSV and Excel files from the folder and store in self.raw_data.

        Raises:
            FileNotFoundError: If the folder does not exist.
            ValueError: If reading any file fails.
        """
        if not os.path.exists(self.folder):
            raise FileNotFoundError(f"Folder not found: {self.folder}")

        dfs: List[pd.DataFrame] = []
        for filename in sorted(os.listdir(self.folder)):
            if not (filename.endswith(".xlsx") or filename.endswith(".csv")):
                continue

            file_path = os.path.join(self.folder, filename)
            match = self.date_pattern.search(filename)
            file_date = match.group(1) if match else None

            try:
                if filename.endswith(".xlsx"):
                    df = pd.read_excel(file_path)
                else:
                    df = pd.read_csv(file_path)
            except Exception as e:
                raise ValueError(f"Failed to read {file_path}: {e}") from e

            df["date"] = file_date
            dfs.append(df)

        self.raw_data = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    @staticmethod
    def _reorder_and_sort(df: pd.DataFrame) -> pd.DataFrame:
        """Reorder columns to have 'date' first and sort rows by 'date'.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Reordered and sorted DataFrame.
        """
        if df.empty:
            return df
        cols = ["date"] + [c for c in df.columns if c != "date"]
        df = df[cols].sort_values(by="date").reset_index(drop=True)
        if "card" in df.columns:
            df["card"] = df["card"].astype("Int64")
        return df

    @staticmethod
    def _filter_empty_or_allna(df: pd.DataFrame) -> pd.DataFrame:
        """Remove entirely empty or all-NA columns from the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Filtered DataFrame with only columns having at least one non-NA value.
        """
        if df.empty:
            return df
        return df.loc[:, df.notna().any()]

    @staticmethod
    def _drop_rows_missing_scores(df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows where any hole score or round summary columns are NA.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Filtered DataFrame with rows removed if any hole score,
                round_total_score, or round_relative_score is missing.
        """
        if df.empty:
            return df
        hole_cols = [c for c in df.columns if re.match(r"^hole_\\d+$", c)]
        cols_to_check = hole_cols + [
            c for c in ["round_total_score", "round_relative_score"] if c in df.columns
        ]
        return df.dropna(subset=cols_to_check)

    def load_data(self) -> pd.DataFrame:
        """Load, clean, and process tournament data.

        This method loads all CSV/XLSX files in the given folder, removes
        missing or empty data, ensures required columns exist, and returns
        a combined cleaned DataFrame.

        Returns:
            pd.DataFrame: Cleaned and combined DataFrame.
        """
        self._load_folder()
        if self.raw_data is None or self.raw_data.empty:
            self.combined_all = pd.DataFrame()
            return self.combined_all

        df = self._filter_empty_or_allna(self.raw_data)
        df = self._drop_rows_missing_scores(df)

        # Ensure handicap columns exist
        for col in self.handicap_cols:
            if col not in df.columns:
                df[col] = pd.NA

        if "division" in df.columns:
            df["division"] = df["division"].replace({"Match": "24seas", "25seas": "25seas"})

        self.combined_all = self._reorder_and_sort(df)
        return self.combined_all

    def save_csv(self, folder: str = ".", season: Optional[str] = None) -> None:
        """Save the combined DataFrame to a CSV file.

        Args:
            folder (str): Destination folder. Defaults to current directory.
            season (Optional[str]): Optional division filter to only save that season.

        Raises:
            ValueError: If no data is available to save.
        """
        if self.combined_all is None:
            print("Data not loaded yet. Loading now...")
            self.load_data()

        if self.combined_all is None or self.combined_all.empty:
            raise ValueError("No data available to save.")

        df_to_save = self.combined_all
        if season:
            df_to_save = df_to_save[df_to_save["division"] == season]

        os.makedirs(folder, exist_ok=True)
        df_to_save.to_csv(os.path.join(folder, "combined_all.csv"), index=False)


if __name__ == "__main__":
    data_loader = DiscGolfData(
        folder="tournament_data",
        date_regex=r"(\d{4}-\d{2}-\d{2})"
    )
    data_loader.save_csv(folder=".", season="25seas")
