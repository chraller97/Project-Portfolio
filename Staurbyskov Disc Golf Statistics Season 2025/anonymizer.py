"""
Service for anonymizing player names and exporting reduced Disc Golf tournament data.

This module provides the `DiscGolfAnonymizer` class, which loads all Excel or CSV files
from a subfolder (e.g., `tournament_data`), replaces player names with anonymized IDs,
selects relevant columns, and saves anonymized versions to `tournament_data_anonymized`
with filenames prefixed by `anonymized_`.
"""

from typing import List, Dict
import os
import pandas as pd


class DiscGolfAnonymizer:
    """Anonymizer for player names in Disc Golf data.

    Attributes:
        in_folder (str): Path to the folder containing raw tournament files.
        out_folder (str): Path to the folder where anonymized files are saved.
        name_cols (List[str]): Possible player name column names.
        mapping (Dict[str, str]): Dictionary mapping original → anonymized names.
    """

    def __init__(
        self,
        in_folder: str = "tournament_data",
        out_folder: str = "tournament_data_anonymized",
    ) -> None:
        self.in_folder = in_folder
        self.out_folder = out_folder
        self.name_cols = ["player", "name", "player_name"]
        self.mapping: Dict[str, str] = {}

    def _load_all_files(self) -> List[tuple[str, pd.DataFrame]]:
        """Load all CSV or Excel files from the input folder."""
        if not os.path.exists(self.in_folder):
            raise FileNotFoundError(f"Folder not found: {self.in_folder}")

        dfs: List[tuple[str, pd.DataFrame]] = []
        for filename in sorted(os.listdir(self.in_folder)):
            path = os.path.join(self.in_folder, filename)
            if filename.endswith(".csv"):
                df = pd.read_csv(path)
            elif filename.endswith(".xlsx"):
                df = pd.read_excel(path)
            else:
                continue
            dfs.append((filename, df))
        return dfs

    def _collect_unique_names(self, dfs: List[tuple[str, pd.DataFrame]]) -> List[str]:
        """Extract unique player names across all dataframes."""
        names = set()
        for _, df in dfs:
            for col in self.name_cols:
                if col in df.columns:
                    names.update(df[col].dropna().astype(str))
        return sorted(names)

    @staticmethod
    def _generate_mapping(names: List[str]) -> Dict[str, str]:
        """Generate deterministic anonymization mapping."""
        return {name: f"Player_{i+1}" for i, name in enumerate(names)}

    @staticmethod
    def _select_relevant_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Select division, anonymized player name, card, scores, and hole columns."""
        hole_cols = [c for c in df.columns if c.startswith("hole_")]
        selected = [
            c for c in ["division", "name", "card", "round_relative_score", "round_total_score"] if c in df.columns
        ]
        selected += hole_cols
        return df[selected]

    def anonymize(self) -> Dict[str, str]:
        """Load all files, anonymize player names, and save anonymized versions."""
        dfs = self._load_all_files()
        names = self._collect_unique_names(dfs)
        self.mapping = self._generate_mapping(names)
        os.makedirs(self.out_folder, exist_ok=True)

        for filename, df in dfs:
            df_copy = df.copy()
            for col in self.name_cols:
                if col in df_copy.columns:
                    df_copy[col] = df_copy[col].replace(self.mapping)
            reduced = self._select_relevant_columns(df_copy)
            out_name = f"anonymized_{filename.replace('.xlsx', '.csv')}"
            out_path = os.path.join(self.out_folder, out_name)
            reduced.to_csv(out_path, index=False)

        return self.mapping

    def save_mapping(self, path: str = "anonymization_mapping.csv") -> None:
        """Save mapping to CSV."""
        if not self.mapping:
            raise ValueError("No mapping to save. Run anonymize() first.")
        pd.DataFrame(list(self.mapping.items()), columns=["original", "anonymized"]).to_csv(path, index=False)


if __name__ == "__main__":
    anonymizer = DiscGolfAnonymizer()
    mapping = anonymizer.anonymize()
    anonymizer.save_mapping()
    print(f"Anonymized {len(mapping)} unique players. Saved to 'tournament_data_anonymized/'.")
