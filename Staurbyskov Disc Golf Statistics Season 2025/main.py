"""Main module to load disc golf data, fit models, and generate a PDF report."""

import sys
import os
import logging

from discgolfdata import DiscGolfData
from models import MixedEffectsModel, SimilarityModel
from Entities.season import Season
from Entities.constants import Constants
from report_manager import Reporter

# ---------------- Configuration ----------------
DATA_FOLDER = "tournament_data"
SEASON_NAME = "25seas"
PLOTS_FOLDER = "Plots"
TABLES_FOLDER = "Tables"
PDF_FILENAME = f"{SEASON_NAME}_Staurbyskov_Disc_Golf_Stats.pdf"
LOG_FILENAME = "report_generation.log"
SIMILARITY_METHOD = "mean_vector"
MIN_ROUNDS = Constants.MIN_PLAYER_ROUNDS

# ---------------- Logger Setup ----------------
logger = logging.getLogger("DiscGolfReport")
logger.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# File handler
file_handler = logging.FileHandler(LOG_FILENAME, mode="a")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# ---------------- Main Execution ----------------
if __name__ == "__main__":
    try:
        logger.info("Starting disc golf report generation.")

        # Ensure output folders exist
        logger.info(f"Ensuring output folders exist: {PLOTS_FOLDER}, {TABLES_FOLDER}")
        os.makedirs(PLOTS_FOLDER, exist_ok=True)
        os.makedirs(TABLES_FOLDER, exist_ok=True)

        # Load and preprocess data
        logger.info(f"Loading data from folder '{DATA_FOLDER}'...")
        data_loader = DiscGolfData(folder=DATA_FOLDER, date_regex=r"(\d{4}-\d{2}-\d{2})")
        data = data_loader.load_data()
        if data is None or data.empty:
            raise ValueError(f"No data loaded from folder '{DATA_FOLDER}'")
        logger.info(f"Successfully loaded {len(data)} rows of data.")

        # Initialize season
        logger.info(f"Initializing season '{SEASON_NAME}'...")
        my_season = Season(df=data, season_name=SEASON_NAME)
        logger.info("Season initialized.")

        # Fit mixed-effects model
        try:
            logger.info("Fitting mixed-effects model...")
            mixed_effects_model = MixedEffectsModel(my_season)
            mixed_effects_model.analyze(min_rounds=MIN_ROUNDS, reml=True, re_formula_mode="date")
            logger.info("Mixed-effects model fitted successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to fit mixed-effects model: {e}") from e

        # Fit similarity model
        try:
            logger.info(f"Fitting similarity model (method='{SIMILARITY_METHOD}')...")
            similarity_model = SimilarityModel(my_season)
            similarity_model.analyze(
                method=SIMILARITY_METHOD,
                metric="minkowski",
                min_rounds=MIN_ROUNDS,
                p=1
            )
            logger.info("Similarity model fitted successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to fit similarity model: {e}") from e

        # Generate PDF report
        try:
            logger.info(f"Generating PDF report '{PDF_FILENAME}'...")
            reporter = Reporter(
                season=my_season,
                mixed_lm=mixed_effects_model,
                similarity_model=similarity_model,
                min_rounds=MIN_ROUNDS,
                save_plots=True,
                save_tables=True,
                plots_folder=PLOTS_FOLDER,
                tables_folder=TABLES_FOLDER
            )
            reporter.build_pdf(PDF_FILENAME)
            logger.info(f"PDF report successfully generated: {PDF_FILENAME}")
        except Exception as e:
            raise RuntimeError(f"Failed to generate PDF report: {e}") from e

        logger.info("Disc golf report generation completed successfully.")

    except Exception as e:
        logger.error(e, exc_info=True)
        sys.exit(1)