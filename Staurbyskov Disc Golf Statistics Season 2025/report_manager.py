"""
Reporter module for generating PDF reports of disc golf seasons.

This module defines the Reporter class, which creates structured PDF reports
for a Season object, including player rankings, tournament summaries,
hole statistics, and visualizations. It relies on Matplotlib, ReportLab,
and optionally pre-analyzed model objects.
"""

from __future__ import annotations

# Standard library
import os
from io import BytesIO
from typing import List, Optional

# Third-party libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator, FixedFormatter, FixedLocator
from scipy.special import expit
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Table,
    TableStyle,
    Image,
    Flowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors

# Local imports
import models
from Entities.constants import Constants
from numbered_canvas_reportlab import NumberedPageCanvas


class Reporter:
    """Generates a PDF report for a Season object using pre-analyzed models.

    This class consolidates results from fitted statistical models, converts
    plots and tables to ReportLab flowables, and builds a styled PDF report.
    It can optionally save all plots and tables to disk for debugging or reuse.

    Args:
        season (Season): Season object containing player and round data.
        mixed_lm (MixedEffectsModel): Pre-fitted mixed-effects model instance.
        similarity_model (SimilarityModel): Pre-fitted similarity model instance.
        min_rounds (int, optional): Minimum number of rounds for player inclusion. Defaults to 10.
        save_plots (bool, optional): Whether to persist plots to disk. Defaults to False.
        save_tables (bool, optional): Whether to persist tables to disk. Defaults to False.
        plots_folder (str, optional): Directory to save generated plots. Defaults to "Plots".
        tables_folder (str, optional): Directory to save generated tables. Defaults to "Tables".

    Attributes:
        STYLESHEET (StyleSheet1): ReportLab default stylesheet.
        PAGE_SIZE (tuple): Page dimensions in points.
        MARGINS (dict): Margins in points.
        COLOR_PALETTE (dict): Consistent color definitions for plots and tables.
        table_style (TableStyle): Standard table style for all tables.
        caption_style (ParagraphStyle): Style used for image captions.
        styles (StyleSheet1): Local reference to the class stylesheet.
        doc (Optional[BaseDocTemplate]): Active ReportLab document instance.
        page_width (float): Effective page width after margins.
        page_height (float): Effective page height after margins.

    Side Effects:
        - Creates `plots_folder` and/or `tables_folder` directories if `save_plots` or `save_tables` are True.
        - May close Matplotlib figures to release memory after embedding.
        - May write PNG and CSV files to disk.

    Notes:
        - Reporter assumes all models are already analyzed.
        - Use `save_plots=True` or `save_tables=True` for debugging or reproducibility.
        - Designed for consistency in figure and table rendering across reports.
    """


    # ---------------- Class Attributes ----------------
    STYLESHEET = getSampleStyleSheet()
    PAGE_SIZE = A4
    MARGINS = dict(left=45, right=45, top=45, bottom=45)

    # Standardized color palette (from logo)
    COLOR_PALETTE = {
        "main": "#092640",  # dark blue – main color for plots
        "highlight": "#F2CB05",  # yellow – error bars, highlights
        "accent": "#403501",  # dark olive – sparingly for secondary elements
        "table_header": "#092640",  # dark blue for table headers
        "table_bg": "#D9D7CC",  # light grey for table backgrounds
        "secondary": "#fae488",  # gold – accent only
        "white": "#FFFFFF",
        "black": "#000000",
        "lightblue": "#81D4FA",
    }
    text_color = colors.HexColor(COLOR_PALETTE["black"])

    # Table style
    table_style = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), COLOR_PALETTE["table_header"]),
            ("TEXTCOLOR", (0, 0), (-1, 0), COLOR_PALETTE["white"]),
            ("BACKGROUND", (0, 1), (-1, -1), COLOR_PALETTE["table_bg"]),
            ("TEXTCOLOR", (0, 1), (-1, -1), COLOR_PALETTE["black"]),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, COLOR_PALETTE["black"]),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ]
    )

    combined_table_style = TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")])

    # Caption style
    caption_style = ParagraphStyle(
        name="CenteredCaption",
        parent=STYLESHEET["Normal"],
        alignment=1,
        textColor=text_color  # convert hex string to Color
    )


    def __init__(
        self,
        season,
        mixed_lm,
        similarity_model,
        min_rounds: int = 10,
        save_plots: bool = False,
        save_tables: bool = False,
        plots_folder: str = "Plots",
        tables_folder: str = "Tables",
    ):
        """Initialize the Reporter with models, configuration, and output settings.

        Sets up environment for report generation, including folder creation,
        page layout parameters, and references to pre-analyzed models.

        Args:
            season (Season): Season object containing all relevant competition data.
            mixed_lm (MixedEffectsModel): Pre-fitted mixed-effects model for statistical summaries.
            similarity_model (SimilarityModel): Pre-fitted similarity model for player comparison.
            min_rounds (int, optional): Minimum number of rounds for including players. Defaults to 10.
            save_plots (bool, optional): Whether to save plots as PNG files. Defaults to False.
            save_tables (bool, optional): Whether to save tables as CSV files. Defaults to False.
            plots_folder (str, optional): Output directory for saved plots. Defaults to "Plots".
            tables_folder (str, optional): Output directory for saved tables. Defaults to "Tables".

        Side Effects:
            - Creates `plots_folder` if `save_plots` is True.
            - Creates `tables_folder` if `save_tables` is True.
            - Initializes internal page size and margin configuration.

        Notes:
            - Does not build or save the PDF; use `build_pdf()` to generate output.
            - Model objects are assumed to be fully analyzed prior to passing.
        """
        self.season = season
        self.mixed_lm = mixed_lm
        self.simm = similarity_model

        self.min_rounds = min_rounds
        self.save_plots = save_plots
        self.save_tables = save_tables
        self.plots_folder = plots_folder
        self.tables_folder = tables_folder

        if self.save_plots:
            os.makedirs(self.plots_folder, exist_ok=True)
        if self.save_tables:
            os.makedirs(self.tables_folder, exist_ok=True)

        self.styles = self.STYLESHEET
        self.page_width = self.PAGE_SIZE[0] - self.MARGINS["left"] - self.MARGINS["right"]
        self.page_height = self.PAGE_SIZE[1] - self.MARGINS["top"] - self.MARGINS["bottom"]

        # Document will be created in build_pdf (so filename is provided there)
        self.doc: Optional[BaseDocTemplate] = None

    # ---------------- Utility Helpers ----------------
    def _fig_to_image(
        self,
        fig: plt.Figure,
        img_width: Optional[float] = None,
        img_height: Optional[float] = None,
        dpi: int = 300,
        save_name: Optional[str] = None
    ) -> Image:
        """Convert a Matplotlib figure to a ReportLab Image flowable.

        Saves the figure to an in-memory PNG buffer and optionally to disk.
        Preserves aspect ratio if width or height is not specified.

        Args:
            fig (plt.Figure): Matplotlib figure to convert.
            img_width (float, optional): Desired image width in points. Defaults to page width.
            img_height (float, optional): Desired image height in points. Computed if None.
            dpi (int, optional): Resolution in dots per inch. Defaults to 300.
            save_name (str, optional): Filename for saving the image if `save_plots` is True.

        Returns:
            reportlab.platypus.Image: Image flowable for PDF embedding.

        Side Effects:
            - Saves figure as a PNG in `plots_folder` if `save_plots` is True.
            - Closes the figure to free memory.
        """
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        if self.save_plots and save_name:
            with open(os.path.join(self.plots_folder, save_name), "wb") as f:
                f.write(buf.getbuffer())
        plt.close(fig)
        buf.seek(0)

        # Compute default size preserving aspect ratio if needed
        if img_width is None or img_height is None:
            try:
                fig_w_in, fig_h_in = fig.get_size_inches()
                fig_dpi = fig.dpi if hasattr(fig, "dpi") else dpi
                px_w, px_h = fig_w_in * fig_dpi, fig_h_in * fig_dpi
                aspect = px_h / px_w if px_w != 0 else 0.5
            except (AttributeError, TypeError):
                aspect = 0.5
            if img_width is None:
                img_width = self.page_width
            if img_height is None:
                img_height = img_width * aspect

        return Image(buf, width=img_width, height=img_height)

    @staticmethod
    def _save_table_to_csv(df: pd.DataFrame, save_name: str) -> None:
        """Save a DataFrame to a CSV file in the tables directory.

        Creates a 'tables' folder if it does not exist, then writes
        the provided DataFrame as a CSV file without an index.

        Args:
            df (pd.DataFrame): DataFrame to save.
            save_name (str): Name of the file (without extension).

        Side Effects:
            - Creates a 'tables' directory if missing.
            - Writes a CSV file to disk.
        """
        os.makedirs("tables", exist_ok=True)
        df.to_csv(os.path.join("tables", f"{save_name}.csv"), index=False)

    def _df_to_table(
        self,
        df: pd.DataFrame,
        width: float,
        save_name: Optional[str] = None
    ) -> Table:
        """Convert a pandas DataFrame into a styled ReportLab Table.

        Optionally saves the DataFrame to CSV if `save_tables` is True.
        Adjusts column widths dynamically for optimal visual balance.

        Args:
            df (pd.DataFrame): DataFrame to convert.
            width (float): Total table width in points.
            save_name (str, optional): Name for saved CSV file if enabled.

        Returns:
            reportlab.platypus.Table: Styled ReportLab table.

        Side Effects:
            - Saves DataFrame as CSV if `save_tables` is True.
        """
        df_clean = df.fillna("").astype(str)
        if save_name and getattr(self, "save_tables", False):
            self._save_table_to_csv(df_clean, save_name)

        data = [list(df_clean.columns)] + list(df_clean.itertuples(index=False, name=None))
        n_cols = len(df_clean.columns)
        if n_cols == 1:
            col_widths = [width]
        elif n_cols == 2:
            col_widths = [0.15 * width, 0.85 * width]
        else:
            col_widths = [0.15 * width, 0.45 * width] + [0.4 * width / (n_cols - 2)] * (n_cols - 2)
        table = Table(data, colWidths=col_widths, hAlign="CENTER")
        table.setStyle(self.table_style)
        return table

    def _list_to_table(
            self,
            data: list[list],
            col_widths: Optional[list[float]] = None,
            save_name: Optional[str] = None
    ) -> Table:
        """Convert a list-of-lists into a styled ReportLab Table.

        The first row of `data` is treated as the header.
        Column widths are evenly distributed unless specified.

        Args:
            data (List[List]): Tabular data (first row = header).
            col_widths (List[float], optional): Custom column widths in points.
            save_name (str, optional): Name for saved CSV file if enabled.

        Returns:
            reportlab.platypus.Table: Styled table ready for PDF inclusion.

        Side Effects:
            - Saves table as CSV if `save_tables` is True.
        """
        if save_name and getattr(self, "save_tables", False) and data:
            df = pd.DataFrame(data[1:], columns=data[0])
            self._save_table_to_csv(df, save_name)

        if col_widths is None:
            n_cols = len(data[0]) if data else 1
            col_widths = [self.page_width / n_cols] * n_cols

        table = Table(data, colWidths=col_widths, hAlign="CENTER")
        table.setStyle(self.table_style)
        return table

    def _get_cmap(self, name: str = "default") -> LinearSegmentedColormap:
        """Generate a reversed LinearSegmentedColormap for consistent visuals.

        Creates a custom colormap using the predefined color palette.
        Used across all heatmaps for uniform visual style.

        Args:
            name (str, optional): Colormap identifier. Defaults to "default".

        Returns:
            matplotlib.colors.LinearSegmentedColormap: Configured colormap.

        Side Effects:
            None.
        """
        cmap_colors = [
            (0.0, self.COLOR_PALETTE["main"]),
            (0.25, self.COLOR_PALETTE["lightblue"]),
            (0.5, self.COLOR_PALETTE["white"]),
            (0.75, self.COLOR_PALETTE["secondary"]),
            (1.0, self.COLOR_PALETTE["highlight"]),
        ]
        return LinearSegmentedColormap.from_list(name, cmap_colors)

    # ---------------- Public Method ----------------
    def build_pdf(self, filename: str) -> None:
        """Assemble all report sections and generate a finalized PDF document.

        Constructs the PDF structure, compiles visualizations, tables, and text,
        and writes the result to the specified filename. Each major section is
        added sequentially to form a complete season report.

        Args:
            filename (str): Output filename for the generated PDF.

        Returns:
            None

        Side Effects:
            - Writes a PDF file to disk.
            - Creates temporary in-memory plots and flowables.
            - May invoke additional model computations (e.g., MixedLM re-fit if required).

        Process:
            1. Initializes a ReportLab `BaseDocTemplate` and layout frame.
            2. Defines a page template with a footer note.
            3. Assembles the full report story from multiple section builders.
            4. Builds and writes the completed PDF to disk.

        Notes:
            - Calls internal `_build_*` methods to append report sections.
            - If the mixed-effects model was not fitted with `re_formula_mode="both"`,
              an additional fit is performed for the pair-effect heatmap.
        """
        # create the doc here with the real filename
        self.doc = BaseDocTemplate(filename, pagesize=self.PAGE_SIZE)
        frame = Frame(
            self.MARGINS["left"],
            self.MARGINS["bottom"],
            self.page_width,
            self.page_height,
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )

        def add_note(canvas, doc):
            canvas.saveState()
            canvas.setFont("Helvetica-Oblique", 8)
            canvas.setFillColor(self.COLOR_PALETTE["accent"])
            canvas.drawRightString(
                self.PAGE_SIZE[0] - self.MARGINS["right"],
                self.MARGINS["bottom"],
                "Note: SD = standard afvigelse, AVG = gennemsnit.",
            )
            canvas.restoreState()
            return canvas

        template = PageTemplate(id="Main", frames=[frame], onPage=add_note)
        self.doc.addPageTemplates([template])

        title_style = ParagraphStyle("TitleMain", parent=self.styles["Title"], textColor=self.text_color)
        heading_style = ParagraphStyle(
            "HeadingMain", parent=self.styles["Heading1"], alignment=1, textColor=self.text_color
        )

        story: list[Flowable] = []
        story += [Paragraph("Sæsonstatistikker – Staurbyskov Discgolf Tirsdagsmatch", title_style), Spacer(1, 20)]
        story += [Paragraph(
            "Denne PDF indeholder statistikker over Staurbyskovs tirsdagsmatch 2025 sæson. "
            "Statistikkerne er delt op i forskellige oversigter med information om huller, spillere og turneringer. Til sidst vises resultaterne af to mere komplekse modeller, som vil blive forklaret i deres egne afsnit. "
            f"For at de statistiske modeller er gyldige tælles kun spillere med minimum {Constants.MIN_PLAYER_ROUNDS} runder med i spillerrangeringerne."
            ),
            Spacer(1, 20)
        ]

        story += self._build_overview_with_holes()
        story += self._build_hole_stats_viz()
        story += self._build_score_histogram()
        story.append(PageBreak())

        story += [Paragraph("Spilleroversigt", heading_style), Spacer(1, 5)]
        story += self._build_player_hole_heatmap()
        story += self._build_player_scatter()
        #story += self._build_player_rankings()
        story.append(PageBreak())


        #story.append(PageBreak())

        story += [Paragraph("Turneringsoversigt", heading_style), Spacer(1, 5)]
        story += self._build_tournament_score_plot(min_players=5)
        story += self._build_tournament_rankings(min_players=5)
        story.append(PageBreak())

        story += self._build_similarity_section(method=self.simm.method)
        story.append(PageBreak())

        story += self._build_mixedlm_rankings()
        #story += self._build_mixedlm_pairplot()

        # pair effects only if model contains that mode
        re_formula_mode = getattr(self.mixed_lm, "re_formula_mode", None)
        if re_formula_mode == "both":
            story += self._build_pair_effect_heatmap(mixed_lm_model=self.mixed_lm)
        else:
            # create a new MixedLM instance fitted with re_formula_mode="both"
            mixed_lm_for_heatmap = models.MixedEffectsModel(self.season).analyze(
                min_rounds=self.min_rounds, reml=True, re_formula_mode="both"
            )
            story += self._build_pair_effect_heatmap(mixed_lm_model=mixed_lm_for_heatmap)
        story.append(PageBreak())

        story += self._build_bump_chart()
        story.append(PageBreak())

        story += self._build_rounds_played_table()

        #story += self._build_players_relative_scores_plot()

        #story += self._build_mixedlm_diagnostics()
        # build the PDF
        self.doc.build(story, canvasmaker=NumberedPageCanvas)

    def _build_overview_with_holes(self) -> list[Flowable]:
        """Build the summary and hole difficulty overview section.

        Combines general season statistics with tables of the hardest and easiest holes.
        Creates a two-column layout containing a textual summary and comparative tables.

        Returns:
            list[Flowable]: ReportLab flowables representing the section content.

        Side Effects:
            - Calls `season.get_hole_rankings()` and `season.summary_lines()` to fetch data.
            - Saves a CSV file of hole statistics if `save_tables` is True.

        Notes:
            - The left column contains key season metrics.
            - The right column includes rankings of hardest and easiest holes based on average score.
        """
        s = self.season
        hardest, easiest = s.get_hole_rankings()
        summary_lines = s.summary_lines()

        # --- Build summary section ---
        summary = [
            Paragraph("Oversigt", self.styles["Heading2"]),
            Spacer(1, 5),
            *[Paragraph(f"<b>{k}:</b> {v}", self.styles["Normal"])
              for k, v in (line.split(":", 1) for line in summary_lines)]
        ]

        # --- Layout setup ---
        left_width = self.page_width * 0.35
        right_width = self.page_width * 0.65
        col_width = right_width / 2

        # --- Tables for hardest/easiest holes ---
        def make_table(data, offset=0, file_name: str = None):
            tbl = [["Pos.", "Hul", "AVG (SD)"]] + [
                [i + 1 + offset, h, f"{avg:.2f} ({sd:.2f})"]
                for i, (h, avg, sd) in enumerate(data)
            ]
            return self._list_to_table(tbl, [col_width * 0.15, col_width * 0.3, col_width * 0.55], save_name=file_name)

        table_left = make_table(hardest, file_name="hardest_hole_stats_table")
        table_right = make_table(easiest, offset=len(hardest), file_name="easiest_hole_stats_table")
        tables = Table([[table_left, table_right]], colWidths=[col_width, col_width])

        caption = Paragraph("Tabel 1: Hul-rangeringer baseret på gennemsnitlig antal skud.", self.caption_style)
        combined = Table([[summary, [tables, caption]]], colWidths=[left_width, right_width])
        combined.setStyle(self.combined_table_style)

        return [combined]

    def _build_hole_stats_viz(self) -> list[Flowable]:
        """Build a visualization of average hole scores with standard deviation bars.

        Generates a figure using `season.plot_hole_stats()` and embeds it into the PDF
        along with a caption and section heading.

        Returns:
            list[Flowable]: ReportLab flowables representing the hole statistics plot section.

        Side Effects:
            - Saves the figure as "hole_stats.png" if `save_plots` is True.
            - Closes the Matplotlib figure after embedding.

        Notes:
            - The plot shows mean scores per hole (blue) with standard deviation (yellow lines).
        """
        fig = self.season.plot_hole_stats(self.COLOR_PALETTE)
        img = self._fig_to_image(fig, img_width=self.page_width, img_height=self.page_width * 0.3,
                                 save_name="hole_stats.png")
        caption = Paragraph(
            "Figur 1: Gennemsnitlig antal skud per hul (blå prik) ± standardafvigelse (gule linjer).",
            self.caption_style,
        )
        heading = Paragraph("Hul Statistik", self.styles["Heading2"])
        return [heading, img, caption, Spacer(1, 5)]

    def _build_score_histogram(self) -> list[Flowable]:
        """Build a histogram visualization of player round scores.

        Generates and embeds a histogram of total round scores with a fitted density curve.
        Includes a section heading and caption for clarity.

        Returns:
            list[Flowable]: ReportLab flowables representing the score histogram section.

        Side Effects:
            - Saves the figure as "score_hist.png" if `save_plots` is True.
            - Closes the Matplotlib figure after embedding.

        Notes:
            - The '0' score corresponds to playing the course at even par.
        """
        fig = self.season.plot_score_histogram(self.COLOR_PALETTE)
        img = self._fig_to_image(fig, img_width=self.page_width, img_height=self.page_width * 0.4,
                                 save_name="score_hist.png")
        caption = Paragraph("Figur 2: Histogram over runde-scores med tæthedsfunktion. Scoren '0' svarer således til at gå banen i even 'E'.", self.caption_style)
        return [
            Paragraph("Runde-scorefordeling", self.styles["Heading2"]),
            img,
            caption,
            Spacer(1, 5),
        ]

    def _build_player_rankings(self) -> list[Flowable]:
        """Build a section showing player rankings by average score and score variability.

        Generates two side-by-side tables ranking players by mean score and score standard deviation.
        Combines both tables into a single centered layout with caption and heading.

        Args:
            min_rounds (int, optional): Minimum number of rounds required for a player to be included. Defaults to 10.

        Returns:
            list[Flowable]: ReportLab flowables representing the player ranking section.

        Side Effects:
            - Calls `season.rank_players()` to compute player rankings.
            - Saves table images if `save_tables` is True.

        Notes:
            - Left table ranks players by lowest average score.
            - Right table ranks players by lowest score variability (consistency).
        """
        data_avg = [list(row) for row in
                    [("Rang", "Spiller", "Gennemsnit")] + self.season.rank_players("avg_score", self.min_rounds)]
        data_sd = [list(row) for row in
                   [("Rang", "Spiller", "Score SD")] + self.season.rank_players("sd_score", self.min_rounds)]

        table_width = self.page_width * 0.475
        spacer = (self.page_width - 2 * table_width) / 3

        table_avg = self._list_to_table(data_avg, col_widths=[table_width * 0.15, table_width * 0.6, table_width * 0.25], save_name="player_avg_table")
        table_sd = self._list_to_table(data_sd, col_widths=[table_width * 0.15, table_width * 0.6, table_width * 0.25], save_name="player_sd_table")

        combined = Table([[None, table_avg, None, table_sd, None]], colWidths=[spacer, table_width, spacer, table_width, spacer], hAlign="CENTER")
        combined.setStyle(self.combined_table_style)

        caption = Paragraph("Tabel 2: Spiller-rangeringer baseret på gennemsnit og standardafvigelse.", self.caption_style)
        return [Paragraph("Spillerrangeringer", self.styles["Heading2"]), combined, caption, Spacer(1, 20)]

    def _build_player_scatter(self) -> list[Flowable]:
        """Build a scatter plot showing the relationship between player average and score variation.

        Each point represents a player with mean round score on the x-axis and standard deviation on the y-axis.
        Includes explanatory text, figure, caption, and section header.

        Returns:
            list[Flowable]: ReportLab flowables representing the player scatter plot section.

        Side Effects:
            - Saves the figure as "player_avg_sd_scatter.png" if `save_plots` is True.
            - Closes the Matplotlib figure after embedding.

        Notes:
            - Provides insight into player consistency and overall performance.
            - Players near the lower-left are both consistent and perform well.
        """
        fig = self.season.plot_avg_vs_sd(self._get_cmap("player_scatter_cmap"), min_rounds=self.min_rounds-2)
        if fig is None:
            return [
                Paragraph("Spiller Scatter (Gennemsnit vs. SD)", self.styles["Heading2"]),
                Paragraph("Ingen data tilgængelig."),
                Spacer(1, 10),
            ]

        img = self._fig_to_image(
            fig,
            img_width=self.page_width * 0.8,
            img_height=self.page_width * 0.5,
            save_name="player_avg_sd_scatter.png",
        )
        caption = Paragraph("Figur 4: Punktplot af spillernes gennemsnit og standardafvigelse.", self.caption_style)
        forklaring = Paragraph("<b>Forklaring:</b> For hver spiller aflæses dennes gennemsnitlige runde-score på x-aksen og standardafvigelse på y-aksen. "
                               "Standardafvigelsen er hvor mange skud en spiller typisk svinger omkring sit gennemsnit. "
                               "Eks: Christian Ørnskovs gennemsnit er ca. 44. Hans SD på ca. 3,25 viser at størstedelen af hans runder ligger mellem 40,75 og 47,75.", self.styles["Normal"])

        return [
            Paragraph("Spiller punktplot (AVG vs. SD)", self.styles["Heading2"]),
            forklaring,
            img,
            caption,
            Spacer(1, 20),
        ]

    def _build_player_hole_heatmap(self) -> list[Flowable]:
        """Build a heatmap visualization of player performance per hole.

        Creates a matrix-style figure where each cell shows a player’s average score relative to par on each hole.
        Adds a descriptive explanation, caption, and section header.

        Args:
            min_rounds (int, optional): Minimum number of rounds required to include a player. Defaults to 10.

        Returns:
            list[Flowable]: ReportLab flowables representing the player-hole heatmap section.

        Side Effects:
            - Saves the figure as "player_hole_heatmap.png" if `save_plots` is True.
            - Closes the Matplotlib figure after embedding.

        Notes:
            - Useful for identifying which holes differentiate top players.
            - Darker cells indicate higher scores relative to par.
        """
        fig = self.season.plot_player_hole_heatmap(self.min_rounds, self._get_cmap("player_hole_cmap"))
        if fig is None:
            return [
                Paragraph("Spiller-hul Heatmap", self.styles["Heading2"]),
                Paragraph("Ingen data tilgængelig."),
                Spacer(1, 10),
            ]

        img_height = self.page_width * (
                    len([p for p in self.season.players.values() if p.nr_rounds >= self.min_rounds]) * 0.3 / 10) * 1.35
        img = self._fig_to_image(fig, img_width=self.page_width, img_height=img_height,
                                 save_name="player_hole_heatmap.png")
        caption = Paragraph("Figur 3: Heatmap af spillere vs. huller, score i forhold til par.", self.caption_style)

        forklaring = Paragraph("<b>Forklaring:</b> For hver spiller, gå ud til det hul man vil vide noget om. Værdien i boksen viser det forventede antal skud over/under par den spiller bruger på det hul.", self.styles["Normal"])

        return [
            Paragraph("Spiller-hul Heatmap", self.styles["Heading2"]),
            forklaring,
            img,
            caption,
            Spacer(1, 5),
        ]

    def _build_tournament_rankings(self, min_players: int = 5) -> list[Flowable]:
        """Build a section showing tournament rankings by average score and score variability.

        Creates two tables displaying the top 15 tournaments ranked by mean and standard deviation
        of player scores. Combines both into a single centered layout with caption and section heading.

        Args:
            min_players (int, optional): Minimum number of players required for a tournament to be included. Defaults to 5.

        Returns:
            list[Flowable]: ReportLab flowables representing the tournament ranking section.

        Side Effects:
            - Calls `season.rank_tournaments()` to generate rankings.
            - Saves tables as images if `save_tables` is True.

        Notes:
            - Lower average score indicates higher overall performance.
            - Lower SD reflects more consistent player performance across tournaments.
        """
        df_avg = self.season.rank_tournaments("avg_score", min_players).reset_index().head(15)
        df_avg = df_avg.rename(columns={"avg_score": "AVG"})

        df_sd = self.season.rank_tournaments("sd_score", min_players).reset_index().head(15)
        df_sd = df_sd.rename(columns={"sd_score": "SD"})

        table_width = self.page_width * 0.48
        table_avg = self._df_to_table(df_avg, table_width, save_name="tournament_avg_table")
        table_sd = self._df_to_table(df_sd, table_width, save_name="tournament_sd_table")

        spacer = (self.page_width - 2 * table_width) / 3
        combined = Table(
            [[None, table_avg, None, table_sd, None]],
            colWidths=[spacer, table_width, spacer, table_width, spacer],
            hAlign="CENTER",
        )
        combined.setStyle(self.combined_table_style)

        caption = Paragraph(
            "Tabel 2: Rangering af top 15 turneringer efter gennemsnitlig score og standardafvigelse.",
            self.caption_style,
        )
        return [
            Paragraph(
                "Turneringsrangeringer (Top 15)",
                ParagraphStyle("Heading2Main", parent=self.styles["Heading2"], textColor=self.text_color),
            ),
            combined,
            caption,
            Spacer(1, 10),
        ]

    def _build_tournament_score_plot(self, min_players: int = 5) -> list[Flowable]:
        """Build a time-series plot of tournament scores.

        Generates a scatter plot showing tournament average scores over time, with marker size
        representing the number of players per event. Includes explanatory text, caption, and header.

        Args:
            min_players (int, optional): Minimum number of players required to include a tournament. Defaults to 5.

        Returns:
            list[Flowable]: ReportLab flowables representing the tournament score plot section.

        Side Effects:
            - Saves the figure as "tournament_score_time.png" if `save_plots` is True.
            - Closes the Matplotlib figure after embedding.

        Notes:
            - Provides insight into tournament performance trends across the season.
            - The yellow line indicates a 5-tournament moving average of mean scores.
        """
        fig = self.season.plot_tournament_scores_over_time(self.COLOR_PALETTE, min_players)
        if fig is None:
            return [
                Paragraph("Turneringsscore over tid", self.styles["Heading2"]),
                Paragraph("Ingen data tilgængelig.", self.styles["Normal"]),
                Spacer(1, 10),
            ]

        img = self._fig_to_image(
            fig,
            img_width=self.page_width,
            img_height=self.page_width * (3.5 / 7),
            save_name="tournament_score_time.png",
        )
        caption = Paragraph(
            "Figur 5: Gennemsnitlige scores pr. turnering (størrelsen af prikken afspejler antal spillere).",
            self.caption_style,
        )
        forklaring = Paragraph("<b>Forklaring:</b> Hver prik viser den gennemsnitlige runde-score blandt de deltagene spillere den dato. "
                               "Størrelsen af prikken viser hvor mange spillere der deltog den dato. "
                               "Den gule linje viser gennemsnit for de seneste 5 turneringer ved hver dato.",
                               self.styles["Normal"])
        return [
            Paragraph("Turneringsscore over tid", self.styles["Heading2"]),
            forklaring,
            img,
            caption,
            Spacer(1, 10),
        ]

    def _build_similarity_section(self, method: str = "mean_vector") -> list[Flowable]:
        """Build a section analyzing player similarity using a chosen metric.

        Generates a heatmap showing similarity between players based on their hole scores
        and a ranking table derived from the similarity measure. Includes explanatory text,
        figure, table (if available), and captions.

        Args:
            method (str, optional): Similarity calculation method used for ranking. Defaults to "mean_vector".

        Returns:
            list[Flowable]: ReportLab flowables representing the similarity analysis section.

        Side Effects:
            - Calls `simm.plot_similarity_heatmap()` and `simm.rank_by_similarity()`.
            - Saves the heatmap figure as "similarity_heatmap.png" if `save_plots` is True.
            - Saves the ranking table as CSV if `save_tables` is True.
            - Closes the Matplotlib figure after embedding.

        Notes:
            - Lower total distance in the ranking table indicates a more typical player.
            - The heatmap cells show the number of strokes needed to match another player’s pattern.
            - Provides insight into patterns of strengths and weaknesses among players.
        """
        cmap = self._get_cmap("similarity_cmap")

        try:
            fig = self.simm.plot_similarity_heatmap(cmap=cmap)
        except ValueError:
            return [
                Paragraph("Lighedsanalyse", self.styles["Heading2"]),
                Paragraph("Lighed ikke beregnet.", self.styles["Normal"]),
                Spacer(1, 10),
            ]

        img = self._fig_to_image(
            fig,
            img_width=self.page_width * 0.85,
            img_height=self.page_width * 0.6375,
            save_name="similarity_heatmap.png",
        )

        caption_fig = Paragraph(
            f"Figur 6: Heatmap over parvise mønsterligheder baseret på '{method}'-lighed",
            self.caption_style
        )

        ranking_df = self.simm.rank_by_similarity()
        table = (
            self._df_to_table(ranking_df, width=self.page_width, save_name="similarity_table.png")
            if not ranking_df.empty
            else Paragraph("Ingen rangeringsdata.", self.styles["Normal"])
        )

        caption_table = Paragraph(
            f"Tabel 3: Spillerrangering baseret på '{method}'-lighed. Udregnet som summen af hver række i figuren ovenfor. Lavere totalafstand = mere typisk spiller.",
            self.caption_style,
        )

        forklaring = Paragraph(
            "<b>Forklaring:</b> Alle spillere har nogle huller, de er gode til, og nogle de ikke er, i forhold til deres eget niveau, altså et mønster. "
            "Værdierne i hver boks viser hvor mange skud der skal ændres for at få en spillers mønster til at ligne en andens. <br/>"
            "Eks: Christian og Frederik bruger begge 3 skud per hul i gennemsnit. "
            "På hul 1 bruger Christian tilgengæld 3.5 skud i gennemsnit, hvilket er +0,5 ift. hans niveau, og Frederik bruger 2.5 hvilket er -0,5 ift. hans niveau. "
            "Der er altså '1 skuds forskel' mellem Christian og Frederik på hul 1. "
            "Udregner man dette for alle huller mellem Christian og Frederik og summer værdierne sammen, fås værdien i boksen mellem Christian og Frederik.",
            self.styles["Normal"],
        )

        return [
            Paragraph(f"Lighedsanalyse", self.styles["Heading2"]),
            forklaring,
            Spacer(1, 10),
            img,
            caption_fig,
            Spacer(1, 10),
            table,
            caption_table
        ]

    def _build_mixedlm_rankings(self) -> list[Flowable]:
        """Build a section showing individual improvements and effects on other players.

        Generates two tables: left shows each player's overall improvement over the season,
        right shows the expected effect a player has on others' scores when playing together.
        Includes explanatory text, combined layout, caption, and section header.

        Returns:
            list[Flowable]: ReportLab flowables representing the mixed-effects ranking section.

        Side Effects:
            - Calls `mixed_lm.compute_individual_improvement()` and `mixed_lm.compute_effects_on_others()`.
            - Saves tables as images if `save_tables` is True.

        Notes:
            - Useful for evaluating individual performance and interaction effects.
        """
        simple_mixed_model = models.MixedEffectsModel(self.season)
        simple_mixed_model.analyze(min_rounds=self.min_rounds, reml=True, re_formula_mode="date", several_predictors=False)
        df_improve = simple_mixed_model.compute_individual_improvement()

        #df_improve = self.mixed_lm.compute_individual_improvement()
        df_effects = self.mixed_lm.compute_effects_on_others()

        available_width = self.page_width - 10
        table_width_1 = available_width * 0.45
        table_width_2 = available_width * 0.55

        table_improve = self._df_to_table(df_improve, table_width_1, save_name="improvement_table")
        table_effects = self._df_to_table(df_effects, table_width_2, save_name="effects_table")

        combined_top = Table([[table_improve, None, table_effects]], colWidths=[table_width_1, 10, table_width_2],
                             hAlign="CENTER")
        combined_top.setStyle(self.combined_table_style)

        caption_top = Paragraph(
            "Tabel 4 & 5: Individuel forbedring (venstre, målt i antal skud om ugen) og effekt på andre spillere (højre, målt i gennemsnitlig antal skud ekstra/mindre når man er på kort sammen).",
            self.caption_style
        )

        forklaring = Paragraph("<b>Forklaring:</b> Tabellen til venstre viser hvor meget hver person overordnet set har forbedret sig gennem sæson, målt i antal skud på uge."
                               "Tabellen til højre viser den forventede effekt en spiller har på en anden persons runde-score når man spiller på kort sammen.", self.styles["Normal"])

        return [
            Paragraph("Mixed-effects model Rangering", self.styles["Heading2"]),
            forklaring,
            Spacer(1, 5),
            combined_top,
            caption_top,
            Spacer(1, 5),
        ]

    def _build_mixedlm_pairplot(self) -> list[Flowable]:
        """Build a pairplot showing player skills and mixed-effects relationships.

        Displays histograms and density plots on the diagonal and scatter plots off-diagonal
        for player skill metrics, individual improvement, and effects on others.

        Returns:
            list[Flowable]: ReportLab flowables representing the mixed-effects pairplot section.

        Side Effects:
            - Saves the figure as "mixedlm_pairplot.png" if `save_plots` is True.
            - Closes the Matplotlib figure after embedding.
        """
        fig = self.mixed_lm.plot_mixedlm_pairplot(figsize=(10, 10))
        img = self._fig_to_image(
            fig,
            img_width=self.page_width,
            img_height=self.page_width,  # square for 3x3 grid
            save_name="mixedlm_pairplot.png",
        )
        caption = Paragraph(
            "Figur X: Spillerfærdigheder, individuel forbedring og effekt på andre. "
            "Diagonal viser histogram + densitet, off-diagonal scatterplots.",
            self.caption_style
        )

        return [
            Paragraph("Spillerfærdigheder og effekter (MixedLM)", self.styles["Heading2"]),
            img,
            Spacer(1, 10),
            caption,
            Spacer(1, 10),
        ]

    def _build_pair_effect_heatmap(self, mixed_lm_model: models.MixedEffectsModel) -> list[Flowable]:
        """Build a heatmap showing pairwise effects between players.

        Visualizes positive and negative effects a player has on another's round score.
        Includes figure, caption, and section header.

        Args:
            mixed_lm_model (MixedEffectsModel): Fitted mixed-effects model used to compute pair effects.

        Returns:
            list[Flowable]: ReportLab flowables representing the pairwise effect heatmap section.

        Side Effects:
            - Saves the heatmap figure as "pair_effect_heatmap.png" if `save_plots` is True.
            - Closes the Matplotlib figure after embedding.

        Notes:
            - Yellow cells indicate negative effects, blue cells positive effects.
        """
        cmap = self._get_cmap("pair_effect_cmap")
        fig = mixed_lm_model.plot_pair_effect_heatmap(cmap=cmap)
        if fig is None:
            return [
                Paragraph("Parvise effekter mellem spillere", self.styles["Heading2"]),
                Paragraph("Ingen data tilgængelig."),
                Spacer(1, 10)
            ]

        img = self._fig_to_image(
            fig,
            img_width=self.page_width * 0.8,
            img_height=self.page_width * 0.6,
            save_name="pair_effect_heatmap.png"
        )
        caption = Paragraph(
            "Figur 7: Heatmap over parvise effekter mellem spillere, når man er på kort sammen. (gul = negativ effekt, blå = positiv effekt).",
            self.caption_style
        )

        return [
            Paragraph("Parvise effekter mellem spillere", self.styles["Heading2"]),
            img,
            Spacer(1, 10),
            caption,
            Spacer(1, 1),
        ]

    def _build_mixedlm_diagnostics(self) -> list[Flowable]:
        """Build a section showing diagnostics for the fitted mixed-effects model.

        Includes summary statistics in a table and a residuals vs. fitted values plot.
        Provides insight into model fit and assumptions.

        Returns:
            list[Flowable]: ReportLab flowables representing the diagnostics section.

        Side Effects:
            - Calls `mixed_lm.get_summary_stats()` and `mixed_lm.plot_residuals_vs_fitted()`.
            - Saves the figure as "mixedlm_resid_fitted.png" if `save_plots` is True.
            - Saves the summary table as CSV if `save_tables` is True.
            - Closes the Matplotlib figure after embedding.

        Notes:
            - Residual plot uses a red reference line at zero.
            - Tables round numeric values to three decimals for readability.
        """
        """Build a section showing diagnostics for the fitted mixed effects model."""
        if self.mixed_lm.model_fit is None:
            return [
                Paragraph("Mixed model diagnostics", self.styles["Heading2"]),
                Paragraph("Ingen model er endnu fit.", self.styles["Normal"]),
                Spacer(1, 10)
            ]

        # Summary stats
        summary = self.mixed_lm.get_summary_stats()
        summary_df = pd.DataFrame(summary.items(), columns=["Stat", "Value"])
        summary_df["Value"] = summary_df["Value"].apply(lambda x: round(x, 3) if isinstance(x, (float, np.floating)) else x)

        # Use _df_to_table
        table = self._df_to_table(summary_df, width=self.page_width, save_name="mixedlm_diagnostics_table")

        # Residuals vs fitted plot
        fig = self.mixed_lm.plot_residuals_vs_fitted()
        img = self._fig_to_image(fig, img_width=self.page_width, img_height=self.page_width * 0.6,
                                 save_name="mixedlm_resid_fitted.png")

        caption = Paragraph(
            "Figur: Residualer vs. fitted values for mixed effects model. Rød streg = 0 reference.",
            self.caption_style
        )

        return [
            Paragraph(
                "Mixed Model Diagnostics",
                ParagraphStyle("Heading2Main", parent=self.styles["Heading2"], textColor=self.text_color)
            ),
            Spacer(1, 5),
            table,
            Spacer(1, 10),
            img,
            Spacer(1, 5),
            caption,
            Spacer(1, 20)
        ]

    def _build_bump_chart(self) -> list[Flowable]:
        """Build a bump chart comparing player ranks across multiple metrics.

        Combines rankings from average score, score standard deviation, similarity,
        individual improvement, and effect on others. Displays smooth lines connecting
        ranks across metrics, plateaus at each metric, value labels, and player names
        on the right axis. Includes caption and explanatory text.

        Args:
            min_rounds (int, optional): Minimum number of rounds required for a player to be included. Defaults to 10.

        Returns:
            list[Flowable]: ReportLab flowables representing the bump chart section.

        Side Effects:
            - Calls `season.rank_players()`, `simm.rank_by_similarity()`, `mixed_lm.compute_individual_improvement()`, and `mixed_lm.compute_effects_on_others()`.
            - Saves the figure as "bump_chart.png" if `save_plots` is True.
            - Closes the Matplotlib figure after embedding.

        Notes:
            - Lines are colored according to normalized rank in the first metric (AVG).
            - Plateaus represent exact ranks at each metric, and smooth curves interpolate between metrics.
            - Useful for visual comparison of player performance across multiple metrics.
        """
        """Build bump chart comparing player ranks across metrics for eligible players."""

        # --- Prepare each ranking source ---
        df_avg = pd.DataFrame(
            self.season.rank_players("avg_score", self.min_rounds),
            columns=["rank", "player", "value"]
        )

        df_sd = pd.DataFrame(
            self.season.rank_players("sd_score", self.min_rounds),
            columns=["rank", "player", "value"]
        )

        df_sim = self.simm.rank_by_similarity()[["Pos.", "Spiller", "Total distance"]]
        df_sim = df_sim.rename(columns={"Pos.": "rank", "Spiller": "player", "Total distance": "value"})

        simple_mixed_model = models.MixedEffectsModel(self.season)
        simple_mixed_model.analyze(min_rounds=self.min_rounds, reml=True, re_formula_mode="date",
                                   several_predictors=False)
        df_imp = simple_mixed_model.compute_individual_improvement()
        if not df_imp.empty:
            df_imp = df_imp.rename(columns={"Pos.": "rank", "Spiller": "player", "Forbedring": "value"})
            df_imp["rank"] = df_imp["rank"].rank(method="min", ascending=True)
        else:
            df_imp = pd.DataFrame(columns=["rank", "player", "value"])

        df_eff = self.mixed_lm.compute_effects_on_others()
        if not df_eff.empty:
            df_eff = df_eff.rename(columns={"Pos.": "rank", "Spiller": "player", "Effekt": "value"})
        else:
            df_eff = pd.DataFrame(columns=["rank", "player", "value"])

        metrics = {
            "AVG": df_avg,
            "SD": df_sd,
            "Lighed": df_sim,
            "Forbedring": df_imp,
            "Påvirkning": df_eff,
        }

        dfs = []
        for metric_name, df in metrics.items():
            if df.empty:
                continue
            df = df.copy()
            df["metric"] = metric_name
            dfs.append(df[["player", "metric", "rank", "value"]])

        df_all = pd.concat(dfs, ignore_index=True)

        # --- Numeric x positions for metrics ---
        metric_positions = {m: i + 1 for i, m in enumerate(metrics.keys())}
        df_all["metric_pos"] = df_all["metric"].map(metric_positions)

        n_players: int = df_all["player"].nunique()

        # --- Color mapping based on first metric (AVG) rank ---
        first_metric = "AVG"
        cmap = self._get_cmap("bumpchart")
        rank_ref = df_all[df_all["metric"] == first_metric].set_index("player")["rank"]
        rank_norm = (rank_ref - rank_ref.min()) / (rank_ref.max() - rank_ref.min())

        # --- Plot ---
        fig_height = max(6.0, float(n_players) * 0.5)
        fig, ax = plt.subplots(figsize=(8, fig_height), subplot_kw=dict(ylim=(0.5, 0.5 + n_players)))
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.set_xticks(range(1, len(metrics) + 1))
        ax.set_xticklabels(metrics.keys())

        plateau_halfwidth = 0.1
        lw = 2  # single linewidth variable used for both spline and plateaus
        curve_sharpness = 15

        for player, sub in df_all.groupby("player"):
            rank_value = rank_norm.loc[player]
            # if rank maps to the middle (0.5), change to light grey
            if np.isclose(rank_value, 0.5):
                color = (0.85, 0.85, 0.85, 1.0)  # light grey
            else:
                color = cmap(rank_value)

            x = sub["metric_pos"].to_numpy()
            y = sub["rank"].to_numpy()

            # dense x and smooth y as before
            x_dense = np.linspace(x.min() - plateau_halfwidth, x.max() + plateau_halfwidth, 8000)
            y_smooth = np.full_like(x_dense, np.nan, dtype=float)

            for i in range(len(x) - 1):
                x0_left = x[i] - plateau_halfwidth
                x0_right = x[i] + plateau_halfwidth
                x1_left = x[i + 1] - plateau_halfwidth

                mask_plateau = (x_dense >= x0_left) & (x_dense <= x0_right)
                y_smooth[mask_plateau] = y[i]

                mask_curve = (x_dense >= x0_right) & (x_dense <= x1_left)
                if mask_curve.any():
                    t = (x_dense[mask_curve] - x0_right) / (x1_left - x0_right)
                    w = expit(curve_sharpness * (t - 0.5))
                    y_smooth[mask_curve] = y[i] * (1 - w) + y[i + 1] * w

            mask_last = (x_dense >= x[-1] - plateau_halfwidth) & (x_dense <= x[-1] + plateau_halfwidth)
            y_smooth[mask_last] = y[-1]

            valid = ~np.isnan(y_smooth)
            y_smooth = np.interp(x_dense, x_dense[valid], y_smooth[valid])

            # draw smooth spline with chosen linewidth and cap/join style
            ax.plot(
                x_dense,
                y_smooth,
                color=color,
                lw=lw,
                zorder=2,
                solid_capstyle="butt",
                solid_joinstyle="miter",
                antialiased=True,
            )

            # draw plateaus with ax.plot so thickness exactly matches spline
            for xi, yi in zip(x, y):
                ax.plot(
                    [xi - plateau_halfwidth, xi + plateau_halfwidth],
                    [yi, yi],
                    color=color,
                    lw=lw,
                    zorder=3,
                    solid_capstyle="butt",
                    solid_joinstyle="miter",
                    antialiased=True,
                )

            # text labels
            for xi, yi, val in zip(x, y, sub["value"]):
                ax.text(
                    xi,
                    yi - 0.1,  # shift slightly up (since y-axis is inverted)
                    f"{val:.2f}",
                    va="bottom",
                    ha="center",
                    fontsize=7,
                    color="black",
                )

        # --- Right side labels (last metric) ---
        last_metric = list(metrics.keys())[-1]
        right_df = df_all[df_all["metric"] == last_metric].sort_values("rank")
        yax_right = ax.secondary_yaxis("right")
        yax_right.yaxis.set_major_locator(FixedLocator(right_df["rank"].to_list()))
        yax_right.yaxis.set_major_formatter(FixedFormatter(right_df["player"].to_list()))

        ax.invert_yaxis()
        ax.set(xlabel="", ylabel="Pos.", title="Graf over forskellige spillerrangeringer")
        ax.grid(axis="x", linestyle="--", alpha=0.6)
        ax.set_xlim(0.7, len(metrics) + 0.35)
        plt.tight_layout()

        img_height = self.page_width * 0.9  # larger than default 0.55
        img = self._fig_to_image(
            fig,
            img_width=self.page_width,
            img_height=img_height,
            save_name="bump_chart.png",
        )

        caption = Paragraph(
            "Figur 8: Spillerrangeringer på tværs af metrikker.",
            self.caption_style,
        )

        forklaring = Paragraph(
            "<b>Forklaring:</b> Grafen viser spillerne rangeret efter forskellige mål: Gennemsnit (AVG), Standardafvigelse (SD), Lighed, Forbedring og Effekt på andre (Påvirkning).",
            self.styles["Normal"]
        )

        return [
            Paragraph("Sammenligning af spillerrangeringer", self.styles["Heading2"]),
            forklaring,
            Spacer(1, 5),
            img,
            caption,
            Spacer(1, 15),
        ]

    def _build_rounds_played_table(self) -> list[Flowable]:
        """Build a table showing players ranked by number of rounds played (highest first).

        Returns:
            list[Flowable]: ReportLab flowables representing the table, caption, and spacer.

        Side Effects:
            - Calls `season.rank_players()` to retrieve player rankings.
            - Saves the table image if `save_tables` is True.

        Notes:
            The table ranks players by the number of rounds played, sorted in descending order
            so the most active players appear at the top.
        """
        ranked = self.season.rank_players("nr_rounds", self.min_rounds)
        ranked_reversed = list(reversed(ranked))
        ranked_reversed = [(i + 1, name, val) for i, (_, name, val) in enumerate(ranked_reversed)]

        data_rounds = [list(row) for row in
                       [("Rang", "Spiller", "Antal runder")] + ranked_reversed]

        table_width = self.page_width * 0.6
        table = self._list_to_table(
            data_rounds,
            col_widths=[table_width * 0.15, table_width * 0.6, table_width * 0.25],
            save_name="player_rounds_table"
        )

        caption = Paragraph("Tabel 6: Rangering af spillere efter antal spillede runder.", self.caption_style)
        return [
            Paragraph("Antal runder spillet", self.styles["Heading2"]),
            table,
            caption,
            Spacer(1, 20)]