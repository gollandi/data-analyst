"""
template_analysis.py
─────────────────────────────────────────────────────────────────────
Template for a complete peer-review grade analysis.
Can be run as a script or converted to Jupyter notebook via:
    jupytext --to notebook template_analysis.py

STUDY:   [FILL IN STUDY TITLE]
AUTHOR:  GJ Ollandini
DATE:    [FILL IN]
JOURNAL: [TARGET JOURNAL]
──────────────────────────────────────────────────────────────────────
ANALYSIS PRE-REGISTRATION CHECKLIST (complete before running):
  [ ] Primary outcome defined
  [ ] Primary hypothesis stated (H0 / H1)
  [ ] Sample size / power calculation documented (see §2)
  [ ] Significance threshold fixed (α = 0.05)
  [ ] Multiple testing strategy declared (BH-FDR)
  [ ] Analysis registered on OSF / ClinicalTrials.gov if applicable
──────────────────────────────────────────────────────────────────────
"""

# %%
# ─── §0  ENVIRONMENT ──────────────────────────────────────────────────────────
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.notion_extractor import NotionExtractor
from src.data_cleaner import DataCleaner
from src.stats_pipeline import StatsPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("outputs/analysis.log"),
    ],
)
logger = logging.getLogger(__name__)

# Reproducibility
SEED = 42
np.random.seed(SEED)

logger.info("Environment configured. Seed: %d", SEED)

# %%
# ─── §1  DATA EXTRACTION ─────────────────────────────────────────────────────
"""
Pulls data from Notion. Snapshot is saved automatically to data/raw/.
To re-run analysis on a specific snapshot, load the Parquet file directly:
    df_raw = pd.read_parquet("data/raw/compass_circumcision_YYYYMMDD_HHMMSS.parquet")
"""
extractor = NotionExtractor()

# Option A: Pull live from Notion
df_raw = extractor.get_database(
    "compass_circumcision",                     # ← alias in config/settings.yaml
    rename_map={
        "Patient Age": "age",                   # Notion column → Python variable
        "Pain Score D1": "pain_d1",
        "Pain Score D7": "pain_d7",
        "Wound Complication": "complication",
        "Procedure Date": "procedure_date",
        "Follow-up Days": "followup_days",
    }
)

# Option B: Load from snapshot (for reproducibility)
# df_raw = pd.read_parquet("data/raw/compass_circumcision_20240315_094512.parquet")

logger.info("Raw data: %d rows, %d columns", *df_raw.shape)
logger.info("Columns: %s", df_raw.columns.tolist())

# %%
# ─── §2  POWER CALCULATION ────────────────────────────────────────────────────
"""
Document your a-priori power calculation here.
This section is required by most peer-review journals.

Example (two independent groups, t-test):
    Effect size d = 0.5 (medium, clinically meaningful difference)
    Power (1-β) = 0.80
    α = 0.05 (two-tailed)
    → n = 64 per group (128 total)
"""
import pingouin as pg

power_result = pg.power_ttest(
    d=0.5,          # Expected Cohen's d — change to your assumption
    power=0.80,
    alpha=0.05,
    alternative="two-sided",
)
logger.info("Required n per group: %.0f", power_result)
print(f"\n📊 Power analysis: n ≈ {power_result:.0f} per group required\n")

# %%
# ─── §3  DATA CLEANING ───────────────────────────────────────────────────────
"""
All exclusion criteria documented here → goes directly into CONSORT diagram.
"""
cleaner = DataCleaner(df_raw, study_id="circumcision_outcomes")

df_clean = (
    cleaner
    .drop_duplicates(on="_notion_id")
    .enforce_types({
        "age": "float",
        "pain_d1": "float",
        "pain_d7": "float",
        "complication": "category",
        "procedure_date": "datetime",
        "followup_days": "float",
    })
    .exclude("age < 18", reason="Paediatric patients excluded per protocol")
    .exclude("followup_days < 30", reason="Insufficient follow-up (<30 days)")
    .clip_outliers("age", lo=18, hi=90, method="nullify")
    .clip_outliers("pain_d1", lo=0, hi=10, method="nullify")
    .flag_missing_threshold(threshold=0.3)
    .build()
)

# Print CONSORT flow
cleaner.print_flow()
print(cleaner.audit_trail.to_string(index=False))

# %%
# ─── §4  EXPLORATORY DATA ANALYSIS ──────────────────────────────────────────
sp = StatsPipeline(df_clean, label="circumcision_outcomes")

# Table 1
table1 = sp.describe(
    continuous=["age", "pain_d1", "pain_d7", "followup_days"],
    categorical=["complication"],
)
print("\n=== TABLE 1: DESCRIPTIVE STATISTICS ===")
print(table1.to_string(index=False))
table1.to_csv("outputs/tables/table1_descriptive.csv", index=False)

# %%
# ─── §5  PRIMARY ANALYSIS ────────────────────────────────────────────────────
"""
State your primary hypothesis:
H0: No difference in [outcome] between [group A] and [group B]
H1: [group A] differs from [group B] on [outcome]
"""

# Example: compare pain scores between two technique groups
result_primary = sp.compare_two_groups(
    outcome="pain_d7",
    group_col="complication",  # ← replace with your actual grouping variable
)
print("\n=== PRIMARY OUTCOME ===")
for k, v in result_primary.items():
    if not isinstance(v, dict):
        print(f"  {k:30s}: {v}")

# %%
# ─── §6  SECONDARY ANALYSES ──────────────────────────────────────────────────
"""
List all secondary outcomes here. BH correction applied automatically.
"""
secondary_outcomes = ["pain_d1"]   # extend as needed

secondary_results = []
for outcome in secondary_outcomes:
    r = sp.compare_two_groups(outcome=outcome, group_col="complication")
    secondary_results.append(r)

# %%
# ─── §7  CORRELATION ANALYSIS ────────────────────────────────────────────────
corr_df = sp.correlation_matrix(
    columns=["age", "pain_d1", "pain_d7", "followup_days"],
    plot=True,
)
print("\n=== CORRELATIONS ===")
print(corr_df[["X", "Y", "r", "CI95%", "p-corr", "BF10"]].to_string(index=False))
corr_df.to_csv("outputs/tables/correlations.csv", index=False)

# %%
# ─── §8  REGRESSION MODEL ────────────────────────────────────────────────────
"""
Multivariable analysis — adjust for confounders.
Disclose variable selection strategy (clinical a-priori vs stepwise).
"""
reg_df = sp.linear_regression(
    outcome="pain_d7",
    predictors=["age", "pain_d1"],  # ← extend with confounders
)
print("\n=== REGRESSION: pain_d7 ===")
print(reg_df.to_string(index=False))
reg_df.to_csv("outputs/tables/regression.csv", index=False)

# %%
# ─── §9  EXPORT ALL RESULTS ──────────────────────────────────────────────────
results_path = sp.export_results()
print(f"\n✅ All results exported → {results_path}")
print("\n📁 Outputs:")
for p in Path("outputs").rglob("*"):
    if p.is_file():
        print(f"   {p}")
