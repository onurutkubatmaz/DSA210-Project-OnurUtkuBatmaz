# =============================================================================
# DSA 210 — Introduction to Data Science | 2025–2026 Spring Term
# Project: The Price of Professionalism
# FILE: 01_data_collection.py
#
# This script documents the data collection process and builds the
# master dataset used throughout the project.
#
# Data Sources:
#   - FBRef (https://fbref.com, via StatBomb): disciplinary stats
#   - Transfermarkt (https://www.transfermarkt.com): squad market values
#
# Output:
#   data/processed/master_dataset.csv
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path

# ── 0) Project directory setup ───────────────────────────────────────────────
PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name.lower() in ["notebook", "notebooks", "scripts"]:
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_RAW       = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

print("PROJECT_ROOT   :", PROJECT_ROOT)
print("DATA_RAW       :", DATA_RAW)
print("DATA_PROCESSED :", DATA_PROCESSED)

# =============================================================================
# CELL 1: Dataset overview
# =============================================================================
# The dataset covers 5 seasons (2020–21 through 2024–25) across 6 leagues:
#   - Premier League    (FBRef id: 9)
#   - La Liga           (FBRef id: 12)
#   - Serie A           (FBRef id: 11)
#   - Bundesliga        (FBRef id: 20)
#   - Ligue 1           (FBRef id: 13)
#   - Süper Lig         (FBRef id: 26)
#
# FBRef URL pattern (Squad Standard Stats):
#   https://fbref.com/en/comps/{id}/{season}/
#
# Transfermarkt URL pattern:
#   https://www.transfermarkt.com/{league}/startseite/wettbewerb/{id}/plus/?saison_id={year}
#
# Data was collected manually via "Share & Export → Get table as CSV" on FBRef
# and screenshot extraction on Transfermarkt due to bot-detection restrictions.
#
# Variables collected per team-season:
#   FBRef:          Squad, Poss (%), CrdY, CrdR
#   Transfermarkt:  Club, Squad Size, Avg Market Value (€M), Total Market Value (€M)
#
# NOTE: Gaziantep FK and Hatayspor were excluded from the 2022–23 Süper Lig
# season due to their mid-season withdrawal following the February 2023
# earthquake, which created unequal match counts across teams and rendered
# disciplinary statistics non-comparable.

LEAGUES = [
    {"name": "Premier League", "fbref_id": 9,  "tm_id": "GB1"},
    {"name": "La Liga",        "fbref_id": 12, "tm_id": "ES1"},
    {"name": "Serie A",        "fbref_id": 11, "tm_id": "IT1"},
    {"name": "Bundesliga",     "fbref_id": 20, "tm_id": "L1"},
    {"name": "Ligue 1",        "fbref_id": 13, "tm_id": "FR1"},
    {"name": "Süper Lig",      "fbref_id": 26, "tm_id": "TR1"},
]

SEASONS = ["2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025"]

print("\nLeagues:", [l["name"] for l in LEAGUES])
print("Seasons:", SEASONS)

# =============================================================================
# CELL 2: Load per-league Excel files
# =============================================================================
# Each Excel file was built from manually collected data.
# Files expected in data/raw/:
#   PL_Combined.xlsx, LaLiga_Combined.xlsx, SerieA_Combined.xlsx,
#   Bundesliga_Combined.xlsx, Ligue1_Combined.xlsx, SuperLig_Combined.xlsx

LEAGUE_FILES = {
    "Premier League": DATA_RAW / "PL_Combined.xlsx",
    "La Liga":        DATA_RAW / "LaLiga_Combined.xlsx",
    "Serie A":        DATA_RAW / "SerieA_Combined.xlsx",
    "Bundesliga":     DATA_RAW / "Bundesliga_Combined.xlsx",
    "Ligue 1":        DATA_RAW / "Ligue1_Combined.xlsx",
    "Süper Lig":      DATA_RAW / "SuperLig_Combined.xlsx",
}

dfs = []
for league, path in LEAGUE_FILES.items():
    if path.exists():
        df = pd.read_excel(path)
        dfs.append(df)
        print(f"  ✅ {league}: {len(df)} rows loaded")
    else:
        print(f"  ❌ {league}: file not found at {path}")

# =============================================================================
# CELL 3: Combine into master dataset
# =============================================================================
master = pd.concat(dfs, ignore_index=True)
print(f"\nMaster dataset shape: {master.shape}")
print("Columns:", master.columns.tolist())
print("\nRows per league:")
print(master["League"].value_counts())
print("\nRows per season:")
print(master["Season"].value_counts().sort_index())

# =============================================================================
# CELL 4: Feature engineering
# =============================================================================
# Discipline Points already computed in each file: CrdY×1 + CrdR×3
# AMV already present from Transfermarkt
# CoA (Cost of Aggression) already computed: Discipline Points × AMV

# Verify no nulls in key columns
key_cols = ["Discipline Points", "AMV (€M)", "Poss (%)", "CoA"]
null_counts = master[key_cols].isnull().sum()
print("\nNull counts in key columns:")
print(null_counts)

# =============================================================================
# CELL 5: Descriptive statistics
# =============================================================================
print("\n── Descriptive Statistics ──")
print(master[key_cols + ["CrdY", "CrdR"]].describe().round(2))

print("\n── Mean Discipline Points by League ──")
print(master.groupby("League")["Discipline Points"].mean().round(1).sort_values())

print("\n── Mean AMV (€M) by League ──")
print(master.groupby("League")["AMV (€M)"].mean().round(2).sort_values(ascending=False))

# =============================================================================
# CELL 6: Save master dataset as CSV
# =============================================================================
out_path = DATA_PROCESSED / "master_dataset.csv"
master.to_csv(out_path, index=False)

print(f"\n✅ Master dataset saved to: {out_path}")
print(f"   Total observations: {len(master)}")
print(f"   Leagues: {master['League'].nunique()}")
print(f"   Seasons: {master['Season'].nunique()}")
print(f"   Unique teams: {master['Squad'].nunique()}")
