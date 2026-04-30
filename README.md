# The Price of Professionalism
### DSA 210 – Introduction to Data Science | 2025–2026 Spring Term
**Onur Utku Batmaz**

---

## Project Overview

This project investigates whether the financial industrialization of football leads to more risk-averse, disciplined behavior on the pitch. The central hypothesis is that teams with higher average player market values incur fewer cards to protect their investments, while lower-budget teams rely on physical aggression to bridge the quality gap.

---

## Hypothesis

> **H₀:** There is no relationship between a team's average market value and its disciplinary record.
>
> **H₁:** Teams with higher average market values receive fewer cards (negative correlation).

---

## Data Sources

| Source | Variables | Method |
|--------|-----------|--------|
| [FBRef](https://fbref.com) (via StatBomb) | Yellow cards, red cards, possession % | Manual export via "Share & Export → Get table as CSV" |
| [Transfermarkt](https://www.transfermarkt.com) | Squad size, average market value, total market value | Manual collection |

---

## Dataset

- **583 team-season observations**
- **6 leagues:** Premier League, La Liga, Serie A, Bundesliga, Ligue 1, Süper Lig
- **5 seasons:** 2020–21 through 2024–25

### Engineered Features

| Feature | Formula | Description |
|---------|---------|-------------|
| Discipline Points | CrdY × 1 + CrdR × 3 | Weighted aggression score per team per season |
| AMV | Total Squad Value / Squad Size | Normalized average market value per player |
| CoA | Discipline Points × AMV | Cost of Aggression Index |

### Data Notes

> Gaziantep FK and Hatayspor were excluded from the 2022-23 Süper Lig season due to their mid-season withdrawal following the February 2023 earthquake. The remaining teams in that season played between 34 and 35 nineties — a difference of less than 3% — which was deemed insufficient to warrant normalization of disciplinary statistics.

---

## Repository Structure

```
DSA210-Project/
│
├── data/
│   ├── raw/                        # Per-league Excel files
│   │   ├── PL_Combined.xlsx
│   │   ├── LaLiga_Combined.xlsx
│   │   ├── SerieA_Combined.xlsx
│   │   ├── Bundesliga_Combined.xlsx
│   │   ├── Ligue1_Combined.xlsx
│   │   ├── SuperLig_Combined.xlsx
│   │   └── Master_Dataset.xlsx
│   └── processed/
│       └── master_dataset.csv      # Combined 583-row dataset
│
├── scripts/
│   ├── 01_data_collection.py       # Data pipeline documentation
│   ├── 02_eda.py                   # Exploratory data analysis + plots
│   ├── 03_hypothesis_testing.py    # Statistical hypothesis tests
│   └── 04_ml_analysis.py          # Machine learning pipeline
│
├── outputs/                        # Generated figures
│   ├── plot1_distributions.png
│   ├── plot2_scatter_main.png
│   ├── plot3_scatter_by_league.png
│   ├── plot4_boxplot_league.png
│   ├── plot5_boxplot_amv_groups.png
│   ├── plot6_possession_discipline.png
│   ├── plot7_heatmap.png
│   ├── plot8_season_trend.png
│   ├── regression_analysis.png
│   ├── clustering_analysis.png
│   ├── classification_analysis.png
│   └── league_analysis.png
│
├── DSA210_Proposal.pdf
├── AI_Usage_Logs.md
├── requirements.txt
└── README.md
```

---

## Key Findings (EDA + Hypothesis Testing)

| League | Pearson r | Spearman r | Significant? |
|--------|-----------|------------|--------------|
| Premier League | -0.212 | -0.116 | ⚠️ Pearson only |
| La Liga | -0.431 | -0.414 | ✅ Yes |
| Serie A | -0.516 | -0.417 | ✅ Yes |
| Bundesliga | -0.413 | -0.302 | ✅ Yes |
| Ligue 1 | -0.231 | -0.067 | ⚠️ Pearson only |
| Süper Lig | -0.249 | -0.296 | ✅ Yes |
| **All Leagues** | **-0.377** | **-0.355** | **✅ p < 0.001** |

High-AMV teams average **83 discipline points** vs **94 points** for low-AMV teams (t-test p < 0.001).

Premier League and Ligue 1 show significant Pearson correlations but non-significant Spearman correlations, suggesting the linear relationship in these leagues is driven by outlier clubs (e.g. Man City, PSG) rather than a consistent rank-order trend.

---

## Machine Learning Results

### Feature Selection

AMV (€M) and Possession (%) were used as predictors alongside one-hot encoded League and ordinal-encoded Season. CrdY, CrdR, and CoA were intentionally excluded to prevent data leakage, as they either directly compute or are derived from the target variable.

### Regression — Predicting Discipline Points

| Model | R² | MAE | RMSE |
|-------|-----|-----|------|
| Linear Regression | **0.4125** | 13.51 | 16.89 |
| Ridge (α=1.0) | 0.4125 | 13.52 | 16.89 |
| Lasso (α=0.1) | 0.4109 | 13.55 | 16.91 |
| Random Forest | 0.3858 | 13.70 | 17.27 |

The models explain approximately 41% of the variance in discipline points using market value, possession, league, and season. The remaining 59% reflects inherent noise in football — referee tendencies, match context, individual player decisions — which is expected for behavioral data.

### Clustering — Team Profiles

K-Means clustering (k=2, silhouette=0.426) reveals two dominant team profiles:

| Cluster | AMV (€M) | Disc. Points | Poss (%) | Count |
|---------|----------|--------------|----------|-------|
| Low Value / Aggressive | 3.93 | 94.3 | 48.2% | 430 |
| High Value / Disciplined | 15.68 | 72.4 | 55.2% | 153 |

This is the strongest evidence supporting the central hypothesis: high-value teams receive on average 22 fewer discipline points per season while controlling 7% more possession.

### Classification — Aggression Level Prediction

Discipline points were binned into three classes (Low < 79, Medium 79–96, High > 96):

| Model | Accuracy |
|-------|----------|
| Logistic Regression | **0.5128** |
| Random Forest | 0.4786 |
| KNN (k=7) | 0.4701 |

5-Fold CV accuracy for Random Forest: **0.5403 ± 0.0400**, substantially above the baseline of 0.33. The "High aggression" class is predicted most reliably (F1=0.65), while "Medium" is hardest to distinguish (F1=0.29), which is expected for a middle category.

---

## How to Reproduce

```bash
# Install dependencies
pip install -r requirements.txt

# Build master dataset
python scripts/01_data_collection.py

# Run EDA
python scripts/02_eda.py

# Run hypothesis tests
python scripts/03_hypothesis_testing.py

# Run ML analysis
python scripts/04_ml_analysis.py
```

---

## AI Usage Disclosure

Generative AI tools were used during the brainstorming, drafting, and coding phases of this project. All prompts and generated outputs are documented in `AI_Usage_Logs.md` as required by course policy.
