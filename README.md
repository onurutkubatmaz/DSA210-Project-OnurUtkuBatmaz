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
│   │   └── SuperLig_Combined.xlsx
│   └── processed/
│       └── master_dataset.csv      # Combined 583-row dataset
│
├── scripts/
│   ├── 01_data_collection.py       # Data pipeline documentation
│   ├── 02_eda.py                   # Exploratory data analysis + plots
│   └── 03_hypothesis_testing.py    # Statistical hypothesis tests
│
├── outputs/                        # Generated figures
│   ├── plot1_distributions.png
│   ├── plot2_scatter_main.png
│   ├── plot3_scatter_by_league.png
│   ├── plot4_boxplot_league.png
│   ├── plot5_boxplot_amv_groups.png
│   ├── plot6_possession_discipline.png
│   ├── plot7_heatmap.png
│   └── plot8_season_trend.png
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
| Premier League | -0.212 | -0.116 | ✅ Pearson only |
| La Liga | -0.431 | -0.414 | ✅ Yes |
| Serie A | -0.516 | -0.417 | ✅ Yes |
| Bundesliga | -0.413 | -0.302 | ✅ Yes |
| Ligue 1 | -0.231 | -0.067 | ✅ Pearson only |
| Süper Lig | -0.249 | -0.296 | ✅ Yes |
| **All Leagues** | **-0.377** | **-0.355** | **✅ p < 0.001** |

High-AMV teams average **83 discipline points** vs **94 points** for low-AMV teams (t-test p < 0.001).

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
```

---

## AI Usage Disclosure

Generative AI tools were used during the brainstorming, drafting, and coding phases of this project. All prompts and generated outputs are documented in `AI_Usage_Logs.md` as required by course policy.