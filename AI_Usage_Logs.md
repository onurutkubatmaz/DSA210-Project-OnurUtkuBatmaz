# AI Usage Logs — DSA 210 Project
**Student:** Onur Utku Batmaz
**Project:** The Price of Professionalism: Market Value and On-Field Discipline
**Tools Used:** Gemini (Google), Claude (Anthropic)

---

## Phase 1 — Proposal

### Purpose
AI tools were used as an editorial and formatting assistant to:
- Translate the student's original hypothesis and conceptual ideas into formal academic English
- Align the proposal layout with DSA 210 course guidelines 
- Convert custom formulas into LaTeX format as requested by the Learning Assistant

### Key Prompts
- *"I have a hypothesis that expensive squads are more disciplined to protect their value. Can you help me phrase this in academic English for my DSA 210 proposal?"*
- *"To avoid roster-size bias (like Chelsea's large squad vs. others), I want to use 'Average Market Value per Player'. How do I write this formula in LaTeX?"*
- *"The TA suggested 115 data points are too few. I decided to expand this to a 5-season multi-season analysis (2020–2025) to reach ~585 observations. Help me update the 'Data Characteristics' section with these numbers."*

### Human Ownership
The core hypothesis, the 5-year period selection, the decision to include Süper Lig, and the use of Average Market Value (AMV) were entirely developed by the student. The rationale for using Average Market Value per Player rather than total squad value was independently developed by the student. In the 2022–23 season, Chelsea registered over 50 players on their squad — meaning their total squad value was artificially inflated relative to clubs with leaner rosters. Using total value would have made Chelsea appear wealthier than they functionally were on a per-player basis, skewing cross-club comparisons. AMV corrects for this by normalizing across squad size. The methodological reasoning behind avoiding player-level analysis (positional bias, minute normalization issues) was also independently developed by the student.

---

## Phase 2 — Data Collection & EDA

### Purpose
AI tools were used to assist with Python scripting and debugging:
- Debugging pandas merge and data cleaning logic
- Reviewing EDA script structure and plot formatting
- Suggesting appropriate statistical visualizations for skewed distributions

### Key Prompts
- *"My FBRef and Transfermarkt data have slightly different team name formats. How do I merge them reliably in pandas?"*
- *"How should I visualize a right-skewed distribution alongside a more normal one on the same figure?"*
- *"I'm getting a KeyError on 'AMV' — can you check why?"*

### Human Ownership
All data was collected manually by the student from FBRef and Transfermarkt. The decision to engineer Discipline Points (CrdY×1 + CrdR×3) and the CoA index were the student's own. All analytical interpretations of the EDA output were made by the student.

---

## Phase 3 — Hypothesis Testing & ML

### Purpose
AI tools were used for code review and debugging:
- Reviewing hypothesis testing script for correct scipy usage
- Checking for data leakage in ML feature selection
- Debugging K-Means silhouette scoring and classification binning logic

### Key Prompts
- *"Is it correct to use scipy.stats.pearsonr for this? I want to make sure I'm testing the right thing."*
- *"I have CrdY and CrdR in my dataset — should I include them as ML features or would that be data leakage?"*
- *"My confusion matrix shows the Medium class is very hard to predict. Is this expected?"*

### Human Ownership
The choice of statistical tests (Pearson, Spearman, t-test, ANOVA), the decision to exclude CoA/CrdY/CrdR from ML features, the ML model selection, and all interpretations of results were made by the student. The finding that Premier League and Ligue 1 correlations are outlier-driven was independently observed and articulated by the student.

---

## Phase 4 — Final Report

### Purpose
AI tools were used to assist with report writing and LaTeX formatting:
- Structuring the report sections according to DSA 210 guidelines
- Proofreading and improving academic tone of student-written text
- Converting the report to LaTeX/Overleaf format for PDF export
- Placing figures in appropriate sections with captions

### Key Prompts
- *"I've written my findings — can you help me improve the academic tone without changing the meaning?"*
- *"Write me the LaTeX code for this report so I can compile it in Overleaf."*
- *"Place these 12 figures in the appropriate sections of the report."*

### Human Ownership
All analytical content, findings, interpretations, limitations, and conclusions in the final report were written and owned by the student. AI assistance was limited to formatting, academic phrasing, and LaTeX conversion. The data, analysis pipeline, and all results are entirely the student's own work.

---

## Summary

| Phase | AI Role | Human Role |
|-------|---------|------------|
| Proposal | Academic phrasing, LaTeX formatting | Hypothesis, research question, data design |
| Data & EDA | Debugging, plot structure | Data collection, feature engineering, interpretation |
| Hypothesis & ML | Code review, leakage check | Test selection, model choice, all interpretations |
| Final Report | Formatting, LaTeX, proofreading | All content, findings, conclusions |
