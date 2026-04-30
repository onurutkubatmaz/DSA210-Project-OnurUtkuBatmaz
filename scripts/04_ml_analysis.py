"""
DSA 210 – Introduction to Data Science | 2025–2026 Spring Term
Project: The Price of Professionalism
FILE: 04_ml_analysis.py

Machine Learning Pipeline:
  1. Data loading & preprocessing
  2. Regression  – Linear, Ridge, Lasso, Random Forest
  3. Clustering  – K-Means (elbow + silhouette)
  4. Classification – Low / Mid / High aggression (LR, RF, KNN)

Features used:
  AMV (€M)       ← main predictor (hypothesis variable)
  Poss (%)        ← control variable
  League          ← one-hot encoded
  Season          ← ordinal encoded

Excluded intentionally:
  CoA             ← derived from target * AMV → data leakage
  CrdY / CrdR     ← directly compute the target → data leakage
  Total MV (€M)   ← highly collinear with AMV (r≈0.95+)
"""

# ─────────────────────────────────────────────
# 0. Imports & Paths
# ─────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, classification_report,
                             confusion_matrix, silhouette_score)

# ── Paths ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name.lower() in ["notebook", "notebooks", "scripts"]:
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_RAW = PROJECT_ROOT / "data" / "raw"
OUTPUTS  = PROJECT_ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

print("PROJECT_ROOT :", PROJECT_ROOT)
print("DATA_RAW     :", DATA_RAW)
print("OUTPUTS      :", OUTPUTS)

# ─────────────────────────────────────────────
# 1. Load & Preprocess
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("DSA 210 – ML ANALYSIS PIPELINE")
print("=" * 60)

df = pd.read_excel(DATA_RAW / "Master_Dataset.xlsx", sheet_name="Master Data")
print(f"\n[DATA] Loaded {len(df)} rows × {len(df.columns)} columns")

# Ordinal-encode Season (2020-21 → 0, …, 2024-25 → 4)
season_order = sorted(df["Season"].unique())
df["Season_enc"] = df["Season"].map({s: i for i, s in enumerate(season_order)})

# One-hot encode League
df_encoded = pd.get_dummies(df, columns=["League"], prefix="L", dtype=int)

FEATURES = ["AMV (€M)", "Poss (%)", "Season_enc"] + \
           [c for c in df_encoded.columns if c.startswith("L_")]
TARGET   = "Discipline Points"

X = df_encoded[FEATURES].values
y = df_encoded[TARGET].values

print(f"[DATA] Features : {FEATURES}")
print(f"[DATA] Target   : {TARGET}  (mean={y.mean():.1f}, std={y.std():.1f})\n")

# ─────────────────────────────────────────────
# 2. REGRESSION
# ─────────────────────────────────────────────
print("─" * 60)
print("PART 1 – REGRESSION")
print("─" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

reg_models = {
    "Linear Regression" : LinearRegression(),
    "Ridge (α=1.0)"     : Ridge(alpha=1.0),
    "Lasso (α=0.1)"     : Lasso(alpha=0.1, max_iter=10000),
    "Random Forest"     : RandomForestRegressor(
                            n_estimators=200, max_depth=6,
                            random_state=42, n_jobs=-1),
}

reg_results = {}
for name, model in reg_models.items():
    Xt = X_train_sc if name != "Random Forest" else X_train
    Xv = X_test_sc  if name != "Random Forest" else X_test
    model.fit(Xt, y_train)
    preds = model.predict(Xv)
    r2  = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    reg_results[name] = {"R²": r2, "MAE": mae, "RMSE": rmse}
    print(f"  {name:<22} R²={r2:.4f}  MAE={mae:.2f}  RMSE={rmse:.2f}")

best_reg_name = max(reg_results, key=lambda k: reg_results[k]["R²"])
print(f"\n  ✓ Best regressor: {best_reg_name} (R²={reg_results[best_reg_name]['R²']:.4f})")

# Feature importance from Random Forest
rf_reg = reg_models["Random Forest"]
importances = pd.Series(rf_reg.feature_importances_, index=FEATURES).sort_values(ascending=False)

# Cross-validation on Random Forest
cv_scores = cross_val_score(
    RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1),
    X, y, cv=5, scoring="r2")
print(f"  RF 5-Fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─────────────────────────────────────────────
# 3. CLUSTERING
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("PART 2 – CLUSTERING")
print("─" * 60)

# Use 2 interpretable features for visual clustering
cluster_features = ["AMV (€M)", "Discipline Points"]
X_cl = df[cluster_features].values
sc_cl = StandardScaler()
X_cl_sc = sc_cl.fit_transform(X_cl)

# Elbow + Silhouette to find optimal k
inertias, silhouettes = [], []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_cl_sc)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_cl_sc, km.labels_))

# Pick k with best silhouette
best_k = list(K_range)[np.argmax(silhouettes)]
print(f"  Silhouette scores: { {k: round(s,3) for k,s in zip(K_range, silhouettes)} }")
print(f"  ✓ Optimal k = {best_k}")

# Fit final clustering
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["Cluster"] = km_final.fit_predict(X_cl_sc)

# Cluster profiles
cluster_profile = df.groupby("Cluster")[
    ["AMV (€M)", "Discipline Points", "Poss (%)"]].mean().round(2)
cluster_profile["Count"] = df["Cluster"].value_counts().sort_index().values
print("\n  Cluster Profiles:")
print(cluster_profile.to_string())

# Label clusters meaningfully
cluster_labels = {}
for c in range(best_k):
    amv  = cluster_profile.loc[c, "AMV (€M)"]
    disc = cluster_profile.loc[c, "Discipline Points"]
    if amv > cluster_profile["AMV (€M)"].median():
        cluster_labels[c] = "High Value" if disc < cluster_profile["Discipline Points"].median() else "High Value / Aggressive"
    else:
        cluster_labels[c] = "Low Value / Disciplined" if disc < cluster_profile["Discipline Points"].median() else "Low Value / Aggressive"
df["Cluster_Label"] = df["Cluster"].map(cluster_labels)
print("\n  Cluster Label Map:", cluster_labels)

# ─────────────────────────────────────────────
# 4. CLASSIFICATION
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("PART 3 – CLASSIFICATION (Aggression Level)")
print("─" * 60)

# Bin discipline points into 3 classes using tercile thresholds
q1 = np.percentile(y, 33)
q3 = np.percentile(y, 67)
print(f"  Thresholds: Low < {q1:.0f}  |  Medium {q1:.0f}–{q3:.0f}  |  High > {q3:.0f}")

def label_aggression(pts):
    if pts < q1:   return "Low"
    elif pts <= q3: return "Medium"
    else:           return "High"

y_cls = np.array([label_aggression(v) for v in y])
print(f"  Class distribution: { pd.Series(y_cls).value_counts().to_dict() }")

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_cls, test_size=0.2, random_state=42, stratify=y_cls)

sc2 = StandardScaler()
X_tr_sc = sc2.fit_transform(X_tr)
X_te_sc = sc2.transform(X_te)

cls_models = {
    "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest"       : RandomForestClassifier(n_estimators=200, max_depth=6,
                                                    random_state=42, n_jobs=-1),
    "KNN (k=7)"           : KNeighborsClassifier(n_neighbors=7),
}

cls_results = {}
for name, model in cls_models.items():
    Xt = X_tr_sc if name != "Random Forest" else X_tr
    Xv = X_te_sc if name != "Random Forest" else X_te
    model.fit(Xt, y_tr)
    preds = model.predict(Xv)
    acc   = accuracy_score(y_te, preds)
    cls_results[name] = {"accuracy": acc, "preds": preds, "model": model}
    print(f"  {name:<22} Accuracy = {acc:.4f}")

best_cls_name = max(cls_results, key=lambda k: cls_results[k]["accuracy"])
print(f"\n  ✓ Best classifier: {best_cls_name} (Acc={cls_results[best_cls_name]['accuracy']:.4f})")

# CV on best classifier
cv_cls = cross_val_score(
    RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1),
    X, y_cls, scoring="accuracy",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
print(f"  RF 5-Fold CV Acc: {cv_cls.mean():.4f} ± {cv_cls.std():.4f}")

print(f"\n  Best Classifier – Full Report:")
print(classification_report(y_te, cls_results[best_cls_name]["preds"], zero_division=0))

# ─────────────────────────────────────────────
# 5. VISUALIZATIONS
# ─────────────────────────────────────────────
print("─" * 60)
print("Generating plots…")

palette = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}
sns.set_theme(style="whitegrid", font_scale=1.0)
LEAGUE_ORDER = ["Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1", "Süper Lig"]

# ── Figure 1: Regression Summary (2×2) ───────────────────────────
fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle("Regression Analysis – Predicting Discipline Points", fontsize=14, fontweight="bold")

# 1a) Model comparison bar
ax = axes[0, 0]
names = list(reg_results.keys())
r2s   = [reg_results[n]["R²"] for n in names]
bars  = ax.barh(names, r2s, color=["#3498db","#2ecc71","#e67e22","#9b59b6"])
ax.set_xlabel("R² Score")
ax.set_title("Model Comparison – R²")
ax.set_xlim(0, max(r2s) * 1.2)
for bar, val in zip(bars, r2s):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9)

# 1b) Actual vs Predicted (Random Forest)
ax = axes[0, 1]
rf_preds = reg_models["Random Forest"].predict(X_test)
ax.scatter(y_test, rf_preds, alpha=0.5, s=25, color="#9b59b6", edgecolors="none")
mn, mx = y_test.min(), y_test.max()
ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect fit")
ax.set_xlabel("Actual Discipline Points")
ax.set_ylabel("Predicted Discipline Points")
ax.set_title("Actual vs Predicted (Random Forest)")
ax.legend(fontsize=9)

# 1c) Feature importance
ax = axes[1, 0]
top_features = importances.head(8)
short_names = [f.replace("L_","").replace(" (%)", "%").replace(" (€M)","") for f in top_features.index]
bars = ax.barh(short_names[::-1], top_features.values[::-1], color="#3498db")
ax.set_xlabel("Feature Importance")
ax.set_title("RF Feature Importances (Top 8)")

# 1d) AMV vs Discipline Points scatter with regression line
ax = axes[1, 1]
for league in LEAGUE_ORDER:
    sub = df[df["League"] == league]
    ax.scatter(sub["AMV (€M)"], sub["Discipline Points"], alpha=0.4, s=18, label=league)
z = np.polyfit(df["AMV (€M)"], df["Discipline Points"], 1)
p_fn = np.poly1d(z)
x_line = np.linspace(df["AMV (€M)"].min(), df["AMV (€M)"].max(), 200)
ax.plot(x_line, p_fn(x_line), "k--", lw=1.5, label=f"Trend (slope={z[0]:.2f})")
ax.set_xlabel("Average Market Value (€M)")
ax.set_ylabel("Discipline Points")
ax.set_title("AMV vs Discipline Points by League")
ax.legend(fontsize=7, ncol=2)

plt.tight_layout()
fig1.savefig(OUTPUTS / "regression_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ regression_analysis.png")

# ── Figure 2: Clustering (1×3) ────────────────────────────────────
fig2, axes = plt.subplots(1, 3, figsize=(17, 5))
fig2.suptitle("K-Means Clustering – Team Profiles", fontsize=14, fontweight="bold")

cluster_colors = sns.color_palette("tab10", best_k)

# 2a) Elbow + Silhouette
ax = axes[0]
ax2 = ax.twinx()
ax.plot(list(K_range), inertias, "bo-", label="Inertia")
ax2.plot(list(K_range), silhouettes, "rs--", label="Silhouette")
ax.axvline(best_k, color="gray", linestyle=":", lw=1.5, label=f"k={best_k}")
ax.set_xlabel("Number of Clusters (k)")
ax.set_ylabel("Inertia", color="blue")
ax2.set_ylabel("Silhouette Score", color="red")
ax.set_title("Elbow + Silhouette Method")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

# 2b) Cluster scatter (AMV vs Discipline Points)
ax = axes[1]
for c in range(best_k):
    mask = df["Cluster"] == c
    ax.scatter(df.loc[mask, "AMV (€M)"], df.loc[mask, "Discipline Points"],
               color=cluster_colors[c], alpha=0.55, s=25,
               label=f"C{c}: {cluster_labels[c]}")
centers = sc_cl.inverse_transform(km_final.cluster_centers_)
ax.scatter(centers[:, 0], centers[:, 1], c="black", marker="X", s=150, zorder=5, label="Centroids")
ax.set_xlabel("AMV (€M)")
ax.set_ylabel("Discipline Points")
ax.set_title(f"K-Means Clusters (k={best_k})")
ax.legend(fontsize=8)

# 2c) Cluster profile heatmap
ax = axes[2]
profile_plot = cluster_profile[["AMV (€M)", "Discipline Points", "Poss (%)"]].T
sns.heatmap(profile_plot, annot=True, fmt=".2f", cmap="YlOrRd",
            ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8})
ax.set_title("Cluster Mean Values (Heatmap)")
ax.set_xticklabels([f"C{c}\n{cluster_labels[c]}" for c in range(best_k)],
                   fontsize=8, rotation=30, ha="right")

plt.tight_layout()
fig2.savefig(OUTPUTS / "clustering_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ clustering_analysis.png")

# ── Figure 3: Classification (2×2) ───────────────────────────────
fig3, axes = plt.subplots(2, 2, figsize=(14, 10))
fig3.suptitle("Classification Analysis – Aggression Level Prediction", fontsize=14, fontweight="bold")

order = ["Low", "Medium", "High"]

# 3a) Classifier accuracy comparison
ax = axes[0, 0]
cls_names = list(cls_results.keys())
accs = [cls_results[n]["accuracy"] for n in cls_names]
bars = ax.barh(cls_names, accs, color=["#2ecc71","#3498db","#e67e22"])
ax.set_xlabel("Accuracy")
ax.set_title("Classifier Comparison – Accuracy")
ax.set_xlim(0, 1.05)
for bar, val in zip(bars, accs):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9)

# 3b) Confusion matrix – best classifier
ax = axes[0, 1]
best_preds = cls_results[best_cls_name]["preds"]
cm = confusion_matrix(y_te, best_preds, labels=order)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=order, yticklabels=order, linewidths=0.5)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix – {best_cls_name}")

# 3c) Class distribution (actual)
ax = axes[1, 0]
class_counts = pd.Series(y_cls).value_counts().reindex(order)
bars = ax.bar(order, class_counts.values,
              color=[palette["Low"], palette["Medium"], palette["High"]], edgecolor="white")
ax.set_xlabel("Aggression Class")
ax.set_ylabel("Count")
ax.set_title("Aggression Class Distribution")
for bar, val in zip(bars, class_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            str(val), ha="center", fontsize=10)

# 3d) RF feature importance for classification
ax = axes[1, 1]
rf_cls = cls_results["Random Forest"]["model"]
imp_cls = pd.Series(rf_cls.feature_importances_, index=FEATURES).sort_values(ascending=False)
short = [f.replace("L_","").replace(" (%)", "%").replace(" (€M)","") for f in imp_cls.head(8).index]
ax.barh(short[::-1], imp_cls.head(8).values[::-1], color="#e74c3c")
ax.set_xlabel("Feature Importance")
ax.set_title("RF Feature Importances – Classification (Top 8)")

plt.tight_layout()
fig3.savefig(OUTPUTS / "classification_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ classification_analysis.png")

# ── Figure 4: League-level deep dive ─────────────────────────────
fig4, axes = plt.subplots(1, 2, figsize=(14, 5))
fig4.suptitle("League-Level Analysis", fontsize=14, fontweight="bold")

# 4a) Box plot: AMV by League
ax = axes[0]
league_data = [df[df["League"] == lg]["AMV (€M)"].values for lg in LEAGUE_ORDER]
bp = ax.boxplot(league_data, patch_artist=True, notch=False,
                medianprops=dict(color="black", lw=2))
colors_box = sns.color_palette("tab10", 6)
for patch, color in zip(bp["boxes"], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_xticks(range(1, 7))
ax.set_xticklabels([lg.replace(" ", "\n") for lg in LEAGUE_ORDER], fontsize=8)
ax.set_ylabel("AMV (€M)")
ax.set_title("Squad Market Value Distribution by League")

# 4b) Mean Discipline Points by League × Season (heatmap)
ax = axes[1]
pivot = df.pivot_table(values="Discipline Points", index="League",
                       columns="Season", aggfunc="mean").reindex(LEAGUE_ORDER)
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="RdYlGn_r", ax=ax,
            linewidths=0.5, cbar_kws={"label": "Mean Disc. Pts"})
ax.set_title("Mean Discipline Points – League × Season")
ax.set_ylabel("")
ax.tick_params(axis="x", rotation=30)

plt.tight_layout()
fig4.savefig(OUTPUTS / "league_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ league_analysis.png")

print("\n" + "=" * 60)
print("ALL DONE. Output files saved to:", OUTPUTS)
print("  regression_analysis.png")
print("  clustering_analysis.png")
print("  classification_analysis.png")
print("  league_analysis.png")
print("=" * 60)
