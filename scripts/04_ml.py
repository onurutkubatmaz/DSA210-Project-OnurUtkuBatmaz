import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats

# ── Data ──────────────────────────────────────────────────────────────────────
df = pd.read_excel("data/raw/Master_Dataset.xlsx")

X_simple   = df[["AMV (€M)"]].values
X_multi    = df[["AMV (€M)", "Poss (%)"]].values
y          = df["Discipline Points"].values
leagues    = df["League"].unique()

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ── Helper ────────────────────────────────────────────────────────────────────
def fit_report(X, y, label):
    model = LinearRegression().fit(X, y)
    r2    = r2_score(y, model.predict(X))
    rmse  = np.sqrt(mean_squared_error(y, model.predict(X)))
    cv_r2 = cross_val_score(model, X, y, cv=kf, scoring="r2").mean()
    print(f"\n{label}")
    print(f"  R²          : {r2:.4f}")
    print(f"  CV R² (5-fold): {cv_r2:.4f}")
    print(f"  RMSE        : {rmse:.2f}")
    print(f"  Coefficients: {dict(zip(['AMV','Poss'] if X.shape[1]==2 else ['AMV'], model.coef_.round(4)))}")
    print(f"  Intercept   : {model.intercept_:.4f}")
    return model, r2, cv_r2, rmse

# ── Model 1: Simple Linear Regression ────────────────────────────────────────
model1, r2_1, cv_r2_1, rmse_1 = fit_report(X_simple, y, "Model 1 — AMV → Discipline Points")

# ── Model 2: Multiple Linear Regression ──────────────────────────────────────
model2, r2_2, cv_r2_2, rmse_2 = fit_report(X_multi, y, "Model 2 — AMV + Poss → Discipline Points")

# Partial correlation: AMV effect after controlling for Possession
resid_y   = y - LinearRegression().fit(df[["Poss (%)"]], y).predict(df[["Poss (%)"]])
resid_x   = df["AMV (€M)"].values - LinearRegression().fit(df[["Poss (%)"]], df["AMV (€M)"]).predict(df[["Poss (%)"]])
partial_r, partial_p = stats.pearsonr(resid_x, resid_y)
print(f"\nPartial correlation AMV → Discipline (controlling Poss): r={partial_r:.4f}, p={partial_p:.4e}")

# ── Model 3: Per-league regression ───────────────────────────────────────────
print("\nPer-League Simple Regression:")
league_results = {}
for lg in sorted(leagues):
    sub = df[df["League"] == lg]
    m   = LinearRegression().fit(sub[["AMV (€M)"]], sub["Discipline Points"])
    r2  = r2_score(sub["Discipline Points"], m.predict(sub[["AMV (€M)"]]))
    league_results[lg] = {"coef": m.coef_[0], "intercept": m.intercept_, "r2": r2, "n": len(sub)}
    print(f"  {lg:<20} coef={m.coef_[0]:+.3f}  R²={r2:.4f}  n={len(sub)}")

# ── Plot 1: Actual vs Predicted ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (model, label, r2) in zip(axes, [
    (model1, "Model 1: AMV only",        r2_1),
    (model2, "Model 2: AMV + Possession", r2_2),
]):
    X = X_simple if "only" in label else X_multi
    pred = model.predict(X)
    ax.scatter(y, pred, alpha=0.4, edgecolors="none", color="#4C8CBF")
    mn, mx = min(y.min(), pred.min()), max(y.max(), pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect fit")
    ax.set_xlabel("Actual Discipline Points")
    ax.set_ylabel("Predicted Discipline Points")
    ax.set_title(f"{label}\nR² = {r2:.4f}")
    ax.legend()
fig.suptitle("Actual vs Predicted — Linear Regression Models", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/plot9_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Plot 2: R² comparison bar chart ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
models  = ["Simple\n(AMV only)", "Multiple\n(AMV + Poss)"]
r2s     = [r2_1, r2_2]
colors  = ["#4C8CBF", "#E07B54"]
bars    = ax.bar(models, r2s, color=colors, width=0.4, edgecolor="white")
for bar, val in zip(bars, r2s):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.005, f"{val:.4f}", ha="center", fontsize=11)
ax.set_ylabel("R²")
ax.set_title("Model Comparison — R²", fontweight="bold")
ax.set_ylim(0, max(r2s) * 1.25)
plt.tight_layout()
plt.savefig("outputs/plot10_r2_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Plot 3: Per-league R² and coefficient ─────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
lg_names = list(league_results.keys())
r2_vals  = [league_results[l]["r2"]   for l in lg_names]
coef_vals= [league_results[l]["coef"] for l in lg_names]
palette  = ["#4472C4","#9B59B6","#E07B54","#F0A500","#4BACC6","#C0504D"]

ax1.barh(lg_names, r2_vals, color=palette)
ax1.set_xlabel("R²")
ax1.set_title("R² by League", fontweight="bold")
for i, v in enumerate(r2_vals):
    ax1.text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=9)

ax2.barh(lg_names, coef_vals, color=palette)
ax2.axvline(0, color="black", lw=0.8)
ax2.set_xlabel("Regression Coefficient (AMV → Discipline)")
ax2.set_title("Coefficient by League", fontweight="bold")
for i, v in enumerate(coef_vals):
    ax2.text(v - 0.05 if v < 0 else v + 0.02, i, f"{v:.3f}", va="center", fontsize=9)

fig.suptitle("Per-League Linear Regression Results", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/plot11_league_regression.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Plot 4: Residuals ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (model, label, X) in zip(axes, [
    (model1, "Model 1", X_simple),
    (model2, "Model 2", X_multi),
]):
    residuals = y - model.predict(X)
    ax.scatter(model.predict(X), residuals, alpha=0.4, color="#4C8CBF", edgecolors="none")
    ax.axhline(0, color="red", lw=1.5, linestyle="--")
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals")
    ax.set_title(f"Residual Plot — {label}", fontweight="bold")
fig.suptitle("Residual Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/plot12_residuals.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n✅ ML analysis complete. Plots saved to outputs/")
