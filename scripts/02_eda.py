import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────
from pathlib import Path

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name.lower() in ["notebook", "notebooks", "scripts"]:
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_RAW = PROJECT_ROOT / "data" / "raw"
OUTPUTS  = PROJECT_ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

print("PROJECT_ROOT :", PROJECT_ROOT)
print("DATA_RAW     :", DATA_RAW)
print("OUTPUTS      :", OUTPUTS)

# ── Veri yükle ────────────────────────────────────────────────────
files = [
    DATA_RAW / "PL_Combined.xlsx",
    DATA_RAW / "LaLiga_Combined.xlsx",
    DATA_RAW / "SerieA_Combined.xlsx",
    DATA_RAW / "Bundesliga_Combined.xlsx",
    DATA_RAW / "Ligue1_Combined.xlsx",
    DATA_RAW / "SuperLig_Combined.xlsx",
]
dfs = [pd.read_excel(f) for f in files]
df = pd.concat(dfs, ignore_index=True)

LEAGUE_COLORS = {
    'Premier League': '#3D85C8',
    'La Liga':        '#8E44AD',
    'Serie A':        '#C0392B',
    'Bundesliga':     '#E67E22',
    'Ligue 1':        '#2471A3',
    'Süper Lig':      '#E30A17',
}
leagues = list(LEAGUE_COLORS.keys())

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})

# ══════════════════════════════════════════════════════════════════
# PLOT 1 — Genel Dağılım: AMV ve Discipline Points histogramları
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Distribution of Key Variables (n=583)', fontsize=14, fontweight='bold', y=1.02)

axes[0].hist(df['AMV (€M)'], bins=30, color='#2E86AB', edgecolor='white', linewidth=0.5)
axes[0].set_xlabel('Average Market Value (€M)', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title('AMV Distribution')
axes[0].axvline(df['AMV (€M)'].median(), color='red', linestyle='--', linewidth=1.5, label=f'Median: €{df["AMV (€M)"].median():.1f}M')
axes[0].legend()

axes[1].hist(df['Discipline Points'], bins=25, color='#E84855', edgecolor='white', linewidth=0.5)
axes[1].set_xlabel('Discipline Points (YC×1 + RC×3)', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].set_title('Discipline Points Distribution')
axes[1].axvline(df['Discipline Points'].median(), color='navy', linestyle='--', linewidth=1.5, label=f'Median: {df["Discipline Points"].median():.0f}')
axes[1].legend()

plt.tight_layout()
plt.savefig(OUTPUTS / 'plot1_distributions.png', bbox_inches='tight')
plt.close()
print("Plot 1 kaydedildi.")

# ══════════════════════════════════════════════════════════════════
# PLOT 2 — Ana Scatter: AMV vs Discipline Points (lig renkli)
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 7))

for league, color in LEAGUE_COLORS.items():
    sub = df[df['League'] == league]
    ax.scatter(sub['AMV (€M)'], sub['Discipline Points'],
               color=color, alpha=0.55, s=35, label=league, zorder=3)

# Trend çizgisi
m, b, r, p, _ = stats.linregress(df['AMV (€M)'], df['Discipline Points'])
x_line = np.linspace(df['AMV (€M)'].min(), df['AMV (€M)'].max(), 200)
ax.plot(x_line, m * x_line + b, color='black', linewidth=2,
        linestyle='--', label=f'Trend (r={r:.3f}, p<0.001)', zorder=5)

ax.set_xlabel('Average Market Value per Player (€M)', fontsize=12)
ax.set_ylabel('Discipline Points (YC×1 + RC×3)', fontsize=12)
ax.set_title('AMV vs Discipline Points — All Leagues (2020–25)', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.2, zorder=0)

plt.tight_layout()
plt.savefig(OUTPUTS / 'plot2_scatter_main.png', bbox_inches='tight')
plt.close()
print("Plot 2 kaydedildi.")

# ══════════════════════════════════════════════════════════════════
# PLOT 3 — Lig bazında scatter (6 ayrı panel)
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('AMV vs Discipline Points by League', fontsize=14, fontweight='bold')

for ax, league in zip(axes.flatten(), leagues):
    sub = df[df['League'] == league]
    color = LEAGUE_COLORS[league]
    ax.scatter(sub['AMV (€M)'], sub['Discipline Points'],
               color=color, alpha=0.6, s=30, zorder=3)
    m, b, r, p, _ = stats.linregress(sub['AMV (€M)'], sub['Discipline Points'])
    x_line = np.linspace(sub['AMV (€M)'].min(), sub['AMV (€M)'].max(), 100)
    ax.plot(x_line, m * x_line + b, color='black', linewidth=1.5, linestyle='--')
    sig = "p<0.05" if p < 0.05 else f"p={p:.2f}"
    ax.set_title(f'{league}\nr={r:.3f}, {sig}', fontsize=10, fontweight='bold')
    ax.set_xlabel('AMV (€M)', fontsize=9)
    ax.set_ylabel('Discipline Points', fontsize=9)
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(OUTPUTS / 'plot3_scatter_by_league.png', bbox_inches='tight')
plt.close()
print("Plot 3 kaydedildi.")

# ══════════════════════════════════════════════════════════════════
# PLOT 4 — Boxplot: Lig bazında Discipline Points
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))

data_by_league = [df[df['League'] == l]['Discipline Points'].values for l in leagues]
colors = [LEAGUE_COLORS[l] for l in leagues]

bp = ax.boxplot(data_by_league, patch_artist=True, notch=False,
                medianprops=dict(color='black', linewidth=2))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_xticklabels(leagues, fontsize=10)
ax.set_ylabel('Discipline Points', fontsize=12)
ax.set_title('Discipline Points Distribution by League (2020–25)', fontsize=14, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUTS / 'plot4_boxplot_league.png', bbox_inches='tight')
plt.close()
print("Plot 4 kaydedildi.")

# ══════════════════════════════════════════════════════════════════
# PLOT 5 — Boxplot: Yüksek vs Düşük AMV grubu
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 6))

median_amv = df['AMV (€M)'].median()
df['AMV_Group'] = df['AMV (€M)'].apply(lambda x: f'High AMV\n(≥€{median_amv:.1f}M)' if x >= median_amv else f'Low AMV\n(<€{median_amv:.1f}M)')

high = df[df['AMV (€M)'] >= median_amv]['Discipline Points']
low  = df[df['AMV (€M)'] <  median_amv]['Discipline Points']
t_stat, p_val = stats.ttest_ind(high, low)

bp = ax.boxplot([low.values, high.values], patch_artist=True, notch=False,
                medianprops=dict(color='black', linewidth=2))
bp['boxes'][0].set_facecolor('#E84855'); bp['boxes'][0].set_alpha(0.7)
bp['boxes'][1].set_facecolor('#2E86AB'); bp['boxes'][1].set_alpha(0.7)

ax.set_xticklabels([f'Low AMV\n(<€{median_amv:.1f}M)\nn={len(low)}',
                    f'High AMV\n(≥€{median_amv:.1f}M)\nn={len(high)}'], fontsize=11)
ax.set_ylabel('Discipline Points', fontsize=12)
ax.set_title(f'High vs Low AMV Groups\nt={t_stat:.2f}, p<0.001\nMean diff: {low.mean()-high.mean():.1f} pts', 
             fontsize=12, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUTS / 'plot5_boxplot_amv_groups.png', bbox_inches='tight')
plt.close()
print("Plot 5 kaydedildi.")

# ══════════════════════════════════════════════════════════════════
# PLOT 6 — Possession vs Discipline Points
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))

for league, color in LEAGUE_COLORS.items():
    sub = df[df['League'] == league]
    ax.scatter(sub['Poss (%)'], sub['Discipline Points'],
               color=color, alpha=0.45, s=30, label=league, zorder=3)

m, b, r, p, _ = stats.linregress(df['Poss (%)'], df['Discipline Points'])
x_line = np.linspace(df['Poss (%)'].min(), df['Poss (%)'].max(), 200)
ax.plot(x_line, m * x_line + b, color='black', linewidth=2,
        linestyle='--', label=f'Trend (r={r:.3f}, p<0.001)', zorder=5)

ax.set_xlabel('Possession (%)', fontsize=12)
ax.set_ylabel('Discipline Points', fontsize=12)
ax.set_title('Possession vs Discipline Points (Control Variable)', fontsize=14, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(OUTPUTS / 'plot6_possession_discipline.png', bbox_inches='tight')
plt.close()
print("Plot 6 kaydedildi.")

# ══════════════════════════════════════════════════════════════════
# PLOT 7 — Korelasyon Heatmap
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 6))

corr_cols = ['AMV (€M)', 'Poss (%)', 'Discipline Points', 'CrdY', 'CrdR', 'CoA']
corr_matrix = df[corr_cols].corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=ax,
            linewidths=0.5, annot_kws={'size': 10})
ax.set_title('Correlation Matrix — Key Variables', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUTS / 'plot7_heatmap.png', bbox_inches='tight')
plt.close()
print("Plot 7 kaydedildi.")

# ══════════════════════════════════════════════════════════════════
# PLOT 8 — Sezon bazlı korelasyon trendi
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))

seasons = sorted(df['Season'].unique())
r_vals, p_vals, n_vals = [], [], []
for s in seasons:
    sub = df[df['Season'] == s]
    r, p = stats.pearsonr(sub['AMV (€M)'], sub['Discipline Points'])
    r_vals.append(r)
    p_vals.append(p)
    n_vals.append(len(sub))

bars = ax.bar(seasons, r_vals, color=['#2E86AB' if r < 0 else '#E84855' for r in r_vals],
              alpha=0.8, edgecolor='white')
ax.axhline(0, color='black', linewidth=1)
ax.axhline(-0.3, color='gray', linewidth=1, linestyle='--', alpha=0.5, label='r = -0.3 reference')
ax.set_ylabel("Pearson r (AMV vs Discipline Points)", fontsize=11)
ax.set_title("Correlation by Season — All Leagues", fontsize=13, fontweight='bold')
ax.set_ylim(-0.7, 0.1)

for i, (r, p, n) in enumerate(zip(r_vals, p_vals, n_vals)):
    ax.text(i, r - 0.03, f'r={r:.2f}\nn={n}', ha='center', va='top', fontsize=9)

ax.legend()
ax.grid(True, axis='y', alpha=0.2)
plt.tight_layout()
plt.savefig(OUTPUTS / 'plot8_season_trend.png', bbox_inches='tight')
plt.close()
print("Plot 8 kaydedildi.")

print("\n✅ Tüm 8 grafik oluşturuldu!")
