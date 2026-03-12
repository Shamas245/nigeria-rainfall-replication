# ================================================
# SCRIPT: regression.py
# PURPOSE: Replicate Amare et al. (2018)
#          Panel regression + heterogeneity
# AUTHOR: Shamas Liaqat
# DATE: March 2026
# ================================================

import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
from linearmodels.panel import PooledOLS
import warnings
warnings.filterwarnings('ignore')

BASE = r"D:\New folder\rainfall_replication"
OUT  = f"{BASE}/data/processed"

# ================================================
# STEP 1: LOAD PANEL WITH ASSETS
# ================================================
print("="*55)
print("LOADING PANEL DATA")
print("="*55)

df = pd.read_csv(f"{OUT}/panel_with_assets.csv")
print(f"Rows        : {df.shape[0]:,}")
print(f"Households  : {df['hhid'].nunique():,}")
print(f"Columns     : {list(df.columns)}")

# ================================================
# STEP 2: SET PANEL INDEX
# ================================================
df = df.set_index(['hhid','wave'])

# Variables
Y        = df['log_cons_pc']
controls = ['head_age','head_male',
            'hhsize','rururb']

# ================================================
# STEP 3: HELPER FUNCTION
# ================================================
# Instead of repeating code 5 times,
# we write ONE function that runs
# any model we pass to it.
# This is good coding practice!

def run_fe_model(data, label):
    """
    Run Fixed Effects model with
    household + time effects.
    Returns results object.
    """
    # CRITICAL: Must set MultiIndex
    # INSIDE function so linearmodels
    # can find hhid + wave structure
    data_indexed = data.set_index(
        ['hhid','wave']
    )

    Y_sub = data_indexed['log_cons_pc']
    X_sub = data_indexed[
        ['rain_shock'] + controls
    ]

    model = PanelOLS(
        Y_sub, X_sub,
        entity_effects=True,
        time_effects=True
    )
    result = model.fit(
        cov_type='clustered',
        cluster_entity=True
    )

    b  = result.params['rain_shock']
    se = result.std_errors['rain_shock']
    p  = result.pvalues['rain_shock']
    n  = data.shape[0]

    def stars(p):
        if p < 0.01:  return "***"
        elif p < 0.05: return "**"
        elif p < 0.10: return "*"
        else:          return ""

    print(f"\n{'─'*45}")
    print(f"MODEL: {label}")
    print(f"{'─'*45}")
    print(f"Observations : {n:,}")
    print(f"β rain_shock : {b:.4f} {stars(p)}")
    print(f"Std Error    : {se:.4f}")
    print(f"p-value      : {p:.4f}")
    print(f"Interpretation: 1 SD more rain")
    print(f"  → {b*100:.1f}% change in consumption")

    return result, b, se, p, n

# ================================================
# STEP 4: MODEL 1 — POOLED OLS (baseline)
# ================================================
print("\n" + "="*55)
print("MODEL 1: POOLED OLS (Baseline)")
print("="*55)

X_ols = df[['rain_shock'] + controls].copy()
X_ols['const'] = 1.0

m1 = PooledOLS(Y, X_ols)
r1 = m1.fit(cov_type='clustered',
             cluster_entity=True)

b1 = r1.params['rain_shock']
p1 = r1.pvalues['rain_shock']

print(f"β = {b1:.4f} | p = {p1:.4f}")

# ================================================
# STEP 5: MODEL 2 — FE ALL HOUSEHOLDS
# ================================================
print("\n" + "="*55)
print("MODEL 2: FE — ALL HOUSEHOLDS")
print("="*55)

r2, b2, se2, p2, n2 = run_fe_model(
    df.reset_index(),
    "FE — Full Sample"
)

# ================================================
# STEP 6: MODEL 3 — FE ASSET POOR
# ================================================
print("\n" + "="*55)
print("MODEL 3: FE — ASSET POOR ONLY")
print("="*55)

# Filter to asset poor households
# Reset index needed for filtering
df_reset = df.reset_index()

poor = df_reset[
    df_reset['asset_poor'] == 1
].set_index(['hhid','wave'])

r3, b3, se3, p3, n3 = run_fe_model(
    poor.reset_index(),
    "FE — Asset Poor (bottom 50%)"
)

# ================================================
# STEP 7: MODEL 4 — FE ASSET NONPOOR
# ================================================
print("\n" + "="*55)
print("MODEL 4: FE — ASSET NONPOOR ONLY")
print("="*55)

nonpoor = df_reset[
    df_reset['asset_poor'] == 0
].set_index(['hhid','wave'])

r4, b4, se4, p4, n4 = run_fe_model(
    nonpoor.reset_index(),
    "FE — Asset Nonpoor (top 50%)"
)

# ================================================
# STEP 8: REGIONAL HETEROGENEITY
# ================================================
print("\n" + "="*55)
print("MODEL 5 & 6: NORTH vs SOUTH")
print("="*55)

# Nigeria zones:
# 1=North Central, 2=North East,
# 3=North West = NORTH
# 4=South East, 5=South South,
# 6=South West = SOUTH

df_reset['north'] = df_reset['zone'].isin(
    [1, 2, 3]
).astype(int)

north = df_reset[
    df_reset['north'] == 1
]
south = df_reset[
    df_reset['north'] == 0
]

r5, b5, se5, p5, n5 = run_fe_model(
    north,
    "FE — Northern Nigeria"
)

r6, b6, se6, p6, n6 = run_fe_model(
    south,
    "FE — Southern Nigeria"
)

# ================================================
# STEP 9: SUMMARY TABLE
# ================================================
print("\n" + "="*55)
print("FULL RESULTS vs PAPER")
print("="*55)

def stars(p):
    if p < 0.01:  return "***"
    elif p < 0.05: return "**"
    elif p < 0.10: return "*"
    else:          return ""

print(f"""
╔══════════════════════════════════════════════════════╗
║           REPLICATION RESULTS SUMMARY               ║
╠══════════════════════════════════════════════════════╣
║ Model              Our β    Paper β   Sig   N       ║
╠══════════════════════════════════════════════════════╣
║ Pooled OLS        {b1:>7.4f}    N/A      {stars(p1):<5} {r1.nobs:<6} ║
║ FE Full Sample    {b2:>7.4f}   ~0.368   {stars(p2):<5} {n2:<6} ║
║ FE Asset Poor     {b3:>7.4f}   ~0.600   {stars(p3):<5} {n3:<6} ║
║ FE Asset Nonpoor  {b4:>7.4f}   ~0.190   {stars(p4):<5} {n4:<6} ║
║ FE North          {b5:>7.4f}    larger  {stars(p5):<5} {n5:<6} ║
║ FE South          {b6:>7.4f}    smaller {stars(p6):<5} {n6:<6} ║
╚══════════════════════════════════════════════════════╝

Note: Paper β is for NEGATIVE shock on consumption
      Our β is for POSITIVE shock on consumption
      Signs are consistent — opposite just means
      opposite direction of shock defined.

*** p<0.01  ** p<0.05  * p<0.10
""")

# ================================================
# STEP 10: KEY FINDINGS
# ================================================
print("="*55)
print("KEY FINDINGS")
print("="*55)

diff = abs(b3) - abs(b4)
print(f"""
1. Rain shock effect (full sample):
   β = {b2:.4f} ({b2*100:.1f}% per 1 SD shock)

2. Asset poor vs nonpoor:
   Poor    : {b3*100:.1f}% per 1 SD shock
   Nonpoor : {b4*100:.1f}% per 1 SD shock
   Gap     : {diff*100:.1f} percentage points

3. North vs South:
   North   : {b5*100:.1f}% per 1 SD shock
   South   : {b6*100:.1f}% per 1 SD shock

Paper finding:
   Poor (-60%) more vulnerable
   than Nonpoor (-19%) ✅ or ❌?
   North more vulnerable
   than South ✅ or ❌?
""")

print("✅ Regression complete!")