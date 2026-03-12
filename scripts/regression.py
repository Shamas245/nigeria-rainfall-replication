# ================================================
# SCRIPT: regression.py
# PURPOSE: Replicate Amare et al. (2018)
#          Panel regression results
# AUTHOR: Shamas Liaqat
# DATE: March 2026
# ================================================

import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
from linearmodels.panel import PooledOLS
import warnings
warnings.filterwarnings('ignore')

# ── PATHS ────────────────────────────────────────
BASE = r"D:\New folder\rainfall_replication"
OUT  = f"{BASE}/data/processed"

# ================================================
# STEP 1: LOAD PANEL DATA
# ================================================
print("="*55)
print("LOADING PANEL DATA")
print("="*55)

df = pd.read_csv(f"{OUT}/panel_clean.csv")
print(f"Rows      : {df.shape[0]:,}")
print(f"Households: {df['hhid'].nunique():,}")
print(f"Waves     : {df['wave'].nunique()}")

# ================================================
# STEP 2: SET PANEL STRUCTURE
# ================================================
print("\n" + "="*55)
print("SETTING PANEL STRUCTURE")
print("="*55)

# Panel data needs a MULTI-INDEX
# First level  = household (entity)
# Second level = wave (time)
df = df.set_index(['hhid','wave'])

print("Panel index set: hhid + wave ✅")
print(f"Sample index:")
print(df.index[:5])

# ================================================
# STEP 3: DEFINE VARIABLES
# ================================================

# Outcome variable
Y = df['log_cons_pc']

# Main variable of interest
# (rainfall shock = our instrument/treatment)
X_main = df['rain_shock']

# Control variables
controls = ['head_age','head_male',
            'hhsize','rururb']

# ================================================
# STEP 4: MODEL 1 — POOLED OLS
# ================================================
print("\n" + "="*55)
print("MODEL 1: POOLED OLS")
print("="*55)

# Build X matrix (add constant + controls)
X_ols = df[['rain_shock'] + controls].copy()
X_ols['const'] = 1.0

model1 = PooledOLS(Y, X_ols)
result1 = model1.fit(
    cov_type='clustered',
    cluster_entity=True
)

print(f"\nrain_shock coefficient:")
print(f"  β    = {result1.params['rain_shock']:.4f}")
print(f"  SE   = {result1.std_errors['rain_shock']:.4f}")
print(f"  p    = {result1.pvalues['rain_shock']:.4f}")
print(f"  R²   = {result1.rsquared:.4f}")
print(f"\nInterpretation:")
beta1 = result1.params['rain_shock']
print(f"  1 SD increase in rainfall")
print(f"  → {beta1*100:.1f}% change in consumption")

# ================================================
# STEP 5: MODEL 2 — FIXED EFFECTS (Entity only)
# ================================================
print("\n" + "="*55)
print("MODEL 2: FIXED EFFECTS (Household FE)")
print("="*55)

X_fe = df[['rain_shock'] + controls].copy()

model2 = PanelOLS(
    Y, X_fe,
    entity_effects=True,   # Household FE
    time_effects=False
)
result2 = model2.fit(
    cov_type='clustered',
    cluster_entity=True
)

print(f"\nrain_shock coefficient:")
print(f"  β    = {result2.params['rain_shock']:.4f}")
print(f"  SE   = {result2.std_errors['rain_shock']:.4f}")
print(f"  p    = {result2.pvalues['rain_shock']:.4f}")
print(f"  R²   = {result2.rsquared:.4f}")
print(f"\nInterpretation:")
beta2 = result2.params['rain_shock']
print(f"  1 SD increase in rainfall")
print(f"  → {beta2*100:.1f}% change in consumption")

# ================================================
# STEP 6: MODEL 3 — FIXED EFFECTS (Entity + Time)
# ================================================
print("\n" + "="*55)
print("MODEL 3: FIXED EFFECTS (HH + Time FE)")
print("="*55)

model3 = PanelOLS(
    Y, X_fe,
    entity_effects=True,   # Household FE
    time_effects=True      # Time FE
)
result3 = model3.fit(
    cov_type='clustered',
    cluster_entity=True
)

print(f"\nrain_shock coefficient:")
print(f"  β    = {result3.params['rain_shock']:.4f}")
print(f"  SE   = {result3.std_errors['rain_shock']:.4f}")
print(f"  p    = {result3.pvalues['rain_shock']:.4f}")
print(f"  R²   = {result3.rsquared:.4f}")
print(f"\nInterpretation:")
beta3 = result3.params['rain_shock']
print(f"  1 SD increase in rainfall")
print(f"  → {beta3*100:.1f}% change in consumption")

# ================================================
# STEP 7: COMPARE TO PAPER
# ================================================
print("\n" + "="*55)
print("COMPARISON WITH PAPER (Table 7)")
print("="*55)

print(f"""
{'Model':<30} {'Our β':>10} {'Paper β':>10}
{'─'*50}
{'Pooled OLS':<30} {result1.params['rain_shock']:>10.4f} {'N/A':>10}
{'FE (Household only)':<30} {result2.params['rain_shock']:>10.4f} {'~-0.20':>10}
{'FE (HH + Time)':<30} {result3.params['rain_shock']:>10.4f} {'~-0.368':>10}
{'─'*50}
Paper main result: β = -0.368 (p<0.01)
Our target       : β close to -0.368
""")

# ================================================
# STEP 8: SIGNIFICANCE STARS
# ================================================
print("="*55)
print("FULL RESULTS SUMMARY")
print("="*55)

def stars(p):
    if p < 0.01: return "***"
    elif p < 0.05: return "**"
    elif p < 0.10: return "*"
    else: return ""

models = [
    ("Pooled OLS", result1),
    ("FE Household", result2),
    ("FE HH + Time", result3)
]

print(f"\n{'Model':<20} {'β':>8} {'SE':>8} "
      f"{'p-val':>8} {'Sig':>5}")
print("─"*55)

for name, res in models:
    b = res.params['rain_shock']
    se = res.std_errors['rain_shock']
    p = res.pvalues['rain_shock']
    print(f"{name:<20} {b:>8.4f} {se:>8.4f} "
          f"{p:>8.4f} {stars(p):>5}")

print("\n*** p<0.01  ** p<0.05  * p<0.10")
print("\n✅ Regression complete!")