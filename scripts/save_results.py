# ================================================
# SCRIPT: save_results.py
# PURPOSE: Save clean results table
#          for presentation
# AUTHOR: Shamas Liaqat
# DATE: March 2026
# ================================================

import pandas as pd
import numpy as np

# ── Manual entry from regression output ─────────
results = {
    'Model': [
        'Pooled OLS',
        'FE — Full Sample',
        'FE — Asset Poor',
        'FE — Asset Nonpoor',
        'FE — North Nigeria',
        'FE — South Nigeria'
    ],
    'Beta': [
        -0.0615,
         0.1502,
         0.1554,
         0.1443,
         0.0873,
         0.1464
    ],
    'Std_Error': [
        0.0190,
        0.0233,
        0.0340,
        0.0317,
        0.0417,
        0.0297
    ],
    'P_value': [
        0.0012,
        0.0000,
        0.0000,
        0.0000,
        0.0362,
        0.0000
    ],
    'Observations': [
        12531,
        12531,
         6315,
         6216,
         6485,
         6046
    ],
    'Paper_Beta': [
        'N/A',
        '~-0.368',
        '~-0.600',
        '~-0.190',
        'larger',
        'smaller'
    ],
    'Note': [
        'Baseline — no FE',
        'Main specification',
        'Bottom 50% by assets',
        'Top 50% by assets',
        'Zones 1+2+3',
        'Zones 4+5+6'
    ]
}

df = pd.DataFrame(results)

# Add significance stars
def stars(p):
    if p < 0.01:   return "***"
    elif p < 0.05: return "**"
    elif p < 0.10: return "*"
    else:          return ""

df['Significance'] = df['P_value'].apply(stars)

# Add % interpretation
df['Effect_%'] = (df['Beta'] * 100).round(1)

# ── Save ─────────────────────────────────────────
BASE = r"D:\New folder\rainfall_replication"
out  = f"{BASE}/output/tables/results_table.csv"

import os
os.makedirs(
    f"{BASE}/output/tables",
    exist_ok=True
)

df.to_csv(out, index=False)

# ── Print clean table ────────────────────────────
print("="*65)
print("REPLICATION RESULTS — Amare et al. (2018)")
print("Nigeria LSMS-ISA Panel | 3 Waves | FE Regression")
print("="*65)

print(f"\n{'Model':<22} {'β':>8} {'SE':>8} "
      f"{'Sig':>5} {'N':>7} {'Effect':>8}")
print("─"*65)

for _, row in df.iterrows():
    print(f"{row['Model']:<22} "
          f"{row['Beta']:>8.4f} "
          f"{row['Std_Error']:>8.4f} "
          f"{row['Significance']:>5} "
          f"{row['Observations']:>7,} "
          f"{row['Effect_%']:>7.1f}%")

print("─"*65)
print("*** p<0.01  ** p<0.05  * p<0.10")
print("\nDependent variable: log per capita consumption")
print("Controls: head age, head gender, HH size, rural/urban")
print("Standard errors clustered at household level")
print(f"\n✅ Saved to: {out}")