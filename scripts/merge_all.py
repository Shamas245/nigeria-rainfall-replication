# ================================================
# SCRIPT: merge_all.py
# PURPOSE: Merge all cleaned files into one
#          regression-ready panel dataset
# AUTHOR: Shamas Liaqat
# DATE: March 2026
# ================================================

import pandas as pd
import numpy as np

# ── PATHS ────────────────────────────────────────
BASE = r"D:\New folder\rainfall_replication"
OUT  = f"{BASE}/data/processed"

# ================================================
# STEP 1: LOAD ALL CLEANED FILES
# ================================================
print("="*55)
print("LOADING CLEANED FILES")
print("="*55)

cons = pd.read_csv(f"{OUT}/cons_clean.csv")
gps  = pd.read_csv(f"{OUT}/gps_clean.csv")
hh   = pd.read_csv(f"{OUT}/hh_clean.csv")

print(f"Consumption : {cons.shape[0]:,} rows")
print(f"GPS         : {gps.shape[0]:,} rows")
print(f"Household   : {hh.shape[0]:,} rows")

# ================================================
# STEP 2: MERGE ON hhid + wave
# ================================================
print("\n" + "="*55)
print("MERGING FILES")
print("="*55)

# Merge consumption + GPS
df = cons.merge(gps,
                on=['hhid','wave'],
                how='inner')
print(f"After cons + GPS  : {df.shape[0]:,} rows")

# Merge + household
df = df.merge(hh,
              on=['hhid','wave'],
              how='inner')
print(f"After + household : {df.shape[0]:,} rows")

# ================================================
# STEP 3: KEEP ONLY NEEDED COLUMNS
# ================================================
print("\n" + "="*55)
print("SELECTING COLUMNS")
print("="*55)

keep = [
    'hhid',           # household ID
    'wave',           # time period
    'log_cons_pc',    # OUTCOME variable
    'rain_shock',     # INSTRUMENT
    'head_age',       # control
    'head_male',      # control
    'hhsize',         # control
    'rururb',         # rural/urban
    'zone',           # geographic zone
    'dist_road2',     # distance to road
    'dist_market',    # distance to market
    'rain_actual',    # for reference
    'anntot_avg',     # for reference
    'hhweight'        # survey weight
]

df = df[keep].copy()
print(f"Columns kept: {df.shape[1]}")
print(f"Column names: {list(df.columns)}")

# ================================================
# STEP 4: FILTER TO BALANCED PANEL
# ================================================
print("\n" + "="*55)
print("CREATING BALANCED PANEL")
print("="*55)

# Count how many waves each household appears in
wave_counts = df.groupby('hhid')['wave'].count()

# Keep only households appearing in ALL 3 waves
balanced_hhids = wave_counts[wave_counts == 3].index
print(f"Households in all 3 waves: "
      f"{len(balanced_hhids):,}")

# Filter dataset
df_balanced = df[
    df['hhid'].isin(balanced_hhids)
].copy()

print(f"Balanced panel rows: "
      f"{df_balanced.shape[0]:,}")
print(f"\nRows per wave:")
print(df_balanced['wave']
      .value_counts()
      .sort_index())

# ================================================
# STEP 5: CHECK FOR MISSING VALUES
# ================================================
print("\n" + "="*55)
print("MISSING VALUE CHECK")
print("="*55)

missing = df_balanced.isnull().sum()
print(missing[missing > 0]
      if missing.sum() > 0
      else "No missing values! ✅")

# ================================================
# STEP 6: SUMMARY STATISTICS
# ================================================
print("\n" + "="*55)
print("KEY VARIABLE SUMMARY")
print("="*55)

key_vars = ['log_cons_pc','rain_shock',
            'head_age','head_male',
            'hhsize']

print(df_balanced[key_vars]
      .describe().round(3))

# ================================================
# STEP 7: SAVE
# ================================================
out_path = f"{OUT}/panel_clean.csv"
df_balanced.to_csv(out_path, index=False)
print(f"\n✅ Saved: {out_path}")
print(f"   Rows   : {df_balanced.shape[0]:,}")
print(f"   Columns: {df_balanced.shape[1]}")