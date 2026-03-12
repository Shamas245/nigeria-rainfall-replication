# ================================================
# SCRIPT: clean_consumption.py
# PURPOSE: Clean and stack consumption data
#          across all 3 waves
# AUTHOR: Shamas Liaqat
# DATE: March 2026
# ================================================

import pandas as pd

# ── PATHS ────────────────────────────────────────
BASE  = r"D:\New folder\rainfall_replication"
WAVE1 = f"{BASE}/data/raw/wave1"
WAVE2 = f"{BASE}/data/raw/wave2"
WAVE3 = f"{BASE}/data/raw/wave3"
OUT   = f"{BASE}/data/processed"

# ── PPP DEFLATORS ────────────────────────────────
# Base year = 2010
# Source: World Bank
PPP = {1: 1.00,
       2: 1.23,
       3: 1.52}

# ── COLUMNS TO KEEP ──────────────────────────────
KEEP_COLS = ['hhid','totcons',
             'hhsize','hhweight',
             'rururb','zone']

# ================================================
# FUNCTION: Clean one wave's consumption file
# ================================================
def clean_cons(filepath, wave_num):
    print(f"\nCleaning Wave {wave_num}...")

    # STEP 1: Load file
    df = pd.read_csv(filepath,
                     low_memory=False)
    print(f"  Loaded    : {df.shape[0]} rows")

    # STEP 2: Keep only needed columns
    df = df[KEEP_COLS].copy()

    # STEP 3: Add wave identifier
    df['wave'] = wave_num

    # STEP 4: Remove missing consumption
    before = df.shape[0]
    df = df[df['totcons'].notnull()]
    df = df[df['totcons'] > 0]
    after = df.shape[0]
    print(f"  Dropped   : {before-after} "
          f"missing/zero consumption rows")

    # STEP 5: Convert to real 2010 PPP terms
    df['totcons_real'] = (df['totcons']
                          / PPP[wave_num])

    # STEP 6: Calculate per capita consumption
    df['cons_pc'] = (df['totcons_real']
                     / df['hhsize'])

    # STEP 7: Log transformation
    # (paper uses log consumption)
    import numpy as np
    df['log_cons_pc'] = np.log(df['cons_pc'])

    print(f"  Kept      : {df.shape[0]} rows")
    print(f"  Mean real cons/capita: "
          f"{df['cons_pc'].mean():,.0f} Naira")

    return df

# ================================================
# CLEAN ALL 3 WAVES
# ================================================
print("="*55)
print("CLEANING CONSUMPTION FILES")
print("="*55)

cons1 = clean_cons(
    f"{WAVE1}/cons_agg_wave1_visit1.csv", 1)

cons2 = clean_cons(
    f"{WAVE2}/cons_agg_wave2_visit1.csv", 2)

cons3 = clean_cons(
    f"{WAVE3}/cons_agg_wave3_visit1.csv", 3)

# ================================================
# STACK ALL 3 WAVES
# ================================================
print("\n" + "="*55)
print("STACKING ALL WAVES")
print("="*55)

cons_all = pd.concat(
    [cons1, cons2, cons3],
    ignore_index=True
)

print(f"Total rows     : {cons_all.shape[0]}")
print(f"Total columns  : {cons_all.shape[1]}")
print(f"\nRows per wave:")
print(cons_all['wave'].value_counts().sort_index())

print(f"\nConsumption per capita by wave:")
print(cons_all.groupby('wave')['cons_pc']
      .mean().round(0))

print(f"\nSample of final dataset:")
print(cons_all[['hhid','wave',
                'totcons_real',
                'cons_pc',
                'log_cons_pc']].head(6))

# ================================================
# SAVE CLEANED FILE
# ================================================
out_path = f"{OUT}/cons_clean.csv"
cons_all.to_csv(out_path, index=False)
print(f"\n✅ Saved to: {out_path}")