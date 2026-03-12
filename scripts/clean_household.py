# ================================================
# SCRIPT: clean_household.py
# PURPOSE: Clean household characteristics
#          across all 3 waves
# AUTHOR: Shamas Liaqat
# DATE: March 2026
# ================================================

import pandas as pd
import numpy as np

# ── PATHS ────────────────────────────────────────
BASE  = r"D:\New folder\rainfall_replication"
WAVE1 = f"{BASE}/data/raw/wave1"
WAVE2 = f"{BASE}/data/raw/wave2"
WAVE3 = f"{BASE}/data/raw/wave3"
OUT   = f"{BASE}/data/processed"

HH1 = f"{WAVE1}/Post Harvest Wave 1/Household"
HH2 = f"{WAVE2}/Post Harvest Wave 2/Household"
HH3 = WAVE3

# ================================================
# FUNCTION: Clean one wave's household file
# ================================================
def clean_hh(filepath, wave_num, 
             encoding='utf-8'):
    print(f"\nCleaning Wave {wave_num} "
          f"Household...")

    # STEP 1: Load file
    df = pd.read_csv(filepath,
                 low_memory=False,
                 encoding=encoding)
    print(f"  Loaded    : {df.shape[0]} rows")

    # STEP 2: Keep only household HEAD
    # s1q3 = relationship to head
    # 1 = household head
    df = df[df['s1q3'] == 1].copy()
    print(f"  Heads only: {df.shape[0]} rows")

    # STEP 3: Keep needed columns
    # s1q2 = gender (1=male, 2=female)
    # s1q4 = age
    keep = ['hhid', 's1q2', 's1q4']
    df = df[keep].copy()

    # STEP 4: Rename for clarity
    df = df.rename(columns={
        's1q2': 'head_gender',
        's1q4': 'head_age'
    })

    # STEP 5: Clean age
    # Remove sentinel values (999, 998)
    before = df.shape[0]
    df = df[df['head_age'] < 120]
    df = df[df['head_age'] > 15]
    after = df.shape[0]
    print(f"  Dropped   : {before-after} "
          f"invalid age rows")

    # STEP 6: Create gender dummy
    # 1 = male head, 0 = female head
    df['head_male'] = (
        (df['head_gender'] == 1)
        .astype(int)
    )

    # STEP 7: Add wave
    df['wave'] = wave_num

    print(f"  Kept      : {df.shape[0]} rows")
    print(f"  Mean age  : "
          f"{df['head_age'].mean():.1f} years")
    print(f"  Male head : "
          f"{df['head_male'].mean()*100:.1f}%")

    return df

# ================================================
# CLEAN ALL 3 WAVES
# ================================================
print("="*55)
print("CLEANING HOUSEHOLD FILES")
print("="*55)

hh1 = clean_hh(
    f"{HH1}/sect1_harvestw1.csv", 1)

hh2 = clean_hh(
    f"{HH2}/sect1_harvestw2.csv", 2)

hh3 = clean_hh(
    f"{HH3}/sect1_harvestw3.csv", 3,
    encoding='latin1')

# ================================================
# STACK ALL 3 WAVES
# ================================================
print("\n" + "="*55)
print("STACKING ALL WAVES")
print("="*55)

hh_all = pd.concat(
    [hh1, hh2, hh3],
    ignore_index=True
)

print(f"Total rows     : {hh_all.shape[0]}")
print(f"\nHousehold head stats by wave:")
print(hh_all.groupby('wave')
      .agg(
          mean_age=('head_age','mean'),
          pct_male=('head_male','mean')
      ).round(2))

print(f"\nSample:")
print(hh_all[['hhid','wave',
              'head_age',
              'head_male']].head(6))

# ── Save ─────────────────────────────────────────
out_path = f"{OUT}/hh_clean.csv"
hh_all.to_csv(out_path, index=False)
print(f"\n✅ Saved to: {out_path}") 