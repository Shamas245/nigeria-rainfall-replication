# ================================================
# SCRIPT: clean_gps.py - FIXED
# ================================================

import pandas as pd
import numpy as np

BASE  = r"D:\New folder\rainfall_replication"
WAVE1 = f"{BASE}/data/raw/wave1"
WAVE2 = f"{BASE}/data/raw/wave2"
WAVE3 = f"{BASE}/data/raw/wave3"
OUT   = f"{BASE}/data/processed"

def clean_gps(filepath, wave_num,
              actual_col, lat_col,
              lon_col, road_col):
    print(f"\nCleaning Wave {wave_num} GPS...")

    df = pd.read_csv(filepath,
                     low_memory=False)
    print(f"  Loaded    : {df.shape[0]} rows")

    # Standardize column names
    df = df.rename(columns={
        lat_col   : 'lat',
        lon_col   : 'lon',
        actual_col: 'rain_actual',
        road_col  : 'dist_road2'
    })

    # Keep needed columns
    keep = ['hhid', 'lat', 'lon',
            'rain_actual', 'anntot_avg',
            'dist_road2', 'dist_market']
    df = df[keep].copy()

    df['wave'] = wave_num

    # Remove missing GPS
    before = df.shape[0]
    df = df[df['lat'].notnull()]
    df = df[df['lon'].notnull()]
    after = df.shape[0]
    print(f"  Dropped   : {before-after} "
          f"missing GPS rows")

    # Calculate rainfall shock
    rain_std = df['rain_actual'].std()
    df['rain_shock'] = (
        (df['rain_actual'] - df['anntot_avg'])
        / rain_std
    )

    print(f"  Kept      : {df.shape[0]} rows")
    print(f"  Mean rainfall actual : "
          f"{df['rain_actual'].mean():.0f} mm")
    print(f"  Mean rainfall average: "
          f"{df['anntot_avg'].mean():.0f} mm")
    print(f"  Mean rainfall shock  : "
          f"{df['rain_shock'].mean():.3f}")

    return df

# ================================================
# CLEAN ALL 3 WAVES
# ================================================
print("="*55)
print("CLEANING GPS + RAINFALL FILES")
print("="*55)

gps1 = clean_gps(
    filepath=(f"{WAVE1}/Geodata/"
              f"nga_householdgeovariables_y1.csv"),
    wave_num=1,
    actual_col='h2010_tot',
    lat_col='lat_dd_mod',
    lon_col='lon_dd_mod',
    road_col='dist_road'        # ← Wave 1 name
)

gps2 = clean_gps(
    filepath=(f"{WAVE2}/Geodata Wave 2/"
              f"nga_householdgeovars_y2.csv"),
    wave_num=2,
    actual_col='h2012_tot',
    lat_col='LAT_DD_MOD',
    lon_col='LON_DD_MOD',
    road_col='dist_road2'       # ← Wave 2 name
)

gps3 = clean_gps(
    filepath=(f"{WAVE3}/"
              f"nga_householdgeovars_y3.csv"),
    wave_num=3,
    actual_col='h2015_tot',
    lat_col='LAT_DD_MOD',
    lon_col='LON_DD_MOD',
    road_col='dist_road2'       # ← Wave 3 name
)

# ================================================
# STACK ALL 3 WAVES
# ================================================
print("\n" + "="*55)
print("STACKING ALL WAVES")
print("="*55)

gps_all = pd.concat(
    [gps1, gps2, gps3],
    ignore_index=True
)

print(f"Total rows     : {gps_all.shape[0]}")
print(f"\nRainfall shock by wave:")
print(gps_all.groupby('wave')['rain_shock']
      .agg(['mean','std','min','max'])
      .round(3))

print(f"\nSample of final dataset:")
print(gps_all[['hhid','wave',
               'rain_actual',
               'anntot_avg',
               'rain_shock']].head(6))

# ── Save ─────────────────────────────────────────
out_path = f"{OUT}/gps_clean.csv"
gps_all.to_csv(out_path, index=False)
print(f"\n✅ Saved to: {out_path}")