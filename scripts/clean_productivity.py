# ================================================
# SCRIPT: clean_productivity.py (v2 - FIXED)
# PURPOSE: Calculate agricultural productivity
#          (Naira per Hectare) for all 3 waves
# AUTHOR: Shamas Liaqat
# DATE: March 2026
#
# CHANGES FROM v1:
#   - Wave 1 & 2 now use crosswalk-based prices
#     (prices_w1.csv / prices_w2.csv) built by
#     build_price_crosswalk.py
#   - Wave 3 unchanged (uses direct crop values)
# ================================================

import pandas as pd
import numpy as np

# ── PATHS ────────────────────────────────────────
BASE  = r"D:\New folder\rainfall_replication"
WAVE1 = f"{BASE}/data/raw/wave1"
WAVE2 = f"{BASE}/data/raw/wave2"
WAVE3 = f"{BASE}/data/raw/wave3"
OUT   = f"{BASE}/data/processed"

AG1   = f"{WAVE1}/Post Harvest Wave 1/Agriculture"
AG2   = f"{WAVE2}/Post Harvest Wave 2/Agriculture"

# ── LAND UNIT CONVERSION TO HECTARES ─────────────
LAND_CONV = {
    5: 0.4047,   # Acres  → Hectares
    6: 1.0000,   # Hectares
    7: 0.0001    # Sq meters → Hectares
}

# ================================================
# STEP 1: LOAD CONVERSION FILES
# ================================================
print("="*55)
print("STEP 1: LOADING CONVERSION FILES")
print("="*55)

conv1 = pd.read_csv(f"{WAVE1}/w1agnsconversion.csv", low_memory=False)
conv1 = conv1.rename(columns={
    'agcropid'  : 'cropcode',
    'nscode'    : 'unit_cd',
    'conversion': 'conv_kg'
})
print(f"Conv1 rows: {conv1.shape[0]}")

conv2 = pd.read_csv(f"{WAVE2}/w2agnsconversion.csv", low_memory=False)
conv2 = conv2.rename(columns={
    'nscode'    : 'unit_cd',
    'conversion': 'conv_kg'
})
print(f"Conv2 rows: {conv2.shape[0]}")

conv3 = pd.read_csv(f"{WAVE3}/ag_conv_w3.csv", low_memory=False)
conv3 = conv3.rename(columns={
    'crop_cd'      : 'cropcode',
    'conv_national': 'conv_kg'
})
print(f"Conv3 rows: {conv3.shape[0]}")

# ================================================
# STEP 2: LOAD UNIVERSAL PRICES (Wave 3 per-KG)
# ================================================
# Using Wave 3 prices for all waves as a common
# deflator. This eliminates unit inconsistency
# across waves and is methodologically clean —
# productivity is measured in constant 2015 prices.
# ================================================
print("\n" + "="*55)
print("STEP 2: LOADING UNIVERSAL PRICES")
print("="*55)

p_comm = pd.read_csv(f"{OUT}/prices_universal.csv")
p_nat  = pd.read_csv(f"{OUT}/prices_nat_universal.csv")

# Use same price file for all 3 waves
p1 = p_comm.rename(columns={'price_pkg': 'price_w1'})
p2 = p_comm.rename(columns={'price_pkg': 'price_w2'})
p3 = p_comm.copy()

p1_nat = p_nat.rename(columns={'price_nat_pkg': 'price_nat_w1'})
p2_nat = p_nat.rename(columns={'price_nat_pkg': 'price_nat_w2'})
p3_nat = p_nat.rename(columns={'price_nat_pkg': 'price_nat'})

print(f"Universal prices: {p_comm.shape[0]} community-crop pairs")
print(f"Crops covered   : {p_nat.shape[0]}")

# ================================================
# STEP 3: LOAD CROP FILES
# ================================================
print("\n" + "="*55)
print("STEP 3: LOADING CROP FILES")
print("="*55)

# ── Wave 1 ───────────────────────────────────────
crop1 = pd.read_csv(f"{AG1}/secta3_harvestw1.csv", low_memory=False)
crop1 = crop1.rename(columns={
    'sa3q2' : 'cropcode',
    'sa3q12': 'qty',
    'sa3q13': 'unit_cd',
    'sa3q17': 'crop_value_direct',
    'sa3q5a': 'land_area',
    'sa3q5b': 'land_unit'
})
crop1 = crop1[['hhid','ea','cropcode','qty','unit_cd',
               'crop_value_direct','land_area','land_unit']]
crop1['wave'] = 1
print(f"Wave 1 crops: {crop1.shape[0]} rows")

# ── Wave 2 ───────────────────────────────────────
crop2 = pd.read_csv(f"{AG2}/secta3_harvestw2.csv", low_memory=False)
crop2 = crop2.rename(columns={
    'sa3q12': 'qty',
    'sa3q13': 'unit_cd',
    'sa3q17': 'crop_value_direct',
    'sa3q5a': 'land_area',
    'sa3q5b': 'land_unit'
})
crop2 = crop2[['hhid','ea','cropcode','qty','unit_cd',
               'crop_value_direct','land_area','land_unit']]
crop2['wave'] = 2
print(f"Wave 2 crops: {crop2.shape[0]} rows")

# ── Wave 3 ───────────────────────────────────────
crop3a = pd.read_csv(f"{WAVE3}/secta3i_harvestw3.csv", low_memory=False)
crop3b = pd.read_csv(f"{WAVE3}/secta3ii_harvestw3.csv", low_memory=False)
crop3  = pd.concat([crop3a, crop3b], ignore_index=True)
crop3  = crop3.rename(columns={
    'sa3iq6i' : 'qty',
    'sa3iq6ii': 'unit_cd',
    'sa3iq6a' : 'crop_value_direct',
    'sa3iq5a' : 'land_area',
    'sa3iq5b' : 'land_unit'
})
crop3 = crop3[['hhid','ea','cropcode','qty','unit_cd',
               'crop_value_direct','land_area','land_unit']]
crop3['wave'] = 3
print(f"Wave 3 crops: {crop3.shape[0]} rows (both harvest files)")

# ================================================
# STEP 4: CONVERT HARVEST QUANTITY TO KG
# ================================================
print("\n" + "="*55)
print("STEP 4: CONVERTING HARVEST TO KG")
print("="*55)

def convert_to_kg(df, conv_df, wave_name):
    df = df.copy()

    # Track A: simple units (1=KG, 2=Gram, 3=Litre)
    conditions = [
        df['unit_cd'] == 1,
        df['unit_cd'] == 2,
        df['unit_cd'] == 3,
    ]
    choices = [df['qty'], df['qty'] / 1000, df['qty']]
    df['harvest_kg'] = np.select(conditions, choices, default=np.nan)

    # Track B: container units (unit_cd >= 10)
    container_mask = df['unit_cd'] >= 10
    if container_mask.sum() > 0:
        container_rows = df[container_mask].merge(
            conv_df[['cropcode','unit_cd','conv_kg']],
            on=['cropcode','unit_cd'], how='left'
        )
        df.loc[container_mask, 'harvest_kg'] = (
            container_rows['qty'] * container_rows['conv_kg']
        ).values

    valid = df['harvest_kg'].notnull().sum()
    print(f"{wave_name}: {valid}/{df.shape[0]} rows "
          f"({valid/df.shape[0]*100:.1f}%) have harvest_kg")
    return df

crop1 = convert_to_kg(crop1, conv1, "Wave 1")
crop2 = convert_to_kg(crop2, conv2, "Wave 2")
crop3 = convert_to_kg(crop3, conv3, "Wave 3")

# ================================================
# STEP 5: CONVERT LAND AREA TO HECTARES
# ================================================
for df, name in [(crop1,'Wave 1'),(crop2,'Wave 2'),(crop3,'Wave 3')]:
    df['land_ha'] = df['land_unit'].map(LAND_CONV) * df['land_area']
    
    # Cap at 10 Ha per plot — values above this are data entry errors
    # Standard LSMS cleaning practice for Nigerian smallholder data
    df.loc[df['land_ha'] > 10, 'land_ha'] = np.nan
    
    valid = df['land_ha'].notnull().sum()
    total = df.shape[0]
    print(f"{name}: {valid}/{total} ({valid/total*100:.1f}%) "
          f"standard land units")

# ================================================
# STEP 6: ATTACH PRICES AND CALCULATE CROP VALUE
# ================================================
print("\n" + "="*55)
print("STEP 6: CALCULATING CROP VALUE")
print("="*55)

def calc_value_w1w2(df, price_comm, price_nat, wave_name):
    df = df.copy()

    # Community price column name
    price_col = [c for c in price_comm.columns
                 if c not in ['ea','cropcode']][0]
    nat_col   = [c for c in price_nat.columns
                 if c != 'cropcode'][0]

    df = df.merge(price_comm, on=['ea','cropcode'], how='left')
    df = df.merge(price_nat,  on='cropcode',        how='left')

    df['price_final'] = df[price_col].fillna(df[nat_col])
    df['crop_value']  = df['harvest_kg'] * df['price_final']

    valid    = df['crop_value'].notnull().sum()
    no_price = df['price_final'].isna().sum()
    print(f"{wave_name}: {valid}/{df.shape[0]} rows have crop_value "
          f"({valid/df.shape[0]*100:.1f}%)")
    print(f"  Rows with no price: {no_price}")
    return df

def calc_value_w3(df, price_comm, price_nat):
    df = df.copy()
    df = df.merge(price_comm, on=['ea','cropcode'], how='left')
    df = df.merge(price_nat,  on='cropcode',        how='left')
    df['price_final'] = df['price_pkg'].fillna(df['price_nat'])

    direct_mask = (df['crop_value_direct'].notnull() &
                   (df['crop_value_direct'] > 0))
    df['crop_value'] = np.where(
        direct_mask,
        df['crop_value_direct'],
        df['harvest_kg'] * df['price_final']
    )
    direct_used = direct_mask.sum()
    total_valid = df['crop_value'].notnull().sum()
    print(f"Wave 3: {direct_used} direct values, "
          f"{total_valid - direct_used} used qty×price, "
          f"{df.shape[0] - total_valid} missing")
    return df

crop1 = calc_value_w1w2(crop1, p1, p1_nat, "Wave 1")
crop2 = calc_value_w1w2(crop2, p2, p2_nat, "Wave 2")
crop3 = calc_value_w3(crop3, p3, p3_nat)

# ================================================
# STEP 7: CALCULATE PRODUCTIVITY PER HECTARE
# ================================================
print("\n" + "="*55)
print("STEP 7: CALCULATING PRODUCTIVITY")
print("="*55)

def calc_productivity(df, wave_name):
    df = df.copy()
    df = df[df['crop_value'].notnull() &
            df['land_ha'].notnull()    &
            (df['crop_value'] > 0)     &
            (df['land_ha'] > 0)        &
            np.isfinite(df['crop_value'] / df['land_ha'])]

    hh = df.groupby(['hhid','wave']).agg(
        total_value=('crop_value','sum'),
        total_land =('land_ha',   'sum')
    ).reset_index()

    hh['productivity']     = hh['total_value'] / hh['total_land']
    hh['log_productivity'] = np.log(hh['productivity'])

    # Trim top/bottom 1% outliers
    lo = hh['log_productivity'].quantile(0.01)
    hi = hh['log_productivity'].quantile(0.99)
    hh = hh[(hh['log_productivity'] >= lo) &
            (hh['log_productivity'] <= hi)]

    print(f"{wave_name}: {hh.shape[0]} households")
    print(f"  Median  : {hh['productivity'].median():,.0f} Naira/Ha")
    print(f"  Log mean: {hh['log_productivity'].mean():.3f}  "
          f"sd: {hh['log_productivity'].std():.3f}")
    return hh

prod1 = calc_productivity(crop1, "Wave 1")
prod2 = calc_productivity(crop2, "Wave 2")
prod3 = calc_productivity(crop3, "Wave 3")

# ================================================
# STEP 8: STACK ALL WAVES AND SAVE
# ================================================
print("\n" + "="*55)
print("STEP 8: STACKING AND SAVING")
print("="*55)

prod_all = pd.concat([prod1, prod2, prod3], ignore_index=True)

print(f"Total rows : {prod_all.shape[0]}")
print(f"By wave:")
print(prod_all['wave'].value_counts().sort_index())
print(f"\nSample:")
print(prod_all.head())

prod_all.to_csv(f"{OUT}/productivity_clean.csv", index=False)
print(f"\nSaved → {OUT}/productivity_clean.csv")

