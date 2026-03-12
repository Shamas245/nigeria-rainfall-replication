import pandas as pd

BASE = r"D:\New folder\rainfall_replication"
df = pd.read_csv(f"{BASE}/data/processed/panel_with_assets.csv")

print("Zone distribution:")
print(df['zone'].value_counts().sort_index())

print("\nZone labels from Wave 1 consumption file:")
cons = pd.read_csv(
    f"{BASE}/data/raw/wave1/"
    f"cons_agg_wave1_visit1.csv"
)
print(cons[['zone']].value_counts()
      .sort_index())

# Check if zone has labels
if 'zonename' in cons.columns:
    print(cons[['zone','zonename']]
          .drop_duplicates()
          .sort_values('zone'))