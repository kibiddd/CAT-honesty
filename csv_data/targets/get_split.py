import pandas as pd
import json
import random

# ── CONFIG ──────────────────────────────────────────────────────────────
INPUT_FILE  = "csv_data/targets/provided_facts_with_targets.csv"
OUTPUT_CSV  = "sampled_adv_training.csv"
TOWARD_JSON = "toward_targets.json"
AWAY_JSON   = "away_targets.json"
N_SAMPLES   = 100
RANDOM_SEED = 42          # set to None for a different sample each run

# ── 1. LOAD & SAMPLE ─────────────────────────────────────────────────────
df = pd.read_csv(INPUT_FILE)

sampled = df.sample(n=N_SAMPLES, random_state=RANDOM_SEED).reset_index(drop=True)

# ── 2. ADD BehaviorID COLUMN ─────────────────────────────────────────────
sampled.insert(0, "BehaviorID", [f"adv_training_{i+1}" for i in range(N_SAMPLES)])

# ── 3. SAVE SAMPLED CSV ──────────────────────────────────────────────────
sampled.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(sampled)} rows → {OUTPUT_CSV}")

# ── 4. EXTRACT TARGET COLUMNS ────────────────────────────────────────────
toward_cols = [c for c in sampled.columns if c.startswith("toward_target_")]
away_cols   = [c for c in sampled.columns if c.startswith("away_target_")]

def build_target_dict(df: pd.DataFrame, cols: list[str]) -> dict:
    """Return {BehaviorID: [non-null target strings]}"""
    result = {}
    for _, row in df.iterrows():
        targets = [row[c] for c in cols if pd.notna(row[c]) and str(row[c]).strip()]
        result[row["BehaviorID"]] = targets
    return result

toward_dict = build_target_dict(sampled, toward_cols)
away_dict   = build_target_dict(sampled, away_cols)

# ── 5. SAVE JSON FILES ───────────────────────────────────────────────────
with open(TOWARD_JSON, "w", encoding="utf-8") as f:
    json.dump(toward_dict, f, indent=2, ensure_ascii=False)

with open(AWAY_JSON, "w", encoding="utf-8") as f:
    json.dump(away_dict, f, indent=2, ensure_ascii=False)

print(f"Saved toward_targets → {TOWARD_JSON}")
print(f"Saved away_targets   → {AWAY_JSON}")

# ── 6. QUICK SANITY CHECK ────────────────────────────────────────────────
first_id = list(toward_dict.keys())[0]
print(f"\nSample entry ({first_id}):")
print(f"  toward_targets : {toward_dict[first_id]}")
print(f"  away_targets   : {away_dict[first_id]}")