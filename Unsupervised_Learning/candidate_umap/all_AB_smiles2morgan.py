# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import itertools
import random
from rdkit import Chem
from rdkit import RDLogger
import sys, os

# --- Add parent directory to import path ---
cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
sys.path.append(parent_dir)



from morgan_pooling import build_morgan_generator, morgan_from_smi, _SmilesStandardizer

# Disable RDKit logs
RDLogger.DisableLog('rdApp.*')


# =========================================================
# === Step 1: Extract SMILES from swelling_ratio.csv & youngs_modulus.csv ===
# =========================================================

input_files = ["../database/swelling_ratio.csv", "../database/youngs_modulus.csv"]
smiles_list = []

for file in input_files:
    df = pd.read_csv(file, encoding="gbk")
    for col in ["SMILE A", "SMILE B", "SMILE C"]:
        if col in df.columns:
            smiles_list.extend(df[col].dropna().astype(str).tolist())

# Count frequency
smiles_series = pd.Series(smiles_list)
count_df = smiles_series.value_counts().reset_index()
count_df.columns = ["SMILES", "count"]

count_df.to_csv("smiles_count.csv", index=False, encoding="utf-8-sig")
print("✅ Step 1 Done: Extracted SMILES and saved to smiles_count.csv")


# =========================================
# === Step 2: Random unique SMILES pairs ===
# =========================================

df = pd.read_csv("smiles_count.csv")
smiles_unique = df["SMILES"].dropna().unique().tolist()
print(f"[OK] Loaded {len(smiles_unique)} unique SMILES.")

# Generate all unique combinations (A, B)
pairs = list(itertools.combinations(smiles_unique, 2))
random.shuffle(pairs)

pairs_df = pd.DataFrame(pairs, columns=["SMILE A", "SMILE B"])
pairs_df.to_csv("smiles_count_random_2.csv", index=False, encoding="utf-8-sig")

print(f"✅ Step 2 Done: Generated {len(pairs_df)} SMILES pairs → smiles_count_random_2.csv")


# ==================================================
# === Step 3: Compute Morgan Fingerprints for AB ===
# ==================================================

input_csv = "smiles_count_random_2.csv"
output_npy = "all_random_smiles_AB_concat1024.npy"

# Morgan generator
gen = build_morgan_generator(
    radius=2,
    fp_size=1024,
    use_chirality=False,
    use_features=False,
    use_bond_types=True,
    include_ring=False,
)

# Load data
df = pd.read_csv(input_csv)
std = _SmilesStandardizer(use_std=True)

fps = []
for i, row in df.iterrows():
    smi_a = std.canonical_smiles(row.get("SMILE A", ""))
    smi_b = std.canonical_smiles(row.get("SMILE B", ""))

    fp_a = morgan_from_smi(smi_a, gen, nbits=1024, fp_type="count")
    fp_b = morgan_from_smi(smi_b, gen, nbits=1024, fp_type="count")

    fp_total = (fp_a + fp_b) / 2.0
    fps.append(fp_total)

fps = np.array(fps, dtype=np.float32)
np.save(output_npy, fps)

print(f"✅ Step 3 Done: Generated Morgan fingerprints → {output_npy}")
print(f"   Total samples: {fps.shape[0]}, Fingerprint dim: {fps.shape[1]}")
