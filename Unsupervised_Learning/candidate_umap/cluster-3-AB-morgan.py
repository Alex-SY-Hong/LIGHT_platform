import pandas as pd
import numpy as np
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


# ============================
# Function: Process one CSV
# ============================

def process_file(input_raw):
    print(f"\n=== Processing File: {input_raw} ===")

    # 1. Output file names
    processed_csv = input_raw.replace(".csv", "-process.csv")
    output_npy = input_raw.replace(".csv", "-1024.npy")

    # ============================
    # STEP A: Convert Young's Modulus
    # ============================

    df = pd.read_csv(input_raw)
    df["Young's Modulus (kPa)"] = 10 ** df["Young's Modulus (kPa) log10"]
    df = df.drop(columns=["Young's Modulus (kPa) log10"])

    df.to_csv(processed_csv, index=False)
    print(f"   ➤ Saved processed CSV → {processed_csv}")

    # ============================
    # STEP B: Generate Morgan FP
    # ============================

    print("   ➤ Generating Morgan fingerprints...")

    gen = build_morgan_generator(
        radius=2,
        fp_size=1024,
        use_chirality=False,
        use_features=False,
        use_bond_types=True,
        include_ring=False,
    )

    df = pd.read_csv(processed_csv)
    std = _SmilesStandardizer(use_std=True)

    fps = []
    for _, row in df.iterrows():
        smi_a = std.canonical_smiles(row.get("SMILE A", ""))
        smi_b = std.canonical_smiles(row.get("SMILE B", ""))

        fp_a = morgan_from_smi(smi_a, gen, nbits=1024, fp_type="count")
        fp_b = morgan_from_smi(smi_b, gen, nbits=1024, fp_type="count")

        fp_total = (fp_a + fp_b) / 2.0
        fps.append(fp_total)

    fps = np.array(fps, dtype=np.float32)
    np.save(output_npy, fps)

    print(f"   ➤ Saved FP array: {output_npy}")
    print(f"   ➤ Shape = {fps.shape}")

    print(f"🎉 Completed: {input_raw}\n")



# ============================
# Run for BOTH files
# ============================

file_list = [
    "Prediction-1028-ALL2.csv",
    "Prediction-1028-ALL2-candidate.csv"
]

for f in file_list:
    process_file(f)

print("======================================")
print("🎉 All files processed successfully!")
print("======================================")
