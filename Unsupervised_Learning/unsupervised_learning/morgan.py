import pandas as pd
import numpy as np
from rdkit import Chem
import sys, os

# --- Silence RDKit logs completely ---
from rdkit import RDLogger
import warnings
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")

# --- Add parent directory to import path ---
cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
sys.path.append(parent_dir)

from morgan_pooling import build_morgan_generator, morgan_from_smi, _SmilesStandardizer

# === Parameters ===
input_csv = "final_two_smiles_with_modulus.csv"
output_npy = "AB_concat1024.npy"

# === Morgan fingerprint generator ===
gen = build_morgan_generator(
    radius=2,
    fp_size=1024,          # Output size: 1024 bits
    use_chirality=False,
    use_features=False,
    use_bond_types=True,
    include_ring=False,
)

# === Load SMILES data ===
df = pd.read_csv(input_csv)
std = _SmilesStandardizer(use_std=True)

# === Average pooling of SMILE A / SMILE B fingerprints ===
fps = []
for i, row in df.iterrows():
    smi_a = row.get("SMILE A", "")
    smi_b = row.get("SMILE B", "")
    
    smi_a = std.canonical_smiles(smi_a)
    smi_b = std.canonical_smiles(smi_b)

    fp_a = morgan_from_smi(smi_a, gen, nbits=1024, fp_type="count")
    fp_b = morgan_from_smi(smi_b, gen, nbits=1024, fp_type="count")
    
    # === Simple average pooling ===
    fp_total = (fp_a + fp_b) / 2.0
    fps.append(fp_total)

fps = np.array(fps, dtype=np.float32)

# === Save output ===
np.save(output_npy, fps)
print(f"✅ Generated {fps.shape[0]} samples, each with length {fps.shape[1]}.")
print(f"File saved to: {output_npy}")
