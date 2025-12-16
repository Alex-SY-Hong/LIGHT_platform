import pandas as pd
import numpy as np

# ========== STEP 1: Read and merge two CSV files ==========
df1 = pd.read_csv("../database/youngs_modulus.csv", encoding="gbk")
df2 = pd.read_csv("../database/swelling_ratio.csv", encoding="gbk")

# Add missing columns to df2
for col in ["Young's Modulus (kPa) log10", "Source"]:
    if col not in df2.columns:
        df2[col] = pd.NA

# Add missing column to df1
if "Swelling Ratio (times)" not in df1.columns:
    df1["Swelling Ratio (times)"] = pd.NA

# Merge them
df = pd.concat([df1, df2], ignore_index=True)

# Reorder columns
df = df[
    ["SMILE A",
     "SMILE B",
     "SMILE C",
     "Young's Modulus (kPa) log10",
     "Swelling Ratio (times)",
     "Source"]
]

# ========== STEP 2: Filter rows that contain exactly two SMILES strings ==========
def count_non_empty(x):
    return sum(
        pd.notna(x[col]) and str(x[col]).strip() != ""
        for col in ["SMILE A", "SMILE B", "SMILE C"]
    )

df_two = df[df.apply(count_non_empty, axis=1) == 2]

# ========== STEP 3: Add Young's Modulus (kPa) ==========
df_two["Young's Modulus (kPa)"] = df_two["Young's Modulus (kPa) log10"].apply(
    lambda x: 10**x if pd.notna(x) else np.nan
)

# ========== STEP 4: Save final output ==========
df_two.to_csv("final_two_smiles_with_modulus.csv", index=False, encoding="utf-8")

print("Processing completed! Final file generated: final_two_smiles_with_modulus.csv")
print("Number of rows in final output:", len(df_two))
