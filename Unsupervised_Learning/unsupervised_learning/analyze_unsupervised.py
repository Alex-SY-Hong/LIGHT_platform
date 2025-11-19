import pandas as pd
import glob
import os

# === Folder containing cluster files ===
cluster_dir = "clusters"

# Find all cluster_*.csv inside clusters/
file_list = sorted(glob.glob(os.path.join(cluster_dir, "cluster_*.csv")))

results = []

for file in file_list:
    df = pd.read_csv(file)

    # Total rows (excluding header)
    c = len(df)

    # Count: Swelling Ratio >= 9
    if "Swelling Ratio (times)" in df.columns:
        a = df["Swelling Ratio (times)"].ge(9).sum()
    else:
        a = 0

    # Count: Young's Modulus in [100, 2000]
    if "Young's Modulus" in df.columns:
        b = df["Young's Modulus"].between(100, 2000).sum()
    else:
        b = 0

    # Probabilities
    d = a / c if c > 0 else 0
    e = b / c if c > 0 else 0

    # File name only (without folder path)
    name = os.path.basename(file)

    results.append([name, c, a, b, d, e])

# Create output DataFrame
out_df = pd.DataFrame(results, columns=[
    "cluster_file",
    "total",
    "SR_ge_9",
    "YM_100_2000",
    "prob_SR",
    "prob_YM"
])

print(out_df)

# Save result inside clusters folder
output_path = os.path.join(cluster_dir, "cluster_statistics.csv")
out_df.to_csv(output_path, index=False)

print(f"\n✅ Done! Statistics saved to {output_path}")
