# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA
from umap import UMAP
import matplotlib.colors as mcolors

# ========= Global Font: Arial + Bold =========
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['legend.fontsize'] = 23
mpl.rcParams['legend.title_fontsize'] = 23

# ========= Paths & Config =========
main_file = "all_random_smiles_AB_concat1024.npy"
cluster3_file = "Prediction-1028-ALL2-1024.npy"
candidate_file = "Prediction-1028-ALL2-candidate-1024.npy"
prediction_file = "Prediction-1028-ALL2-process.csv"
SEED = 42

# ========= 1. Load Data =========
X_main = np.load(main_file)
X_cluster3 = np.load(cluster3_file)
X_cand = np.load(candidate_file)

print(f"[OK] Main data: {main_file} -> {X_main.shape}")
print(f"[OK] Cluster 3 data: {cluster3_file} -> {X_cluster3.shape}")
print(f"[OK] Candidate data: {candidate_file} -> {X_cand.shape}")

predictions = pd.read_csv(prediction_file)
print(f"[OK] Prediction file loaded: {prediction_file} -> {predictions.shape}")
young_modulus = predictions["Young's Modulus (kPa)"]

# Combine all
X_all = np.vstack([X_main, X_cluster3, X_cand])
n_main = len(X_main)
n_cluster3 = len(X_cluster3)
n_cand = len(X_cand)

# ========= 2. PCA =========
pca = PCA(n_components=64, random_state=SEED)
X_pca = pca.fit_transform(X_all)
print(f"[OK] PCA done -> shape={X_pca.shape}, variance={pca.explained_variance_ratio_.sum():.3f}")

# ========= 3. UMAP =========
umap = UMAP(
    n_components=2,
    n_neighbors=30,
    min_dist=0.1,
    metric="euclidean",
    random_state=SEED
)
X_umap = umap.fit_transform(X_pca)
print(f"[OK] UMAP projection done -> shape={X_umap.shape}")

# ========= 4. Split Coordinates =========
XY_main = X_umap[:n_main]
XY_cluster3 = X_umap[n_main:n_main + n_cluster3]
XY_cand = X_umap[n_main + n_cluster3:]

# ========= 5. Plotting =========
plt.figure(figsize=(12, 7), dpi=1000)

# Main Data (Gray)
plt.scatter(
    XY_main[:, 0], XY_main[:, 1],
    s=200, alpha=0.7, color="lightgray",
    label="Main Data"
)

# Cluster 3 Gradient
norm = plt.Normalize(vmin=young_modulus.min(), vmax=young_modulus.max())
cmap = plt.cm.Blues

scatter_cluster3 = plt.scatter(
    XY_cluster3[:, 0], XY_cluster3[:, 1],
    c=young_modulus, cmap=cmap, norm=norm,
    s=400, alpha=0.9, edgecolors="black", linewidths=0.4,
    label="Candidate set"
)

# Colorbar (in Arial + Bold)
cbar = plt.colorbar(scatter_cluster3, ax=plt.gca(), fraction=0.046, pad=0.02)
cbar.set_label("Young's Modulus (kPa)", fontsize=20, fontweight='bold')
cbar.ax.tick_params(labelsize=20, width=1.5)

# Candidate Pink Stars
plt.scatter(
    XY_cand[:, 0], XY_cand[:, 1],
    marker="*", s=800, c="#F19AD5", alpha=0.8,
    edgecolors="black", linewidths=0.5,
    label="Candidate Components"
)

# Axes labels (Arial + Bold)
plt.xlabel("UMAP-1", fontsize=24, fontweight='bold')
plt.ylabel("UMAP-2", fontsize=24, fontweight='bold')

# Remove ticks
plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# Border
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

# Legend (Arial + Bold)
plt.legend(loc='upper right', fontsize=18, prop={'weight': 'bold'})

plt.tight_layout()
# ========= Save PNG =========
output_png = "umap_candidate_visualization.png"
plt.savefig(output_png, dpi=1000, bbox_inches="tight")
print(f"\n📁 Saved UMAP figure to: {output_png}")

plt.show()

# ========= 6. Export Candidate Coordinates =========
cand_df = pd.DataFrame(XY_cand, columns=["UMAP-1", "UMAP-2"])
cand_df.index = [f"Candidate_{i+1}" for i in range(n_cand)]
cand_df.to_csv("candidate_umap_coordinates.csv", encoding="utf-8-sig")

print("\n✅ Candidate UMAP coordinates saved to candidate_umap_coordinates.csv")
print(cand_df)
