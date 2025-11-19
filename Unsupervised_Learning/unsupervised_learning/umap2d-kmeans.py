# -*- coding: utf-8 -*-
# KMeans + Visualization + Ellipse + CSV split (using saved UMAP result)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse

# ========= Global Font: Arial + Bold =========
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['legend.fontsize'] = 18
mpl.rcParams['legend.title_fontsize'] = 18

# ========= File Paths =========
NPY_UMAP_PATH = r"umap2d.npy"  # <-- Only read this!
CSV_PATH = r"final_two_smiles_with_modulus.csv"

# ========= KMeans Parameters (unchanged) =========
K = 6
SEED = 42
print(f"[INFO] KMeans clustering with K={K}, random_state={SEED}")

# ========= Load UMAP 2D coordinates =========
XY = np.load(NPY_UMAP_PATH)
print(f"[OK] Loaded UMAP 2D coordinates: {NPY_UMAP_PATH} -> shape={XY.shape}")

# ========= Load CSV (same index order) =========
df = pd.read_csv(CSV_PATH)
print(f"[OK] Loaded CSV file: {CSV_PATH} -> shape={df.shape}")

if len(df) != len(XY):
    raise ValueError("CSV row count does not match UMAP sample count.")
else:
    print("✅ CSV rows match UMAP samples.")

# ========= KMeans on UMAP =========
km = KMeans(n_clusters=K, n_init=10, random_state=SEED)
labels = km.fit_predict(XY)

print(f"[OK] KMeans clustering completed, K={K}")

# ========= Count samples =========
unique_labels, counts = np.unique(labels, return_counts=True)
print("\n===== Cluster Sample Counts =====")
for lbl, cnt in zip(unique_labels, counts):
    print(f"Cluster {lbl}: {cnt} samples")
print("=================================\n")

# ========= Save CSV by cluster =========
os.makedirs("clusters-from-umap", exist_ok=True)

for lbl in unique_labels:
    df_sub = df[labels == lbl]
    save_path = f"clusters-from-umap/cluster_{lbl}.csv"
    df_sub.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved cluster {lbl} -> {save_path} ({len(df_sub)} rows)")

print("\n✅ All clusters saved in folder: clusters-from-umap/")

# ========= Visualization =========
plt.figure(figsize=(7, 6), dpi=1000)

scatter = plt.scatter(
    XY[:, 0], XY[:, 1],
    c=labels,
    s=200,
    alpha=0.9,
    cmap=mcolors.LinearSegmentedColormap.from_list(
        "blue_pink_orange_deep",
        [(20/255, 40/255, 100/255),
         (110/255, 60/255, 160/255),
         (230/255, 80/255, 40/255)],
        N=256
    ),
    marker="o",
    edgecolors="black",
    linewidths=0.4
)

colors = scatter.cmap(scatter.norm(unique_labels))

# ---- Draw Ellipses ----
for i, color in zip(unique_labels, colors):
    pts = XY[labels == i]
    if len(pts) < 5:
        continue

    cov = np.cov(pts, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 5 * np.sqrt(vals)

    ellipse = Ellipse(
        xy=np.mean(pts, axis=0),
        width=width,
        height=height,
        angle=theta,
        facecolor=color,
        edgecolor='none',
        alpha=0.18,
        zorder=1
    )
    plt.gca().add_patch(ellipse)

# ---- Legend ----
handles = [
    plt.Line2D([], [], marker='o', color='none', markerfacecolor=color,
               markeredgecolor='black', markersize=10,
               linestyle='None', label=f"Cluster {i}")
    for i, color in zip(unique_labels, colors)
]

plt.legend(handles=handles, title="Clusters",
           loc='upper right', prop={'weight': 'bold'})

plt.xlabel("UMAP-1", fontsize=20)
plt.ylabel("UMAP-2", fontsize=20)

ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig("cluster_umap_kmeans_from_npy.png", dpi=1000, bbox_inches='tight')
plt.show()

print("✅ Figure saved as: cluster_umap_kmeans_from_npy.png")
