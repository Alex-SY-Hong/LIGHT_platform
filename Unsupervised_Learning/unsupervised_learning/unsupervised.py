# -*- coding: utf-8 -*-
# PCA + UMAP + KMeans + Visualization + Ellipse + CSV split (single-core)

# ========= Force Single-Core Execution =========
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from umap import UMAP
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
NPY_PATH = r"AB_concat1024.npy"
CSV_PATH = r"final_two_smiles_with_modulus.csv"

# ========= Dimensionality Reduction Parameters =========
PCA_N_COMPONENTS = 64
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "euclidean"
SEED = 42

# ========= Fixed K Value =========
K = 6
print(f"[INFO] Using fixed number of clusters K={K}")

# ========= Load Fingerprint Data =========
X = np.load(NPY_PATH)
print(f"[OK] Loaded fingerprint file: {NPY_PATH} -> shape={X.shape}")

# ========= Load CSV (same order as NPY) =========
df = pd.read_csv(CSV_PATH)
print(f"[OK] Loaded CSV file: {CSV_PATH} -> shape={df.shape}")

if len(df) != len(X):
    raise ValueError("CSV row count does not match NPY sample count.")
else:
    print("✅ CSV rows match NPY samples.")

# ========= PCA =========
print(f"[INFO] Performing PCA reduction to {PCA_N_COMPONENTS} dimensions")
pca = PCA(n_components=PCA_N_COMPONENTS, random_state=SEED)
X_pca = pca.fit_transform(X)

# ========= UMAP (Single-Core) =========
print("[INFO] Performing UMAP dimensionality reduction (single-core)...")
um = UMAP(
    n_components=2,
    n_neighbors=UMAP_N_NEIGHBORS,
    min_dist=UMAP_MIN_DIST,
    metric=UMAP_METRIC,
    random_state=SEED,
    n_jobs=1    # <-- force single CPU
)
XY = um.fit_transform(X_pca)
print(f"[OK] UMAP completed: shape={XY.shape}")
np.save("umap2d.npy", XY)
print("✅ Saved UMAP 2D coordinates to umap2d.npy")

# ========= KMeans Clustering (single-core by default) =========
km = KMeans(n_clusters=K, n_init=10, random_state=SEED)
labels = km.fit_predict(XY)
print(f"[OK] Clustering completed, K={K}")

# ========= Count samples per cluster =========
unique_labels, counts = np.unique(labels, return_counts=True)
print("\n===== Cluster Sample Counts =====")
for lbl, cnt in zip(unique_labels, counts):
    print(f"Cluster {lbl}: {cnt} samples")
print("=================================\n")

# ========= Save CSV by cluster =========
os.makedirs("clusters", exist_ok=True)

for lbl in unique_labels:
    df_sub = df[labels == lbl]
    save_path = f"clusters/cluster_{lbl}.csv"
    df_sub.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved cluster {lbl} -> {save_path} ({len(df_sub)} rows)")

print("\n✅ All clusters saved in folder: clusters/")

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
plt.savefig("cluster_umap_kmeans.png", dpi=1000, bbox_inches='tight')
plt.show()

print("✅ Figure saved as: cluster_umap.png")
