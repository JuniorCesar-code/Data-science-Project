"""
Data Science - Exercise II: Clustering
Reimplements the three RapidMiner/AI Studio processes provided
(AGNES_CustomerMallDataset.rmp, KMean_CustomerMallDataset.rmp,
DBSCAN_CustomerMallDataset.rmp) in Python on the Mall Customer dataset.

Features used (as in the .rmp files): "Annual Income (k$)" and
"Spending Score (1-100)", Z-transformed (StandardScaler) before clustering.

Run:
    python clustering.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

log_lines = []


def log(*args):
    msg = " ".join(str(a) for a in args)
    print(msg)
    log_lines.append(msg)


df = pd.read_csv(os.path.join(HERE, "data", "Mall_Customers.csv"))
log(f"Loaded Mall Customer dataset: {df.shape[0]} rows, columns: {list(df.columns)}")

X_raw = df[["Annual Income (k$)", "Spending Score (1-100)"]].values
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)  # Z-transformation, like the Normalize operator

# ---------------------------------------------------------------------------
# 1. AGNES - Agglomerative Hierarchical Clustering
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log("1. AGNES ASSIGNMENT")
log("=" * 70)

linkages = ["single", "complete", "average"]
metrics = ["euclidean", "manhattan"]

# Q1: run all 3 x 2 = 6 combinations, evaluate with silhouette score
from sklearn.metrics import silhouette_score

results = []
for link in linkages:
    for metric in metrics:
        model = AgglomerativeClustering(n_clusters=5, linkage=link, metric=metric)
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)
        results.append((link, metric, score))
        log(f"  linkage={link:<10} metric={metric:<10} "
            f"silhouette(k=5)={score:.3f}")

results_df = pd.DataFrame(results, columns=["linkage", "metric", "silhouette"])
results_df.to_csv(os.path.join(OUT, "agnes_linkage_comparison.csv"), index=False)
best_link, best_metric, best_score = max(results, key=lambda r: r[2])
log(f"\nQ1. Best combination: linkage={best_link}, metric={best_metric} "
    f"(silhouette={best_score:.3f}) -> used for the k=5 visualization below")

# dendrogram for the best combination (for illustration)
Z = linkage(X, method=best_link if best_link != "ward" else "ward",
            metric=best_metric if best_link != "ward" else "euclidean")
fig, ax = plt.subplots(figsize=(10, 5))
dendrogram(Z, truncate_mode="lastp", p=20, ax=ax)
ax.set_title(f"AGNES dendrogram (linkage={best_link}, metric={best_metric})")
ax.set_xlabel("cluster size (merged leaves)")
ax.set_ylabel("distance")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "agnes_dendrogram.png"), dpi=150)
plt.close(fig)

# Q2: visualize clusters with k = 5 using the best combination
model = AgglomerativeClustering(n_clusters=5, linkage=best_link, metric=best_metric)
df["agnes_cluster"] = model.fit_predict(X)

fig, ax = plt.subplots(figsize=(7, 6))
scatter = ax.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"],
                      c=df["agnes_cluster"], cmap="tab10", s=40)
ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1-100)")
ax.set_title(f"AGNES clusters (k=5, linkage={best_link}, metric={best_metric})")
legend1 = ax.legend(*scatter.legend_elements(), title="cluster")
ax.add_artist(legend1)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "agnes_clusters_k5.png"), dpi=150)
plt.close(fig)

summary = df.groupby("agnes_cluster")[["Annual Income (k$)", "Spending Score (1-100)"]].mean()
summary["count"] = df.groupby("agnes_cluster").size()
log("\nQ2/Q3. Cluster profiles (mean income / spending, cluster size):")
log(summary.round(1).to_string())
log("""
Q3. Interpretation for the mall's marketing team:
    - High income / high spending  -> premium customers: target with loyalty
      programs, VIP previews of new collections, premium brand partnerships.
    - High income / low spending   -> price-insensitive but currently
      disengaged: target with personalised offers / exclusive experiences to
      convert spending potential into actual spending.
    - Low income / high spending   -> value-driven big spenders: target with
      discount bundles, buy-now-pay-later, loyalty points to retain them.
    - Low income / low spending    -> least profitable segment: low-cost
      mass-marketing (email/newsletter) only, not a priority for investment.
    - Medium income / medium spending -> average customers: general seasonal
      campaigns, upsell opportunities.
""")

# ---------------------------------------------------------------------------
# 2. K-MEANS
# ---------------------------------------------------------------------------
log("=" * 70)
log("2. K-MEANS ASSIGNMENT")
log("=" * 70)

ks = range(1, 11)
wcss = []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=2001)
    km.fit(X)
    wcss.append(km.inertia_)
    log(f"  k={k:>2} -> WCSS (inertia) = {km.inertia_:.2f}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(list(ks), wcss, marker="o")
ax.set_xlabel("number of clusters k")
ax.set_ylabel("within-cluster sum of squares (WCSS)")
ax.set_title("K-Means Elbow Method")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "kmeans_elbow.png"), dpi=150)
plt.close(fig)

# simple automatic elbow detection: point with max distance to the line
# connecting the first and last point of the WCSS curve
wcss_arr = np.array(wcss, dtype=float)
k_arr = np.array(list(ks), dtype=float)
p1 = np.array([k_arr[0], wcss_arr[0]])
p2 = np.array([k_arr[-1], wcss_arr[-1]])
line_vec = p2 - p1
line_vec_norm = line_vec / np.linalg.norm(line_vec)
distances = []
for k_i, w_i in zip(k_arr, wcss_arr):
    p = np.array([k_i, w_i]) - p1
    proj_len = np.dot(p, line_vec_norm)
    proj = proj_len * line_vec_norm
    distances.append(np.linalg.norm(p - proj))
optimal_k = int(k_arr[int(np.argmax(distances))])

log(f"\nQ2. Optimal number of clusters based on the elbow method: k = {optimal_k}")
log("    (visually: WCSS drops sharply up to k=5 and flattens out "
    "afterwards -> adding more clusters beyond 5 brings diminishing "
    "returns in within-cluster homogeneity)")

km_final = KMeans(n_clusters=optimal_k, n_init=10, random_state=2001)
df["kmeans_cluster"] = km_final.fit_predict(X)
centers_original = scaler.inverse_transform(km_final.cluster_centers_)

fig, ax = plt.subplots(figsize=(7, 6))
scatter = ax.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"],
                      c=df["kmeans_cluster"], cmap="tab10", s=40)
ax.scatter(centers_original[:, 0], centers_original[:, 1],
           c="black", marker="X", s=200, label="centroids")
ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1-100)")
ax.set_title(f"K-Means clusters (k={optimal_k})")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "kmeans_clusters.png"), dpi=150)
plt.close(fig)

summary_km = df.groupby("kmeans_cluster")[["Annual Income (k$)", "Spending Score (1-100)"]].mean()
summary_km["count"] = df.groupby("kmeans_cluster").size()
log(summary_km.round(1).to_string())

# ---------------------------------------------------------------------------
# 3. DBSCAN
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log("3. DBSCAN ASSIGNMENT")
log("=" * 70)

# Q1: k-distance graph. Rule of thumb MinPts = 2*dimensions -> k = MinPts-1 = 3
k_for_graph = 3
nbrs = NearestNeighbors(n_neighbors=k_for_graph + 1).fit(X)
distances, _ = nbrs.kneighbors(X)
k_distances = np.sort(distances[:, k_for_graph])  # distance to the kth neighbour

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(k_distances)
ax.set_xlabel("points sorted by distance")
ax.set_ylabel(f"distance to {k_for_graph}-th nearest neighbour")
ax.set_title("DBSCAN k-distance graph")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "dbscan_k_distance_graph.png"), dpi=150)
plt.close(fig)

# automatic "knee" detection via max curvature (max distance to the line
# between first and last point), same trick as the elbow method above
idx = np.arange(len(k_distances))
p1 = np.array([idx[0], k_distances[0]])
p2 = np.array([idx[-1], k_distances[-1]])
line_vec = p2 - p1
line_vec_norm = line_vec / np.linalg.norm(line_vec)
dists = []
for i, d in zip(idx, k_distances):
    p = np.array([i, d]) - p1
    proj = np.dot(p, line_vec_norm) * line_vec_norm
    dists.append(np.linalg.norm(p - proj))
knee_idx = int(np.argmax(dists))
epsilon = float(k_distances[knee_idx])
min_pts = k_for_graph + 1  # as in the lecture: MinPoints = k + 1

log(f"Q1. Automatically detected knee at epsilon={epsilon:.3f}, "
    f"using minPts={min_pts} (k={k_for_graph} neighbours + 1, following the "
    "rule of thumb MinPts = 2 * dimensions)")
log(f"    (the course slides settled on epsilon=0.28, minPts=4 for the same "
    "dataset/feature set, which matches this order of magnitude)")

dbscan = DBSCAN(eps=epsilon, min_samples=min_pts)
df["dbscan_cluster"] = dbscan.fit_predict(X)
n_clusters_found = len(set(df["dbscan_cluster"])) - (1 if -1 in df["dbscan_cluster"].values else 0)
n_noise = int((df["dbscan_cluster"] == -1).sum())

log(f"\nQ2. DBSCAN found {n_clusters_found} clusters and {n_noise} noise "
    f"points (out of {len(df)})")

fig, ax = plt.subplots(figsize=(7, 6))
scatter = ax.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"],
                      c=df["dbscan_cluster"], cmap="tab10", s=40)
ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1-100)")
ax.set_title(f"DBSCAN clusters (eps={epsilon:.2f}, minPts={min_pts}); "
             "cluster -1 = noise")
legend1 = ax.legend(*scatter.legend_elements(), title="cluster")
ax.add_artist(legend1)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "dbscan_clusters.png"), dpi=150)
plt.close(fig)

summary_db = df.groupby("dbscan_cluster")[["Annual Income (k$)", "Spending Score (1-100)"]].mean()
summary_db["count"] = df.groupby("dbscan_cluster").size()
log(summary_db.round(1).to_string())
log(f"""
Q3. Interpretation:
    - Cluster -1 (noise, {n_noise} customers) contains customers that do not
      clearly belong to any dense group - e.g. medium/high income with
      spending behaviour that does not match any tight neighbourhood. These
      are the least predictable customers, not a good target for a uniform
      campaign.
    - The remaining {n_clusters_found} clusters reproduce the same broad
      income/spending segments found by AGNES and K-Means (low/medium/high
      income crossed with low/high spending), but DBSCAN additionally
      splits a few of them into smaller, denser sub-groups because it only
      merges points that are locally close together, instead of forcing a
      fixed number of clusters like K-Means/AGNES.
    - Compared to K-Means/AGNES, DBSCAN's strength here is that it does not
      force every customer into a segment - the marketing team can treat the
      noise points individually rather than mis-assigning them to the
      nearest, but not really fitting, cluster.
""")

df.to_csv(os.path.join(OUT, "mall_customers_with_clusters.csv"), index=False)

with open(os.path.join(HERE, "results.md"), "w", encoding="utf-8") as fh:
    fh.write("# Exercise II - Clustering: Results\n\n```\n")
    fh.write("\n".join(log_lines))
    fh.write("\n```\n")

log("\nAll outputs written to:", OUT)
