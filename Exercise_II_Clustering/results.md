# Exercise II - Clustering: Results

```
Loaded Mall Customer dataset: 200 rows, columns: ['CustomerID', 'Genre', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']

======================================================================
1. AGNES ASSIGNMENT
======================================================================
  linkage=single     metric=euclidean  silhouette(k=5)=0.276
  linkage=single     metric=manhattan  silhouette(k=5)=0.276
  linkage=complete   metric=euclidean  silhouette(k=5)=0.553
  linkage=complete   metric=manhattan  silhouette(k=5)=0.539
  linkage=average    metric=euclidean  silhouette(k=5)=0.479
  linkage=average    metric=manhattan  silhouette(k=5)=0.551

Q1. Best combination: linkage=complete, metric=euclidean (silhouette=0.553) -> used for the k=5 visualization below

Q2/Q3. Cluster profiles (mean income / spending, cluster size):
               Annual Income (k$)  Spending Score (1-100)  count
agnes_cluster                                                   
0                            86.5                    82.1     39
1                            26.3                    20.9     23
2                            87.0                    18.6     38
3                            25.1                    80.0     21
4                            54.4                    50.2     79

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

======================================================================
2. K-MEANS ASSIGNMENT
======================================================================
  k= 1 -> WCSS (inertia) = 400.00
  k= 2 -> WCSS (inertia) = 270.03
  k= 3 -> WCSS (inertia) = 157.70
  k= 4 -> WCSS (inertia) = 108.92
  k= 5 -> WCSS (inertia) = 65.57
  k= 6 -> WCSS (inertia) = 55.10
  k= 7 -> WCSS (inertia) = 44.87
  k= 8 -> WCSS (inertia) = 37.20
  k= 9 -> WCSS (inertia) = 32.41
  k=10 -> WCSS (inertia) = 29.08

Q2. Optimal number of clusters based on the elbow method: k = 5
    (visually: WCSS drops sharply up to k=5 and flattens out afterwards -> adding more clusters beyond 5 brings diminishing returns in within-cluster homogeneity)
                Annual Income (k$)  Spending Score (1-100)  count
kmeans_cluster                                                   
0                             25.7                    79.4     22
1                             26.3                    20.9     23
2                             88.2                    17.1     35
3                             55.3                    49.5     81
4                             86.5                    82.1     39

======================================================================
3. DBSCAN ASSIGNMENT
======================================================================
Q1. Automatically detected knee at epsilon=0.313, using minPts=4 (k=3 neighbours + 1, following the rule of thumb MinPts = 2 * dimensions)
    (the course slides settled on epsilon=0.28, minPts=4 for the same dataset/feature set, which matches this order of magnitude)

Q2. DBSCAN found 7 clusters and 18 noise points (out of 200)
                Annual Income (k$)  Spending Score (1-100)  count
dbscan_cluster                                                   
-1                            89.8                    63.7     18
 0                            25.1                    77.3     16
 1                            24.6                     9.6     12
 2                            22.4                    34.4      7
 3                            55.0                    48.3     89
 4                            80.5                    82.6     32
 5                            81.0                    12.8     23
 6                           102.3                    21.3      3

Q3. Interpretation:
    - Cluster -1 (noise, 18 customers) contains customers that do not
      clearly belong to any dense group - e.g. medium/high income with
      spending behaviour that does not match any tight neighbourhood. These
      are the least predictable customers, not a good target for a uniform
      campaign.
    - The remaining 7 clusters reproduce the same broad
      income/spending segments found by AGNES and K-Means (low/medium/high
      income crossed with low/high spending), but DBSCAN additionally
      splits a few of them into smaller, denser sub-groups because it only
      merges points that are locally close together, instead of forcing a
      fixed number of clusters like K-Means/AGNES.
    - Compared to K-Means/AGNES, DBSCAN's strength here is that it does not
      force every customer into a segment - the marketing team can treat the
      noise points individually rather than mis-assigning them to the
      nearest, but not really fitting, cluster.

```
