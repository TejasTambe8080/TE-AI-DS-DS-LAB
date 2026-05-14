# PRACTICAL 9: CLUSTERING - K-MEANS

## Practical Title
Unsupervised Learning with K-Means Clustering

## Aim
To perform unsupervised clustering on unlabeled data to identify natural groups and patterns.

## Objective
1. Understand K-Means algorithm and centroid concept
2. Implement K-Means clustering
3. Determine optimal K using elbow method
4. Calculate within-cluster and between-cluster distances
5. Interpret and visualize clusters

## Problem Statement
Apply K-Means clustering to Iris dataset to identify natural groupings of flowers, compare with actual species.

---

## THEORY

### 1. K-Means Algorithm
**Objective:** Minimize within-cluster sum of squares (WCSS)

$$WCSS = \sum_{i=1}^{k} \sum_{p \in C_i} ||p - c_i||^2$$

Where:
- k = number of clusters
- Cᵢ = i-th cluster
- cᵢ = centroid of cluster i
- p = data point

### 2. K-Means Steps
1. **Initialize:** Randomly select k centroids
2. **Assign:** Assign each point to nearest centroid
3. **Update:** Calculate new centroids
4. **Repeat:** Until convergence (centroids don't change)

### 3. Elbow Method
- Calculate WCSS for k=1 to 10
- Plot k vs WCSS
- Find "elbow" (point of diminishing returns)
- Choose k at elbow

### 4. Silhouette Score
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

Where:
- a(i) = avg distance to points in same cluster
- b(i) = avg distance to points in nearest cluster
- Range: -1 to 1, higher = better

### 5. Advantages & Disadvantages

**Advantages:**
- Fast, simple, scalable
- Works well with spherical clusters

**Disadvantages:**
- Requires specifying k
- Random initialization
- Assumes similar-sized clusters

---

## COMPLETE CODE

```python
# ===== PRACTICAL 9: K-MEANS CLUSTERING =====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, confusion_matrix

print("=" * 80)
print("PRACTICAL 9: K-MEANS CLUSTERING - IRIS DATASET")
print("=" * 80)

# ===== STEP 1: LOAD AND PREPARE DATA =====
print("\nSTEP 1: LOAD IRIS DATASET")
print("-" * 80)

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
species_names = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

print(f"Dataset shape: {X.shape}")
print(f"Features: {X.shape[1]}")
print(f"Samples: {X.shape[0]}")

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print(f"Features normalized: Yes")

# ===== STEP 2: ELBOW METHOD =====
print("\n" + "=" * 80)
print("STEP 2: DETERMINE OPTIMAL K - ELBOW METHOD")
print("-" * 80)

wcss = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled_df)
    wcss.append(kmeans.inertia_)

print(f"\nWCSS for different k values:")
for k, inertia in zip(k_range, wcss):
    print(f"  k={k}: {inertia:.2f}")

# Find elbow (max decrease in WCSS)
differences = [wcss[i] - wcss[i+1] for i in range(len(wcss)-1)]
elbow_k = np.argmax(differences) + 1
print(f"\nElbow point: k={elbow_k} (max decrease: {max(differences):.2f})")

# ===== STEP 3: TRAIN K-MEANS WITH OPTIMAL K =====
print("\n" + "=" * 80)
print(f"STEP 3: TRAIN K-MEANS WITH K={3}")
print("-" * 80)

optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled_df)

print(f"Model trained with k={optimal_k}")
print(f"Inertia (WCSS): {kmeans.inertia_:.4f}")
print(f"Cluster centers shape: {kmeans.cluster_centers_.shape}")

# ===== STEP 4: ANALYZE CLUSTERS =====
print("\n" + "=" * 80)
print("STEP 4: CLUSTER ANALYSIS")
print("-" * 80)

print(f"\nCluster Sizes:")
unique, counts = np.unique(clusters, return_counts=True)
for cluster_id, count in zip(unique, counts):
    pct = count / len(clusters) * 100
    print(f"  Cluster {cluster_id}: {count} samples ({pct:.1f}%)")

print(f"\nCluster Centers (Scaled Features):")
for i, center in enumerate(kmeans.cluster_centers_):
    print(f"  Cluster {i}: {center.round(3)}")

print(f"\nCluster Centers (Original Scale):")
centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
for i, center in enumerate(centers_original):
    print(f"  Cluster {i}:")
    for feat, val in zip(iris.feature_names, center):
        print(f"    {feat}: {val:.2f}")

# ===== STEP 5: SILHOUETTE ANALYSIS =====
print("\n" + "=" * 80)
print("STEP 5: SILHOUETTE SCORE ANALYSIS")
print("-" * 80)

silhouette_scores = []
for k in k_range:
    kmeans_k = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    cluster_labels = kmeans_k.fit_predict(X_scaled_df)
    sil_score = silhouette_score(X_scaled_df, cluster_labels)
    silhouette_scores.append(sil_score)
    print(f"k={k}: Silhouette Score = {sil_score:.4f}")

best_k_sil = k_range[np.argmax(silhouette_scores)]
print(f"\nBest k by Silhouette: {best_k_sil} (Score: {max(silhouette_scores):.4f})")

# Silhouette score for optimal k
sil_score_optimal = silhouette_score(X_scaled_df, clusters)
print(f"Silhouette Score for k={optimal_k}: {sil_score_optimal:.4f}")

# ===== STEP 6: COMPARE WITH TRUE LABELS =====
print("\n" + "=" * 80)
print("STEP 6: COMPARE CLUSTERS WITH ACTUAL SPECIES")
print("-" * 80)

# Create confusion matrix between clusters and actual species
cm = confusion_matrix(y, clusters)
print(f"\nCluster vs Species Confusion Matrix:")
print(f"     Cluster 0  Cluster 1  Cluster 2")
species = ['Setosa', 'Versicolor', 'Virginica']
for i, s in enumerate(species):
    print(f"{s:12} {cm[i, 0]:8} {cm[i, 1]:8} {cm[i, 2]:8}")

# Analyze cluster composition
print(f"\nCluster Composition (Actual Species):")
for cluster_id in range(optimal_k):
    cluster_mask = clusters == cluster_id
    print(f"  Cluster {cluster_id}:")
    for species_id, species_name in species_names.items():
        count = np.sum((y == species_id) & cluster_mask)
        pct = count / np.sum(cluster_mask) * 100
        print(f"    {species_name}: {count} ({pct:.1f}%)")

# ===== STEP 7: VISUALIZATION =====
print("\n" + "=" * 80)
print("STEP 7: VISUALIZATION")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Elbow Method
axes[0, 0].plot(k_range, wcss, 'bo-', linewidth=2, markersize=8)
axes[0, 0].axvline(x=3, color='r', linestyle='--', linewidth=2, label='k=3')
axes[0, 0].set_xlabel('Number of Clusters (k)')
axes[0, 0].set_ylabel('WCSS')
axes[0, 0].set_title('Elbow Method - WCSS vs k', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Silhouette Scores
axes[0, 1].plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
axes[0, 1].axvline(x=3, color='r', linestyle='--', linewidth=2, label='k=3')
axes[0, 1].set_xlabel('Number of Clusters (k)')
axes[0, 1].set_ylabel('Silhouette Score')
axes[0, 1].set_title('Silhouette Score vs k', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Clusters (2D projection - first 2 features)
scatter = axes[1, 0].scatter(X_scaled_df.iloc[:, 0], X_scaled_df.iloc[:, 1],
                            c=clusters, cmap='viridis', s=50, alpha=0.6, edgecolor='black')
# Plot centroids
centroids = kmeans.cluster_centers_
axes[1, 0].scatter(centroids[:, 0], centroids[:, 1], 
                  c='red', marker='X', s=200, edgecolor='black', linewidth=2, label='Centroids')
axes[1, 0].set_xlabel('Feature 1')
axes[1, 0].set_ylabel('Feature 2')
axes[1, 0].set_title('K-Means Clusters (2D Projection)', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[1, 0], label='Cluster')

# Plot 4: Silhouette Plot
y_lower = 10
sil_vals = silhouette_samples(X_scaled_df, clusters)

colors = plt.cm.viridis(np.linspace(0, 1, optimal_k))
for i in range(optimal_k):
    cluster_sil_vals = sil_vals[clusters == i]
    cluster_sil_vals.sort()
    
    size_cluster_i = cluster_sil_vals.shape[0]
    y_upper = y_lower + size_cluster_i
    
    axes[1, 1].fill_betweenx(np.arange(y_lower, y_upper),
                            0, cluster_sil_vals, 
                            facecolor=colors[i], alpha=0.7, edgecolor='black')
    y_lower = y_upper + 10

axes[1, 1].axvline(x=sil_score_optimal, color='r', linestyle='--', linewidth=2, label='Avg Silhouette')
axes[1, 1].set_xlabel('Silhouette Coefficient')
axes[1, 1].set_ylabel('Cluster')
axes[1, 1].set_title('Silhouette Plot', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('kmeans_clustering.png', dpi=150, bbox_inches='tight')
print("Clustering visualizations saved as 'kmeans_clustering.png'")
plt.show()

# ===== STEP 8: SUMMARY =====
print("\n" + "=" * 80)
print("STEP 8: CLUSTERING SUMMARY")
print("=" * 80)

print(f"\nK-MEANS CLUSTERING RESULTS:")
print(f"  Optimal k: {optimal_k}")
print(f"  WCSS: {kmeans.inertia_:.4f}")
print(f"  Silhouette Score: {sil_score_optimal:.4f}")
print(f"  Convergence: {'Yes' if kmeans.n_iter_ < 300 else 'No'}")

print(f"\nCLUSTERING QUALITY:")
if sil_score_optimal > 0.5:
    print(f"  Quality: GOOD (Silhouette > 0.5)")
elif sil_score_optimal > 0.3:
    print(f"  Quality: ACCEPTABLE (Silhouette > 0.3)")
else:
    print(f"  Quality: POOR (Silhouette < 0.3)")

print(f"\n✓ K-MEANS CLUSTERING PRACTICAL COMPLETED!")
print("=" * 80)
```

---

## KEY FORMULAS

### WCSS (Within-Cluster Sum of Squares)
$$WCSS = \sum_{i=1}^{k} \sum_{p \in C_i} ||p - c_i||^2$$

### Silhouette Score
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

---

## EXTERNAL VIVA QUESTIONS (10 Key)

### Q1: How does K-Means algorithm work?
**A:** (1) Initialize centroids, (2) Assign points to nearest centroid, (3) Update centroids, (4) Repeat until convergence.

### Q2: What does WCSS minimize?
**A:** Within-cluster sum of squares. Minimizes variance within clusters.

### Q3: What is elbow method?
**A:** Plot WCSS vs k. Choose k at "elbow" (diminishing returns point).

### Q4: How to determine optimal k?
**A:** Elbow method, silhouette score, domain knowledge, or try multiple k values.

### Q5: What is silhouette score?
**A:** Measure of cluster quality [-1,1]. Higher = better separated clusters.

### Q6: K-Means disadvantages?
**A:** Requires specifying k, random initialization, assumes similar-sized clusters.

### Q7: Can you compare clusters to true labels?
**A:** Yes, use confusion matrix, purity, normalized mutual information.

### Q8: What is k-means++ initialization?
**A:** Initialize centroids far apart. Better than random, faster convergence.

### Q9: How many times to run K-Means?
**A:** Multiple times (n_init parameter) - random initialization, take best result.

### Q10: When to use K-Means?
**A:** When need to find natural groupings in unlabeled data.

---

**Ready for External Examination and Journal Submission**
