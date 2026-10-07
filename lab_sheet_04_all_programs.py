# %% [markdown]
# # Lab Sheet-04: Unsupervised Learning (Clustering and Dimensionality Reduction)
# **MCA III Semester (Session 2026-2027) - COER University, Roorkee**
#
# Each `# %%` block is one experiment (a separate cell in VS Code / Jupyter).
#
# Libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, SciPy
# Dataset: `datasets/customers.csv` (customer segmentation data)

# %% [markdown]
# ## Setup: imports, paths and helper functions

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()
    if BASE_DIR.name == "notebooks":
        BASE_DIR = BASE_DIR.parent

DATA_PATH = BASE_DIR / "datasets" / "customers.csv"
PROCESSED_DIR = BASE_DIR / "processed"
OUT_DIR = BASE_DIR / "outputs"
PROCESSED_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
FEATURES = ["Age", "Annual_Income_k", "Spending_Score",
            "Visits_Per_Month", "Avg_Basket_Value", "Online_Purchase_Pct"]
PLOT_X, PLOT_Y = "Annual_Income_k", "Spending_Score"  # two features used for 2D plots


def load_dataset(path=DATA_PATH):
    """Load the CSV file with basic validation and exception handling."""
    try:
        if not Path(path).exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        data = pd.read_csv(path)
        if data.empty:
            raise ValueError("Dataset is empty")
        return data
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as err:
        print("Error while loading dataset:", err)
        raise


def run_kmeans(data, n_clusters):
    """Fit K-Means and return the fitted model."""
    return KMeans(n_clusters=n_clusters, n_init=10, random_state=RANDOM_STATE).fit(data)


def plot_clusters(ax, x_values, y_values, labels, title, xlabel, ylabel, centers=None):
    """Draw a scatter plot where each cluster has its own colour."""
    sns.scatterplot(x=x_values, y=y_values, hue=labels, palette="tab10",
                    s=45, ax=ax, legend="full")
    if centers is not None:
        ax.scatter(centers[:, 0], centers[:, 1], c="black", marker="X", s=200, label="Centroid")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


# %% [markdown]
# # Part A: Data Preparation (Programs 1-5)

# %% [markdown]
# ## Program 1: Load a dataset suitable for clustering using Pandas

# %%
df = load_dataset()
print("Dataset loaded. Shape:", df.shape)

# %% [markdown]
# ## Program 2: First and last five records

# %%
print("First 5 records:")
print(df.head())
print("\nLast 5 records:")
print(df.tail())

# %% [markdown]
# ## Program 3: Dataset information, summary statistics and data types

# %%
df.info()
print("\nData types:\n", df.dtypes)
print("\nSummary statistics:\n", df.describe().round(2))
print("\nMissing values:", df.isnull().sum().sum())

# %% [markdown]
# ## Program 4: Select relevant numerical features for clustering

# %%
# CustomerID is only a label and Gender is text, so they are not used
X = df[FEATURES].copy()
print("Selected features:", FEATURES)
print("Shape of feature data:", X.shape)

# %% [markdown]
# ## Program 5: Standardize the dataset using StandardScaler

# %%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURES)
print(X_scaled_df.describe().loc[["mean", "std"]].round(3))

# %% [markdown]
# # Part B: K-Means Clustering (Programs 6-14)

# %% [markdown]
# ## Program 6: Implement the K-Means clustering algorithm

# %%
def simple_kmeans(data, k, max_iterations=100, seed=RANDOM_STATE):
    """K-Means from scratch.

    Steps: 1) pick k random points as centres, 2) assign every point to its
    nearest centre, 3) move each centre to the mean of its points,
    4) repeat until the centres stop moving.
    """
    rng = np.random.default_rng(seed)
    centers = data[rng.choice(len(data), k, replace=False)]
    for _ in range(max_iterations):
        distances = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
        labels = distances.argmin(axis=1)
        new_centers = np.array([
            data[labels == i].mean(axis=0) if np.any(labels == i) else centers[i]
            for i in range(k)
        ])
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    inertia = ((data - centers[labels]) ** 2).sum()
    return labels, centers, inertia


own_labels, own_centers, own_inertia = simple_kmeans(X_scaled, k=3)
sklearn_model = run_kmeans(X_scaled, 3)
print("Own K-Means inertia      :", round(own_inertia, 2))
print("Scikit-learn K-Means     :", round(sklearn_model.inertia_, 2))
print("Own cluster sizes        :", np.bincount(own_labels))

# %% [markdown]
# ## Program 7: K-Means with K = 2

# %%
kmeans_k2 = run_kmeans(X_scaled, 2)
print("K = 2")
print("Cluster sizes:", np.bincount(kmeans_k2.labels_))
print("Inertia      :", round(kmeans_k2.inertia_, 2))
print("Silhouette   :", round(silhouette_score(X_scaled, kmeans_k2.labels_), 4))

# %% [markdown]
# ## Program 8: K-Means with K = 3

# %%
kmeans_k3 = run_kmeans(X_scaled, 3)
print("K = 3")
print("Cluster sizes:", np.bincount(kmeans_k3.labels_))
print("Inertia      :", round(kmeans_k3.inertia_, 2))
print("Silhouette   :", round(silhouette_score(X_scaled, kmeans_k3.labels_), 4))

# %% [markdown]
# ## Program 9: K-Means with different values of K

# %%
k_results = []
for k in range(2, 9):
    model = run_kmeans(X_scaled, k)
    k_results.append({
        "K": k,
        "Inertia": round(model.inertia_, 2),
        "Silhouette": round(silhouette_score(X_scaled, model.labels_), 4),
        "Smallest cluster": int(np.bincount(model.labels_).min()),
    })
k_results_df = pd.DataFrame(k_results)
print(k_results_df)

# %% [markdown]
# ## Program 10: Optimal number of clusters using the Elbow Method

# %%
k_values = np.arange(1, 11)
inertia_values = np.array([run_kmeans(X_scaled, k).inertia_ for k in k_values])

# Find the "elbow": the point farthest from the straight line joining the first and last points
line_start = np.array([k_values[0], inertia_values[0]])
line_end = np.array([k_values[-1], inertia_values[-1]])
line_direction = (line_end - line_start) / np.linalg.norm(line_end - line_start)
points = np.column_stack([k_values, inertia_values]) - line_start
distance_to_line = np.abs(points[:, 0] * line_direction[1] - points[:, 1] * line_direction[0])
elbow_k = int(k_values[distance_to_line.argmax()])

plt.figure(figsize=(7, 4))
plt.plot(k_values, inertia_values, marker="o")
plt.axvline(elbow_k, color="red", linestyle="--", label=f"Elbow at K = {elbow_k}")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia (within-cluster sum of squares)")
plt.title("Elbow Method")
plt.legend()
plt.grid(True)
plt.savefig(OUT_DIR / "p10_elbow_method.png", dpi=120, bbox_inches="tight")
plt.show()

best_silhouette_k = int(k_results_df.loc[k_results_df["Silhouette"].idxmax(), "K"])
print("Elbow method suggests K =", elbow_k)
print("Best silhouette score at K =", best_silhouette_k)

# The K used in the rest of the experiments
OPTIMAL_K = best_silhouette_k
print("K used for the remaining programs:", OPTIMAL_K)

# %% [markdown]
# ## Program 11: Visualize clusters using a scatter plot

# %%
kmeans_final = run_kmeans(X_scaled, OPTIMAL_K)
kmeans_labels = kmeans_final.labels_

fig, ax = plt.subplots(figsize=(7, 5))
plot_clusters(ax, df[PLOT_X], df[PLOT_Y], kmeans_labels,
              f"K-Means clusters (K = {OPTIMAL_K})", PLOT_X, PLOT_Y)
plt.savefig(OUT_DIR / "p11_kmeans_scatter.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Program 12: Display the cluster centroids

# %%
# Centroids are in standardized units, so convert them back to real units
centroids_original = pd.DataFrame(
    scaler.inverse_transform(kmeans_final.cluster_centers_), columns=FEATURES
).round(2)
centroids_original.index.name = "Cluster"
print("Centroids (original units):")
print(centroids_original)

fig, ax = plt.subplots(figsize=(7, 5))
plot_clusters(ax, df[PLOT_X], df[PLOT_Y], kmeans_labels, "Clusters with centroids",
              PLOT_X, PLOT_Y)
ax.scatter(centroids_original[PLOT_X], centroids_original[PLOT_Y],
           c="black", marker="X", s=220, label="Centroid")
ax.legend()
plt.savefig(OUT_DIR / "p12_centroids.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Program 13: Assign cluster labels to the original dataset

# %%
df_clustered = df.copy()
df_clustered["KMeans_Cluster"] = kmeans_labels
print(df_clustered.head())
print("\nCustomers in each cluster:")
print(df_clustered["KMeans_Cluster"].value_counts().sort_index())

# %% [markdown]
# ## Program 14: Compare clustering results before and after feature scaling

# %%
kmeans_raw = run_kmeans(X.to_numpy(), OPTIMAL_K)   # without scaling
labels_raw = kmeans_raw.labels_

comparison_scaling = pd.DataFrame({
    "Without scaling": [silhouette_score(X_scaled, labels_raw), np.bincount(labels_raw).min()],
    "With scaling": [silhouette_score(X_scaled, kmeans_labels), np.bincount(kmeans_labels).min()],
}, index=["Silhouette (on scaled data)", "Smallest cluster size"]).round(4)
print(comparison_scaling)
print("\nAgreement between the two results (Adjusted Rand Index):",
      round(adjusted_rand_score(labels_raw, kmeans_labels), 4))
print("(1 means identical groups, near 0 means very different groups)")
print("\nAvg_Basket_Value has much bigger numbers than the other features, so without")
print("scaling it dominates the distance calculation.")

# %% [markdown]
# # Part C: Hierarchical Clustering (Programs 15-21)

# %% [markdown]
# ## Program 15: Implement Agglomerative Hierarchical Clustering

# %%
agglo_model = AgglomerativeClustering(n_clusters=OPTIMAL_K, linkage="ward")
agglo_labels = agglo_model.fit_predict(X_scaled)
print("Agglomerative clustering done. Cluster sizes:", np.bincount(agglo_labels))

# %% [markdown]
# ## Program 16: Generate a dendrogram using SciPy

# %%
linkage_ward = linkage(X_scaled, method="ward")
plt.figure(figsize=(11, 5))
dendrogram(linkage_ward, truncate_mode="lastp", p=30, leaf_rotation=90, leaf_font_size=9)
cut_height = (linkage_ward[-OPTIMAL_K, 2] + linkage_ward[-OPTIMAL_K + 1, 2]) / 2
plt.axhline(cut_height, color="red", linestyle="--", label=f"Cut for {OPTIMAL_K} clusters")
plt.title("Dendrogram (Ward linkage, last 30 merges)")
plt.xlabel("Cluster size / sample")
plt.ylabel("Distance")
plt.legend()
plt.savefig(OUT_DIR / "p16_dendrogram.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Program 17: Hierarchical Clustering using Ward linkage

# %%
ward_labels = AgglomerativeClustering(n_clusters=OPTIMAL_K, linkage="ward").fit_predict(X_scaled)
print("Ward linkage")
print("Cluster sizes:", np.bincount(ward_labels))
print("Silhouette   :", round(silhouette_score(X_scaled, ward_labels), 4))
df_clustered["Hierarchical_Cluster"] = ward_labels

# %% [markdown]
# ## Program 18: Hierarchical Clustering using Complete linkage

# %%
complete_labels = AgglomerativeClustering(n_clusters=OPTIMAL_K, linkage="complete").fit_predict(X_scaled)
print("Complete linkage")
print("Cluster sizes:", np.bincount(complete_labels))
print("Silhouette   :", round(silhouette_score(X_scaled, complete_labels), 4))

# %% [markdown]
# ## Program 19: Compare different linkage methods

# %%
linkage_results = []
for method in ["ward", "complete", "average", "single"]:
    labels = AgglomerativeClustering(n_clusters=OPTIMAL_K, linkage=method).fit_predict(X_scaled)
    linkage_results.append({
        "Linkage": method,
        "Silhouette": round(silhouette_score(X_scaled, labels), 4),
        "Davies-Bouldin": round(davies_bouldin_score(X_scaled, labels), 4),
        "Smallest cluster": int(np.bincount(labels).min()),
    })
linkage_df = pd.DataFrame(linkage_results)
print(linkage_df)
print("\nBest linkage by silhouette score:", linkage_df.loc[linkage_df["Silhouette"].idxmax(), "Linkage"])

# %% [markdown]
# ## Program 20: Visualize Hierarchical Clustering results

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
plot_clusters(axes[0], df[PLOT_X], df[PLOT_Y], ward_labels, "Ward linkage", PLOT_X, PLOT_Y)
plot_clusters(axes[1], df[PLOT_X], df[PLOT_Y], complete_labels, "Complete linkage", PLOT_X, PLOT_Y)
plt.tight_layout()
plt.savefig(OUT_DIR / "p20_hierarchical_clusters.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Program 21: Compare Hierarchical Clustering with K-Means

# %%
print("Silhouette  K-Means      :", round(silhouette_score(X_scaled, kmeans_labels), 4))
print("Silhouette  Hierarchical :", round(silhouette_score(X_scaled, ward_labels), 4))
print("Agreement (Adjusted Rand Index):", round(adjusted_rand_score(kmeans_labels, ward_labels), 4))
print("\nCross table (rows = K-Means cluster, columns = Hierarchical cluster):")
print(pd.crosstab(kmeans_labels, ward_labels,
                  rownames=["KMeans"], colnames=["Hierarchical"]))
print("\nNote: cluster numbers are only names. Cluster 0 in one method can match")
print("cluster 2 in the other.")

# %% [markdown]
# # Part D: Principal Component Analysis (Programs 22-29)

# %% [markdown]
# ## Program 22: Apply PCA to reduce dataset dimensions

# %%
pca_full = PCA()                  # keep all components first, to study the variance
pca_full.fit(X_scaled)
print("Number of original features :", X_scaled.shape[1])
print("Number of principal components:", pca_full.n_components_)
print("\nExplained variance of each component:", pca_full.explained_variance_.round(3))

# %% [markdown]
# ## Program 23: Reduce the dataset to two principal components

# %%
pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca_2d.fit_transform(X_scaled)
pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
print("Shape before PCA:", X_scaled.shape, "| after PCA:", X_pca.shape)
print(pca_df.head())

loadings = pd.DataFrame(pca_2d.components_.T, columns=["PC1", "PC2"], index=FEATURES).round(3)
print("\nHow much each feature contributes to each component:")
print(loadings)

# %% [markdown]
# ## Program 24: Visualize the transformed data in 2D

# %%
fig, ax = plt.subplots(figsize=(7, 5))
plot_clusters(ax, pca_df["PC1"], pca_df["PC2"], kmeans_labels,
              "Data after PCA (coloured by K-Means cluster)", "PC1", "PC2")
plt.savefig(OUT_DIR / "p24_pca_scatter.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Program 25: Explained variance ratio of principal components

# %%
variance_ratio = pca_full.explained_variance_ratio_
variance_table = pd.DataFrame({
    "Component": [f"PC{i + 1}" for i in range(len(variance_ratio))],
    "Explained variance ratio": variance_ratio.round(4),
    "Percent": (variance_ratio * 100).round(2),
})
print(variance_table)

plt.figure(figsize=(7, 4))
plt.bar(variance_table["Component"], variance_ratio, color="steelblue")
plt.ylabel("Explained variance ratio")
plt.title("Explained variance of each principal component")
plt.savefig(OUT_DIR / "p25_explained_variance.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Program 26: Cumulative explained variance

# %%
cumulative_variance = np.cumsum(variance_ratio)
for i, value in enumerate(cumulative_variance, start=1):
    print(f"First {i} component(s) explain {value * 100:.2f}% of the variance")

components_for_90 = int(np.argmax(cumulative_variance >= 0.90) + 1)
print(f"\nComponents needed to keep at least 90% of the information: {components_for_90}")

plt.figure(figsize=(7, 4))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o")
plt.axhline(0.90, color="red", linestyle="--", label="90% line")
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("Cumulative explained variance")
plt.legend()
plt.grid(True)
plt.savefig(OUT_DIR / "p26_cumulative_variance.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Program 27: Compare the dataset before and after dimensionality reduction

# %%
X_reconstructed = pca_2d.inverse_transform(X_pca)
reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2)

print(f"Before PCA: {X_scaled.shape[1]} features")
print(f"After PCA : {X_pca.shape[1]} features")
print(f"Information kept by 2 components: {pca_2d.explained_variance_ratio_.sum() * 100:.2f}%")
print(f"Information lost                : {100 - pca_2d.explained_variance_ratio_.sum() * 100:.2f}%")
print(f"Average reconstruction error    : {reconstruction_error:.4f}")

# %% [markdown]
# ## Program 28: K-Means clustering on the PCA-transformed dataset

# %%
kmeans_pca = run_kmeans(X_pca, OPTIMAL_K)
pca_labels = kmeans_pca.labels_
print("Cluster sizes:", np.bincount(pca_labels))

fig, ax = plt.subplots(figsize=(7, 5))
plot_clusters(ax, pca_df["PC1"], pca_df["PC2"], pca_labels,
              "K-Means on PCA data", "PC1", "PC2", centers=kmeans_pca.cluster_centers_)
ax.legend()
plt.savefig(OUT_DIR / "p28_kmeans_on_pca.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Program 29: Compare clustering performance before and after PCA

# %%
comparison_pca = pd.DataFrame({
    "Before PCA (6 features)": [
        silhouette_score(X_scaled, kmeans_labels),
        davies_bouldin_score(X_scaled, kmeans_labels),
        calinski_harabasz_score(X_scaled, kmeans_labels),
    ],
    "After PCA (2 components)": [
        silhouette_score(X_pca, pca_labels),
        davies_bouldin_score(X_pca, pca_labels),
        calinski_harabasz_score(X_pca, pca_labels),
    ],
}, index=["Silhouette (higher is better)", "Davies-Bouldin (lower is better)",
          "Calinski-Harabasz (higher is better)"]).round(4)
print(comparison_pca)
print("\nAgreement between both clusterings (Adjusted Rand Index):",
      round(adjusted_rand_score(kmeans_labels, pca_labels), 4))
print("Note: the scores are measured in different spaces (6D and 2D), so they")
print("show the quality in each space, not an exact win or loss.")

# %% [markdown]
# # Part E: Evaluation and Analysis (Programs 30-35)

# %% [markdown]
# ## Program 30: Calculate the Silhouette Score

# %%
overall_silhouette = silhouette_score(X_scaled, kmeans_labels)
sample_silhouette = silhouette_samples(X_scaled, kmeans_labels)
print(f"Overall Silhouette Score (K = {OPTIMAL_K}): {overall_silhouette:.4f}")
print("\nAverage silhouette of each cluster:")
print(pd.Series(sample_silhouette).groupby(kmeans_labels).mean().round(4))
print("\nHow to read it: close to +1 = well separated clusters, near 0 = overlapping,")
print("negative = points probably in the wrong cluster.")

# %% [markdown]
# ## Program 31: Compare Silhouette Scores for different values of K

# %%
plt.figure(figsize=(7, 4))
plt.plot(k_results_df["K"], k_results_df["Silhouette"], marker="o", color="green")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Silhouette score")
plt.title("Silhouette score for different K")
plt.grid(True)
plt.savefig(OUT_DIR / "p31_silhouette_vs_k.png", dpi=120, bbox_inches="tight")
plt.show()
print(k_results_df[["K", "Silhouette"]].to_string(index=False))
print("\nHighest silhouette score at K =", best_silhouette_k)

# %% [markdown]
# ## Program 32: Characteristics of each cluster using descriptive statistics

# %%
cluster_profile = df_clustered.groupby("KMeans_Cluster")[FEATURES].mean().round(2)
cluster_profile["Customers"] = df_clustered["KMeans_Cluster"].value_counts().sort_index()
print("Average values in each cluster:")
print(cluster_profile)

print("\nSpread (standard deviation) in each cluster:")
print(df_clustered.groupby("KMeans_Cluster")[FEATURES].std().round(2))

print("\nGender share in each cluster:")
print(pd.crosstab(df_clustered["KMeans_Cluster"], df_clustered["Gender"], normalize="index").round(2))

# %% [markdown]
# ## Program 33: Visualize cluster distributions using pair plots

# %%
pair_columns = ["Age", "Annual_Income_k", "Spending_Score", "Avg_Basket_Value"]
pair_grid = sns.pairplot(df_clustered, vars=pair_columns, hue="KMeans_Cluster",
                         palette="tab10", plot_kws={"s": 25, "alpha": 0.7})
pair_grid.figure.suptitle("Cluster distributions (pair plot)", y=1.02)
pair_grid.savefig(OUT_DIR / "p33_pairplot.png", dpi=100)
plt.show()

# %% [markdown]
# ## Program 34: Save the clustered dataset as a CSV file

# %%
try:
    df_clustered["PC1"] = pca_df["PC1"].round(4)
    df_clustered["PC2"] = pca_df["PC2"].round(4)
    output_file = PROCESSED_DIR / "customers_clustered.csv"
    df_clustered.to_csv(output_file, index=False)
    print("Saved:", output_file)
    print("Columns:", df_clustered.columns.tolist())
except OSError as err:
    print("Could not save file:", err)

# %% [markdown]
# ## Program 35: Compare K-Means and Hierarchical Clustering and summarize

# %%
final_comparison = pd.DataFrame({
    "K-Means": [
        silhouette_score(X_scaled, kmeans_labels),
        davies_bouldin_score(X_scaled, kmeans_labels),
        calinski_harabasz_score(X_scaled, kmeans_labels),
    ],
    "Hierarchical (Ward)": [
        silhouette_score(X_scaled, ward_labels),
        davies_bouldin_score(X_scaled, ward_labels),
        calinski_harabasz_score(X_scaled, ward_labels),
    ],
}, index=["Silhouette (higher is better)", "Davies-Bouldin (lower is better)",
          "Calinski-Harabasz (higher is better)"]).round(4)
print(final_comparison)

agreement = adjusted_rand_score(kmeans_labels, ward_labels)
print("\n----- Summary of findings -----")
print(f"1. Best number of clusters found: K = {OPTIMAL_K} (silhouette), elbow method suggested K = {elbow_k}.")
print(f"2. K-Means silhouette = {final_comparison.loc[final_comparison.index[0], 'K-Means']}, "
      f"Hierarchical silhouette = {final_comparison.loc[final_comparison.index[0], 'Hierarchical (Ward)']}.")
print(f"3. Agreement between the two methods: Adjusted Rand Index = {agreement:.3f} (1.0 means identical groups).")
print(f"4. PCA with 2 components keeps {pca_2d.explained_variance_ratio_.sum() * 100:.1f}% of the information.")
print("5. Scaling the features first is important, because large-valued columns")
print("   otherwise dominate the distance calculation.")

# %% [markdown]
# ## Conclusion
# All 35 experiments of Lab Sheet-04 were completed: the data was standardized,
# clustered with K-Means and Hierarchical Clustering, reduced with PCA,
# evaluated with the Silhouette Score and other metrics, and the clustered
# dataset was saved.
