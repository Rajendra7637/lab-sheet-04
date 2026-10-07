# Lab Sheet-04: Clustering and Dimensionality Reduction

**Name:** <your name>
**Roll No:** <your roll number>
**Course:** MCA, 3rd Semester (2026-2027)
**University:** COER University, Roorkee

## About this project

This is my work for Lab Sheet-04 (Lab Assessment on Unsupervised Learning).
It has 35 programs. They group customers into clusters with K-Means and
Hierarchical Clustering, reduce the data with PCA, and check the quality of
the clusters with the Silhouette Score and other measures.

## What is in the folder

| File / Folder | What it does |
|---|---|
| `lab_sheet_04_all_programs.py` | All 35 programs in one file. Each program is its own cell. |
| `notebooks/lab_sheet_04.ipynb` | The same programs as a Jupyter notebook. |
| `datasets/customers.csv` | The dataset: 300 customers, 8 columns. |
| `processed/customers_clustered.csv` | The dataset with cluster labels, saved by Program 34. |
| `scripts/generate_dataset.py` | Creates `customers.csv` again if it gets deleted. |
| `outputs/` | The graphs saved by the programs. |
| `requirements.txt` | The list of libraries to install. |

## Dataset

`customers.csv` is a sample customer dataset, like the Mall Customers data.
Columns: CustomerID, Gender, Age, Annual_Income_k, Spending_Score,
Visits_Per_Month, Avg_Basket_Value, Online_Purchase_Pct.
The 6 numeric columns (not CustomerID or Gender) are used for clustering.
It is a made-up dataset. You can replace it with the real Mall Customers, Wine
or Iris CSV and change the `FEATURES` list at the top of the main file.

## Libraries used

NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, SciPy, Jupyter Notebook.
Python version: 3.11 or above.

## How to run it

1. Open the project folder in VS Code.
2. Open the terminal and make a virtual environment:

   ```
   python -m venv venv
   venv\Scripts\activate
   ```

   (On Linux or Mac, use `source venv/bin/activate`.)

3. Install the libraries:

   ```
   pip install -r requirements.txt
   ```

4. Run the whole file:

   ```
   python lab_sheet_04_all_programs.py
   ```

   Or open the file in VS Code and click **Run Cell** above any program.
   You can also open the `.ipynb` file and run the cells there.

## What the programs cover

- **Programs 1-5:** Load, explore and standardize the data.
- **Programs 6-14:** K-Means (own version and Scikit-learn), K = 2 and 3,
  different K values, Elbow Method, centroids, and the effect of scaling.
- **Programs 15-21:** Hierarchical Clustering, dendrogram, Ward and Complete
  linkage, and comparison with K-Means.
- **Programs 22-29:** PCA, explained variance, and K-Means before and after PCA.
- **Programs 30-35:** Silhouette Score, cluster profiles, pair plots, saving
  the clustered data, and the final comparison.

## Cluster analysis

<Run the programs, then describe each cluster in your own words. For example:
which cluster has high income and low spending, and what kind of customers
they might be.>

## Observations

<Write 3-4 points in your own words after you see your outputs. For example:
what K the elbow and silhouette methods gave, whether K-Means and Hierarchical
Clustering agreed, and how much information PCA kept.>

## Conclusion

<Write 2-3 lines in your own words about what you learned.>
