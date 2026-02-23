# Credit Card Customer Segmentation | Deep Learning Clustering

Deep learning-enhanced customer segmentation using an Autoencoder for feature compression followed by K-Means clustering to identify distinct credit card usage profiles.

## Problem Statement
Segment credit card customers into meaningful groups based on spending behavior, payment patterns, and credit usage — enabling personalized financial product recommendations.

## Dataset
| Attribute | Detail |
|---|---|
| File | `CC GENERAL.csv` |
| Records | ~9,000 credit card holders |
| Features | 18 behavioral features (balance, purchases, payments, credit limit, etc.) |
| Task | Unsupervised clustering |

## Methodology
1. **EDA & Visualization** — Feature distributions, correlation heatmap, missing value analysis
2. **Preprocessing** — `SimpleImputer` (median) + `StandardScaler`
3. **Autoencoder** — Deep feature compression to lower-dimensional representation
4. **Optimal k Selection** — Elbow method + Silhouette Score analysis
5. **K-Means Clustering** — `n_clusters=4` on Autoencoder embeddings
6. **Cluster Profiling** — Mean feature values per segment for business interpretation
7. **Visualization** — PCA / t-SNE 2D cluster plots

## Results
| Model | Clusters | Silhouette Score |
|---|---|---|
| K-Means (on Autoencoder embeddings) | k = 4 | **0.309** |

**Segment Profiles:**
- Segment 0: Low balance, low purchases — Inactive / Minimal users
- Segment 1: High balance, high credit limit — Premium / High-value customers
- Segment 2: Moderate activity, installment-focused — Installment buyers
- Segment 3: Active purchasers, low cash advance — Transactional users

## Technologies
`Python` · `scikit-learn` · `TensorFlow/Keras` · `Pandas` · `NumPy` · `Seaborn` · `Matplotlib` · `joblib`

## File Structure
```
13_Creditcard_Customer_Clustering/
├── project_notebook.ipynb   # Main notebook
├── CC GENERAL.csv           # Dataset
└── models/                  # Saved Autoencoder and KMeans models
```

## How to Run
```bash
cd 13_Creditcard_Customer_Clustering
jupyter notebook project_notebook.ipynb
```
