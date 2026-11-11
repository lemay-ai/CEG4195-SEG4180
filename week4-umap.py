# container: scipy
# Outside the course container: pip install umap-learn
"""UMAP case study: find the structure in 64-dimensional data without labels.

The data is scikit-learn's copy of the UCI handwritten digits: 1,797 images of 8 x 8
pixels, so each row has 64 columns. UMAP maps every row to 2 numbers, keeping each
point's nearest neighbours close. It never sees the labels: they only colour the plot
afterwards, to check what UMAP found. The plot is saved as week4-umap.png.
"""
import matplotlib.pyplot as plt
from matplotlib import patheffects
import numpy as np
import umap
from sklearn.datasets import load_digits
from sklearn.neighbors import NearestNeighbors

# region: data
digits = load_digits(as_frame=True)
X = digits.data.astype(int)  # 1,797 rows x 64 columns (pixel_0_0 to pixel_7_7), ink from 0 to 16
y = digits.target            # the digit, 0 to 9: used only to colour the plot
print(X.assign(digit=y).head().to_string(max_cols=11))
# endregion

# region: umap
# All 64 columns are pixels on the same 0 to 16 scale, so they need no rescaling.
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42,
                    compatibility_layout=False)  # the newer layout, a future default
embedding = reducer.fit_transform(X)  # 1,797 rows x 2 columns
# endregion

# How well do the islands match the digits? For each point, the share of its
# 10 nearest neighbours in the 2-D map that are the same digit.
_, neighbours = NearestNeighbors(n_neighbors=11).fit(embedding).kneighbors(embedding)
same = (y.to_numpy()[neighbours[:, 1:]] == y.to_numpy()[:, None]).mean()
print(f"\n{embedding.shape[0]} digits, {X.shape[1]} columns -> {embedding.shape[1]} columns")
print(f"Neighbours in the UMAP map that are the same digit: {same:.1%}")

fig, ax = plt.subplots(figsize=(9, 7.5))
scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap="tab10", s=10, alpha=0.8)
for digit in range(10):  # label each digit next to its most central point (its medoid)
    points = embedding[y == digit]
    medoid = points[np.argmin(((points[:, None] - points[None]) ** 2).sum(-1).sum(1))]
    ax.annotate(str(digit), medoid, xytext=(14, 14), textcoords="offset points", fontsize=18,
                weight="bold", ha="center", va="center",
                path_effects=[patheffects.withStroke(linewidth=4, foreground="white")])
fig.colorbar(scatter, ax=ax, ticks=range(10), label="Digit (not used by UMAP)")
ax.set_title("UMAP of 1,797 handwritten digits: 64 pixels down to 2 dimensions")
ax.set_xlabel("UMAP dimension 1")
ax.set_ylabel("UMAP dimension 2")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("week4-umap.png", dpi=110)
plt.show()
