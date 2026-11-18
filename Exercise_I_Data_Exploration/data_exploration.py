"""
Data Science - Exercise I: Introduction & Data Exploration
Solved in Python (pandas / matplotlib / seaborn) instead of Altair AI Studio.

Covers the three student assignments from the slide deck:
  A. Data Cleaning          -> Iris dataset
  B. Exploratory Data Analysis -> Iris dataset
  C. Feature Engineering    -> Titanic dataset

Run:
    python data_exploration.py
Outputs (plots) are written to ./outputs, a written answer sheet to results.md
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

log_lines = []


def log(*args):
    msg = " ".join(str(a) for a in args)
    print(msg)
    log_lines.append(msg)


# ---------------------------------------------------------------------------
# A. DATA CLEANING ASSIGNMENT - Iris dataset
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log("A. DATA CLEANING ASSIGNMENT - Iris dataset")
log("=" * 70)

iris_raw = load_iris(as_frame=True)
iris = iris_raw.frame.copy()
# original sklearn columns: sepal length (cm), sepal width (cm),
#                            petal length (cm), petal width (cm), target
species_map = dict(enumerate(iris_raw.target_names))
iris["species"] = iris["target"].map(species_map)
iris = iris.drop(columns=["target"])

log(f"Q1. Dataset imported: {iris.shape[0]} observations, "
    f"{iris.shape[1] - 1} numerical variables + 1 species label")
log("Q2. Variables included:", list(iris.columns))
log("    -> 4 numerical variables (sepal length/width, petal length/width) "
    "describing flower geometry, 1 categorical label (species: "
    f"{list(species_map.values())})")
log(f"    -> {iris.shape[0]} observations (50 samples per species)")

# Q3: rename variables a1..a4
rename_map = {
    "sepal length (cm)": "a1",
    "sepal width (cm)": "a2",
    "petal length (cm)": "a3",
    "petal width (cm)": "a4",
}
iris = iris.rename(columns=rename_map)
log("Q3. Renamed columns ->", {v: k for k, v in rename_map.items()})
log(iris.head().to_string())

# Q4: missing values
missing = iris.isna().sum()
log("Q4. Missing values per column:\n" + missing.to_string())
log("    -> Total missing values:", int(missing.sum()),
    "(the Iris dataset is clean, no imputation necessary)")

iris.to_csv(os.path.join(OUT, "iris_renamed.csv"), index=False)

# ---------------------------------------------------------------------------
# B. EXPLORATORY DATA ANALYSIS ASSIGNMENT - Iris dataset
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log("B. EXPLORATORY DATA ANALYSIS ASSIGNMENT - Iris dataset")
log("=" * 70)

# Q1: histogram of sepal length (a1) with different bin sizes
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, bins in zip(axes, [5, 10, 100]):
    ax.hist(iris["a1"], bins=bins, color="#4C72B0", edgecolor="white")
    ax.set_title(f"Sepal length (a1) - bins={bins}")
    ax.set_xlabel("sepal length (cm)")
    ax.set_ylabel("frequency")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "q1_histogram_binsizes.png"), dpi=150)
plt.close(fig)
log("Q1. Saved outputs/q1_histogram_binsizes.png")
log("    -> bins=5: too coarse, the bimodal/overlapping structure of the "
    "species disappears")
log("    -> bins=10: a good trade-off, the overall shape (roughly normal, "
    "slightly right-skewed) is visible")
log("    -> bins=100: too fine, almost every bin holds 0-2 points -> the "
    "histogram becomes noisy and random variation looks like structure")

# Q2: boxplot per species for sepal length
fig, ax = plt.subplots(figsize=(6, 5))
iris.boxplot(column="a1", by="species", ax=ax)
ax.set_title("Sepal length (a1) by species")
ax.set_xlabel("species")
ax.set_ylabel("sepal length (cm)")
plt.suptitle("")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "q2_boxplot_sepal_length_by_species.png"), dpi=150)
plt.close(fig)

box_stats = iris.groupby("species")["a1"].describe()[["min", "25%", "50%", "75%", "max"]]
log("Q2. Saved outputs/q2_boxplot_sepal_length_by_species.png")
log(box_stats.to_string())
log("    -> setosa has clearly the smallest sepal length and little "
    "overlap with the other two species")
log("    -> versicolor and virginica overlap considerably in sepal length, "
    "so this attribute alone cannot cleanly separate them "
    "(unlike petal length/width, see scatter matrix)")

# Q3: scatter matrix
colors = {"setosa": "#1f77b4", "versicolor": "#2ca02c", "virginica": "#d62728"}
fig = pd.plotting.scatter_matrix(
    iris[["a1", "a2", "a3", "a4"]], figsize=(9, 9), diagonal="hist",
    c=iris["species"].map(colors), alpha=0.7,
)
plt.suptitle("Scatter matrix (blue=setosa, green=versicolor, red=virginica)")
plt.savefig(os.path.join(OUT, "q3_scatter_matrix.png"), dpi=150)
plt.close("all")

corr = iris[["a1", "a2", "a3", "a4"]].corr()
log("Q3. Saved outputs/q3_scatter_matrix.png")
log("Correlation matrix:\n" + corr.round(2).to_string())
log("    -> a3 (petal length) and a4 (petal width) are almost perfectly "
    f"correlated (r={corr.loc['a3','a4']:.2f}) and together separate the "
    "three species almost perfectly -> most useful features for "
    "classification")
log("    -> a1 (sepal length) correlates fairly strongly with a3/a4 too, "
    "while a2 (sepal width) is only weakly / negatively related to the "
    "others and shows the most overlap between species")

# ---------------------------------------------------------------------------
# C. FEATURE ENGINEERING ASSIGNMENT - Titanic dataset
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log("C. FEATURE ENGINEERING ASSIGNMENT - Titanic dataset")
log("=" * 70)


def load_titanic():
    """Try a couple of online sources, fall back to seaborn's bundled copy."""
    urls = [
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv",
        "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv",
    ]
    for url in urls:
        try:
            df = pd.read_csv(url)
            return df
        except Exception as exc:  # noqa: BLE001
            log(f"    (could not load {url}: {exc})")
    raise RuntimeError("No Titanic source reachable")


try:
    titanic = load_titanic()
    # normalise column names from the seaborn version
    col_map = {}
    if "sex" not in titanic.columns and "Sex" in titanic.columns:
        col_map["Sex"] = "sex"
    if "pclass" not in titanic.columns and "Pclass" in titanic.columns:
        col_map["Pclass"] = "pclass"
    if "fare" not in titanic.columns and "Fare" in titanic.columns:
        col_map["Fare"] = "fare"
    titanic = titanic.rename(columns=col_map)
    titanic_available = True
except Exception as exc:  # noqa: BLE001
    log(f"Could not download the Titanic dataset ({exc}). "
        "Falling back to a small synthetic sample with the same columns "
        "so the feature-engineering steps can still be demonstrated.")
    rng = np.random.default_rng(42)
    n = 200
    titanic = pd.DataFrame({
        "sex": rng.choice(["male", "female"], size=n, p=[0.65, 0.35]),
        "pclass": rng.choice([1, 2, 3], size=n, p=[0.25, 0.25, 0.5]),
        "fare": rng.gamma(shape=2.0, scale=15.0, size=n),
    })
    titanic_available = False

log(f"Titanic dataset loaded (online: {titanic_available}), "
    f"{titanic.shape[0]} rows")

# Q1: encoding strategy for sex and class
log("\nQ1. Encoding strategy")
sex_onehot = pd.get_dummies(titanic["sex"], prefix="sex")
log("    sex -> ONE-HOT encoding (no natural order between male/female):")
log(sex_onehot.head().to_string())

pclass_order = {1: 1, 2: 2, 3: 3}  # 1st class is the "highest" class
pclass_ordinal = titanic["pclass"].map(pclass_order)
log("    pclass -> ORDINAL / Map encoding (1st > 2nd > 3rd class is a "
    "natural order, so we explicitly map 1,2,3 instead of letting a "
    "generic encoder assign arbitrary integers):")
log(pclass_ordinal.head().to_string())

# Q2: log-transform + min-max scaling of fare
fare = titanic["fare"].dropna()
fare = fare[fare > 0]  # log is undefined at 0
log_fare = np.log(fare)
minmax_fare = (fare - fare.min()) / (fare.max() - fare.min())
minmax_log_fare = (log_fare - log_fare.min()) / (log_fare.max() - log_fare.min())

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(fare, bins=30, color="#4C72B0")
axes[0].set_title(f"fare (raw)\nskew={fare.skew():.2f}")
axes[1].hist(log_fare, bins=30, color="#DD8452")
axes[1].set_title(f"log(fare)\nskew={log_fare.skew():.2f}")
axes[2].hist(minmax_log_fare, bins=30, color="#55A868")
axes[2].set_title("log(fare) + min-max scaled\nrange [0,1]")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "q2_fare_transformation.png"), dpi=150)
plt.close(fig)

log(f"\nQ2. fare skewness before log-transform: {fare.skew():.3f} "
    "(strongly right-skewed: most passengers paid low fares, a few paid "
    "very high fares)")
log(f"    fare skewness after log-transform: {log_fare.skew():.3f} "
    "-> much closer to a symmetric / normal shape")
log(f"    min-max scaled fare range: [{minmax_fare.min():.3f}, "
    f"{minmax_fare.max():.3f}] (raw), "
    f"[{minmax_log_fare.min():.3f}, {minmax_log_fare.max():.3f}] "
    "(after log-transform)")
log("    -> log-transform fixes the skewness, min-max scaling afterwards "
    "puts the values into a comparable [0,1] range for modelling")

# ---------------------------------------------------------------------------
with open(os.path.join(HERE, "results.md"), "w", encoding="utf-8") as fh:
    fh.write("# Exercise I - Data Exploration: Results\n\n```\n")
    fh.write("\n".join(log_lines))
    fh.write("\n```\n")

log("\nAll outputs written to:", OUT)
log("Written answer sheet: results.md")
