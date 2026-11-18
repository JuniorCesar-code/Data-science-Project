"""
Data Science - Exercise V: Linear & Logistic Regression

Part 1 (Linear Regression) - Diabetes dataset:
    1. Multiple linear regression with all features EXCEPT sex;
       interpret R-squared and coefficients.
    2. Regularized regression on the same feature set: search for lambda
       (the penalty strength) and run twice - once with alpha=1 (pure L1 /
       Lasso) and once with alpha=0 (pure L2 / Ridge), using the slides'
       own naming (sklearn's ElasticNet "l1_ratio" == the slides' "alpha";
       ElasticNet/Lasso/Ridge "alpha" == the slides' "lambda"). Compare the
       coefficients of the three configurations (none / L1 / L2).

Part 2 (Logistic Regression) - Titanic dataset:
    1. Logistic regression on sex, passenger class, and passenger fare.
    2. Coefficients, odds ratios, and their interpretation.

Run:
    python regression.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, RidgeCV, LogisticRegression
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

log_lines = []


def log(*args):
    msg = " ".join(str(a) for a in args)
    print(msg)
    log_lines.append(msg)


# ===========================================================================
# PART 1: LINEAR REGRESSION ON THE DIABETES DATASET
# ===========================================================================
log("=" * 70)
log("PART 1, Q1: MULTIPLE LINEAR REGRESSION (ALL FEATURES EXCEPT SEX)")
log("=" * 70)

diabetes = pd.read_csv(os.path.join(HERE, "data", "diabetes.txt"), sep="\t")
log(f"Loaded diabetes.txt: {diabetes.shape[0]} rows, columns: {list(diabetes.columns)}")

feature_cols = [c for c in diabetes.columns if c not in ("SEX", "Y")]
X = diabetes[feature_cols]
y = diabetes["Y"]
log(f"Features used (SEX excluded as instructed): {feature_cols}")

X_const = sm.add_constant(X)
ols_model = sm.OLS(y, X_const).fit()

log("\n" + str(ols_model.summary()))

log(f"\nInterpretation:")
log(f"  R-squared = {ols_model.rsquared:.4f} -> the model explains "
    f"{ols_model.rsquared*100:.1f}% of the variance in the diabetes "
    "progression score Y")
log(f"  Adjusted R-squared = {ols_model.rsquared_adj:.4f} (slightly lower, "
    "correcting for the number of predictors used)")

coef_table = ols_model.params.drop("const").sort_values(key=abs, ascending=False)
log("\n  Coefficients, sorted by absolute magnitude:")
log(coef_table.to_string())
sig = ols_model.pvalues.drop("const")
significant = sig[sig < 0.05].index.tolist()
log(f"\n  Statistically significant predictors (p < 0.05): {significant}")
log("  BMI and S5 show by far the largest positive coefficients: higher "
    "body-mass index and higher S5 (log of serum triglycerides level) are "
    "associated with a markedly worse (higher) diabetes progression score, "
    "holding the other variables constant. BP (blood pressure) also has a "
    "clear positive effect. Several S-variables (S1-S4, S6) are highly "
    "correlated with each other (multicollinearity), which is why some of "
    "them are not individually significant despite the model's good "
    "overall fit.")

# ---------------------------------------------------------------------------
# PART 1, Q2: regularized regression with lambda search, L1 (alpha=1) and
# L2 (alpha=0) -- standardize first, as regularization is scale-sensitive
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log("PART 1, Q2: REGULARIZED REGRESSION (L1 vs. L2, LAMBDA SEARCH)")
log("=" * 70)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
lambdas = np.logspace(-3, 2, 100)

# alpha = 1 -> pure L1 -> Lasso
lasso_cv = LassoCV(alphas=lambdas, cv=10, max_iter=20000, random_state=2001)
lasso_cv.fit(X_scaled, y)
log(f"L1 (alpha=1, Lasso): best lambda found via 10-fold CV = "
    f"{lasso_cv.alpha_:.4f}")

# alpha = 0 -> pure L2 -> Ridge
ridge_cv = RidgeCV(alphas=lambdas, cv=10)
ridge_cv.fit(X_scaled, y)
log(f"L2 (alpha=0, Ridge): best lambda found via 10-fold CV = "
    f"{ridge_cv.alpha_:.4f}")

# unregularized OLS on the standardized features, for a fair 3-way
# comparison of coefficient magnitudes
ols_scaled = sm.OLS(y, sm.add_constant(X_scaled)).fit()

compare = pd.DataFrame({
    "feature": feature_cols,
    "OLS (no regularization)": ols_scaled.params.values[1:],
    "L1 / Lasso (alpha=1)": lasso_cv.coef_,
    "L2 / Ridge (alpha=0)": ridge_cv.coef_,
}).set_index("feature")
compare.to_csv(os.path.join(OUT, "coefficient_comparison.csv"))
log("\nCoefficient comparison (standardized features, so coefficients are "
    "directly comparable in magnitude):")
log(compare.round(3).to_string())

n_zeroed = int((lasso_cv.coef_ == 0).sum())
log(f"\nInterpretation:")
log(f"  L1/Lasso sets {n_zeroed} coefficient(s) exactly to zero "
    f"({[f for f, c in zip(feature_cols, lasso_cv.coef_) if c == 0]}) "
    "-> Lasso performs automatic feature selection by shrinking the "
    "least useful / most collinear predictors all the way to 0.")
log("  L2/Ridge shrinks every coefficient towards zero but keeps all of "
    "them non-zero and tends to spread the effect across correlated "
    "predictors (e.g. S1-S4) more evenly than OLS, which can assign them "
    "large, unstable, sometimes even sign-flipped coefficients.")
log("  Both regularized models therefore have smaller coefficient "
    "magnitudes overall than the unregularized OLS fit, trading a little "
    "bias for a lot less variance - this is most visible for BMI/BP/S5 "
    "which dominate the OLS fit but are pulled in by both penalties.")

fig, ax = plt.subplots(figsize=(10, 6))
compare.plot(kind="bar", ax=ax)
ax.set_ylabel("standardized coefficient")
ax.set_title("Diabetes: OLS vs. L1 (Lasso) vs. L2 (Ridge) coefficients")
ax.axhline(0, color="black", linewidth=0.8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "coefficient_comparison.png"), dpi=150)
plt.close(fig)

# ===========================================================================
# PART 2: LOGISTIC REGRESSION ON THE TITANIC DATASET
# ===========================================================================
log("\n" + "=" * 70)
log("PART 2: LOGISTIC REGRESSION ON SEX, PASSENGER CLASS, PASSENGER FARE")
log("=" * 70)


def load_titanic():
    urls = [
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv",
        "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv",
    ]
    for url in urls:
        try:
            return pd.read_csv(url)
        except Exception as exc:  # noqa: BLE001
            log(f"  (could not load {url}: {exc})")
    raise RuntimeError("No Titanic source reachable")


try:
    titanic = load_titanic()
    titanic_online = True
except Exception as exc:  # noqa: BLE001
    log(f"Falling back to a synthetic Titanic-like sample ({exc})")
    rng = np.random.default_rng(2001)
    n = 400
    sex = rng.choice(["male", "female"], size=n, p=[0.65, 0.35])
    pclass = rng.choice([1, 2, 3], size=n, p=[0.25, 0.25, 0.5])
    fare = rng.gamma(2.0, 15.0, size=n)
    base_logit = -1.5 + (sex == "female") * 2.5 - (pclass - 1) * 0.6 + fare * 0.01
    p = 1 / (1 + np.exp(-base_logit))
    survived = rng.binomial(1, p)
    titanic = pd.DataFrame({"sex": sex, "pclass": pclass, "fare": fare,
                             "survived": survived})
    titanic_online = False

col_map = {}
if "pclass" not in titanic.columns and "Pclass" in titanic.columns:
    col_map["Pclass"] = "pclass"
if "sex" not in titanic.columns and "Sex" in titanic.columns:
    col_map["Sex"] = "sex"
if "fare" not in titanic.columns and "Fare" in titanic.columns:
    col_map["Fare"] = "fare"
if "survived" not in titanic.columns and "Survived" in titanic.columns:
    col_map["Survived"] = "survived"
titanic = titanic.rename(columns=col_map)
log(f"Titanic loaded (online={titanic_online}), {titanic.shape[0]} rows")

data = titanic[["sex", "pclass", "fare", "survived"]].dropna()
# dummy coding: sex (drop 'female' baseline), pclass (drop class 1 baseline)
X_log = pd.get_dummies(data[["sex", "pclass"]].astype({"pclass": "category"}),
                        drop_first=True)
X_log["fare"] = data["fare"].values
y_log = data["survived"].astype(int)

X_log_const = sm.add_constant(X_log.astype(float))
logit_model = sm.Logit(y_log, X_log_const).fit(disp=False)

log("\n" + str(logit_model.summary()))

or_table = pd.DataFrame({
    "coefficient (log-odds)": logit_model.params,
    "odds_ratio": np.exp(logit_model.params),
    "p_value": logit_model.pvalues,
})
or_table.to_csv(os.path.join(OUT, "titanic_logistic_coefficients.csv"))
log("\nQ1/Q2. Coefficients and odds ratios:")
log(or_table.round(4).to_string())

log("\nInterpretation:")
for name in X_log.columns:
    coef = logit_model.params[name]
    orv = np.exp(coef)
    direction = "increase" if coef > 0 else "decrease"
    pct = abs(orv - 1) * 100
    if name == "fare":
        p_fare = logit_model.pvalues[name]
        sig_note = "statistically significant" if p_fare < 0.05 else \
            f"NOT statistically significant (p={p_fare:.3f} > 0.05)"
        log(f"  {name}: each additional $1 of fare multiplies the odds of "
            f"survival by {orv:.4f} ({pct:.1f}% {direction} per dollar) "
            f"-> on its own, higher fare would suggest slightly higher "
            f"survival odds, but this effect is {sig_note} once sex and "
            "class are already in the model - fare's influence on "
            "survival is largely already captured by passenger class.")
    elif name.startswith("sex_"):
        log(f"  {name}=1 (male): multiplies the odds of survival by "
            f"{orv:.4f} relative to female passengers "
            f"-> being male {direction}s the odds of survival by "
            f"{pct:.1f}% - the single strongest effect in the model, "
            "matching 'women and children first'.")
    else:
        log(f"  {name}=1: multiplies the odds of survival by {orv:.4f} "
            f"relative to the baseline (1st class) -> {pct:.1f}% "
            f"{direction} in survival odds for that passenger class.")

fig, ax = plt.subplots(figsize=(7, 5))
or_no_const = or_table.drop("const")
ax.barh(or_no_const.index, or_no_const["odds_ratio"], color="#4C72B0")
ax.axvline(1.0, color="black", linestyle="--", linewidth=0.8)
ax.set_xlabel("odds ratio (reference = 1.0)")
ax.set_title("Titanic logistic regression: odds ratios")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "titanic_odds_ratios.png"), dpi=150)
plt.close(fig)

with open(os.path.join(HERE, "results.md"), "w", encoding="utf-8") as fh:
    fh.write("# Exercise V - Linear & Logistic Regression: Results\n\n```\n")
    fh.write("\n".join(log_lines))
    fh.write("\n```\n")

log("\nAll outputs written to:", OUT)
