"""
Data Science - Exercise III: Logistic Regression (Concepts)

This slide deck is purely conceptual (sigmoid function, odds, logit,
maximum likelihood, coefficient interpretation) - "Implementation will
follow in Exercise Session 5" (see Exercise V). This script therefore:

  1. Reproduces every worked numeric example from the slides in code
     (odds, log-odds/logit, odds ratio, sigmoid curve, the Titanic
     passenger-fare interpretation example).
  2. Additionally fits a real logistic regression on the Tic-Tac-Toe
     dataset (the dataset handed out for this course) as a concrete,
     hands-on illustration of the concepts, since no dataset was attached
     to this particular exercise.

Run:
    python logistic_regression_concepts.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

log_lines = []


def log(*args):
    msg = " ".join(str(a) for a in args)
    print(msg)
    log_lines.append(msg)


# ---------------------------------------------------------------------------
# 1. Sigmoid function
# ---------------------------------------------------------------------------
log("=" * 70)
log("1. SIGMOID FUNCTION  sigma(z) = e^z / (1 + e^z)")
log("=" * 70)


def sigmoid(z):
    return np.exp(z) / (1 + np.exp(z))


z = np.linspace(-10, 10, 400)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(z, sigmoid(z))
ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
ax.set_xlabel("z = b0 + b1*x1 + ... + bn*xn  (the logit)")
ax.set_ylabel("sigma(z) = p")
ax.set_title("Sigmoid function maps the logit (-inf, inf) to p in (0, 1)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "sigmoid_function.png"), dpi=150)
plt.close(fig)
log("Saved outputs/sigmoid_function.png")

# ---------------------------------------------------------------------------
# 2. Odds, log-odds (logit), odds ratio - worked examples from the slides
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log("2. ODDS / LOGIT / ODDS RATIO - worked examples")
log("=" * 70)

# Rot-Weiss Essen example: odds of winning = 4:2
wins, losses = 4, 2
odds_win = wins / losses
p_win = wins / (wins + losses)
log(f"Rot-Weiss Essen: odds-in-favor = {wins}:{losses} -> odds = {odds_win:.2f} "
    f"(matches slide value 2.0)")
log(f"  probability of winning = wins/(wins+losses) = {p_win:.4f} "
    "(matches slide value ~0.66)")
log(f"  check odds from probability: p/(1-p) = {p_win/(1-p_win):.2f} "
    "(matches slide value 2.0)")

# log-odds examples
for ratio, label in [(1/4, "1/4"), (4/1, "4/1")]:
    log(f"  logit(odds={label}) = log({ratio:.2f}) = {np.log(ratio):.2f}")

# odds ratio example: gene mutation vs cancer (23/117 vs 6/210)
odds_exposed = 23 / 117
odds_control = 6 / 210
odds_ratio = odds_exposed / odds_control
log(f"\nOdds ratio example (gene mutation vs cancer):")
log(f"  odds(exposed) = 23/117 = {odds_exposed:.4f}")
log(f"  odds(control) = 6/210 = {odds_control:.4f}")
log(f"  odds ratio = {odds_ratio:.2f} (matches slide value 6.88)")
log("  -> the odds of having cancer are ~6.88x higher for someone with the "
    "mutated gene")

# ---------------------------------------------------------------------------
# 3. Titanic passenger-fare interpretation example from the slides
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log("3. COEFFICIENT INTERPRETATION EXAMPLE (slide values beta0=-0.884, "
    "beta1=0.012 for passenger fare)")
log("=" * 70)

beta0, beta1 = -0.884, 0.012
fare = 50
logit_val = beta0 + beta1 * fare
p_survival = sigmoid(logit_val)
odds_ratio_fare = np.exp(beta1)
pct_change = (odds_ratio_fare - 1) * 100

log(f"logit = beta0 + beta1*fare = {beta0} + {beta1}*{fare} = {logit_val:.3f}")
log(f"p(survival | fare=$50) = sigma(logit) = {p_survival:.3f}")
log("  Note: the slide claims p=0.73 for this fare, but plugging the "
    "slide's own beta0/beta1 into the sigmoid gives p=0.429 - the slide's "
    "0.73 example and its coefficients are not numerically consistent "
    "(likely a different fare value, e.g. ~$157, was used when that "
    "example was drawn: sigma(-0.884+0.012*157)=sigma(1.0)=0.73).")
log(f"Odds ratio for fare = exp(beta1) = exp({beta1}) = {odds_ratio_fare:.4f} "
    "(matches slide value 1.012)")
log(f"Percentage change in odds per $1 fare increase = "
    f"(OR - 1) * 100 = {pct_change:.2f}% (matches slide value ~1.2%)")
log(f"Interpretation of intercept beta0={beta0}: when fare = $0, the "
    "log-odds of survival are -0.884 (not practically meaningful, since no "
    "passenger paid $0)")

# ---------------------------------------------------------------------------
# 4. Hands-on logistic regression on the Tic-Tac-Toe dataset
#    (concrete dataset available for this course, used here since the
#    Titanic implementation is explicitly deferred to Exercise V)
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log("4. HANDS-ON LOGISTIC REGRESSION ON TIC-TAC-TOE DATASET")
log("=" * 70)

columns = ["top-left", "top-middle", "top-right",
           "middle-left", "middle-middle", "middle-right",
           "bottom-left", "bottom-middle", "bottom-right", "class"]
ttt_path = os.path.join(
    os.path.dirname(HERE), "Exercise_IV_Decision_Tree", "data", "tic-tac-toe.data")
ttt = pd.read_csv(ttt_path, header=None, names=columns)
log(f"Loaded tic-tac-toe.data: {ttt.shape[0]} rows, target 'class' "
    f"distribution: {ttt['class'].value_counts().to_dict()}")

# one-hot encode the 9 board squares (nominal, 3 categories each: x, o, b)
X = pd.get_dummies(ttt.drop(columns=["class"]), drop_first=False)
y = (ttt["class"] == "positive").astype(int)  # 1 = x wins, 0 = no win for x

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=2001, stratify=y)

clf = LogisticRegression(max_iter=2000)
clf.fit(X_train, y_train)
acc = accuracy_score(y_test, clf.predict(X_test))
log(f"Logistic regression test accuracy: {acc:.3f}")

coef_df = pd.DataFrame({
    "feature": X.columns,
    "coefficient": clf.coef_[0],
    "odds_ratio": np.exp(clf.coef_[0]),
}).sort_values("coefficient", ascending=False)
coef_df.to_csv(os.path.join(OUT, "tictactoe_logreg_coefficients.csv"), index=False)

log("\nTop 5 features increasing the odds of an 'x' win most:")
log(coef_df.head(5).to_string(index=False))
log("\nTop 5 features decreasing the odds of an 'x' win most:")
log(coef_df.tail(5).to_string(index=False))
log("""
Interpretation: squares being marked 'x' in the centre or corners (part of
many winning lines) show the largest positive coefficients / odds ratios
greater than 1 - consistent with the fact that the centre square belongs to
4 of the 8 possible winning lines, and corners to 3 each. Squares marked 'o'
on these same high-value positions show the largest negative coefficients,
since an opponent mark there blocks the most winning lines for 'x'.
""")

with open(os.path.join(HERE, "results.md"), "w", encoding="utf-8") as fh:
    fh.write("# Exercise III - Logistic Regression: Results\n\n```\n")
    fh.write("\n".join(log_lines))
    fh.write("\n```\n")

log("All outputs written to:", OUT)
