# Data Science: Concepts and Practice - Exercises I-V (Python solutions)

All five exercise decks (originally built around Altair AI Studio / RapidMiner)
re-implemented and solved in Python. Each folder is self-contained: run the
`.py` script, read `results.md` for the written answers, and look in
`outputs/` for the generated plots/tables.

| Folder | Dataset(s) | Topics |
|---|---|---|
| `Exercise_I_Data_Exploration` | Iris (sklearn), Titanic (online) | descriptive stats, missing values, histograms/boxplots/scatter matrix, encoding, log-transform & min-max scaling |
| `Exercise_II_Clustering` | Mall_Customers.csv | AGNES (6 linkage/metric combos), K-Means + elbow method, DBSCAN + k-distance graph |
| `Exercise_III_Logistic_Regression` | worked slide examples + tic-tac-toe.data | sigmoid, odds/logit/odds-ratio, coefficient interpretation, hands-on logistic regression |
| `Exercise_IV_Decision_Tree` | Titanic (online), tic-tac-toe.data | entropy/ID3-style trees, confusion matrix, accuracy/precision/recall, hyperparameter search over pruning parameters |
| `Exercise_V_Regression` | diabetes.txt, Titanic (online) | multiple linear regression + R², Lasso (L1) vs Ridge (L2) with lambda search, logistic regression + odds ratios |

## Requirements
```
pip install pandas numpy matplotlib scikit-learn scipy seaborn statsmodels
```

## Notes
- The Titanic dataset wasn't among your uploaded files (the slides only
  reference it / link to Kaggle), so the scripts download it from a public
  mirror at runtime. If you are offline, each script falls back to a small
  synthetic dataset with the same columns so the code still runs end-to-end
  — just re-run with internet access to get the real numbers.
- Where a slide's worked example contained a numeric inconsistency (flagged
  explicitly in `Exercise_III`'s results), the script reports the actual
  computed value rather than forcing a match.
- The three `.rmp` RapidMiner files and the `diabetes.rmhdf5table` binary
  file you provided were used as the specification for Exercise II/V
  (confirming feature choices, normalization, and hyperparameter ranges) —
  the executable logic itself is the Python code in this folder.
