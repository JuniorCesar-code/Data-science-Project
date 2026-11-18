# Exercise IV - Decision Trees: Results

```
======================================================================
ASSIGNMENT I: DECISION TREE ON TITANIC (NON-NUMERICAL FEATURES)
======================================================================
Titanic loaded (online=True), 891 rows, columns: ['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'class', 'who', 'adult_male', 'deck', 'embark_town', 'alive', 'alone']
Selected non-numerical features: ['sex', 'class', 'embarked', 'who', 'alone'] (889 rows after dropping missing values)

Q1. ID3-style decision tree trained (criterion=entropy, min_samples_split=20, min_samples_leaf=10)
Q3. Accuracy = 77.15%
    Precision (not survived / survived) = 78.0% / 75.3%
    Recall    (not survived / survived) = 87.9% / 59.8%

    Accuracy = share of ALL passengers classified correctly (regardless of
      class). A single overall number, can hide that one class is
      predicted far better than the other.
    Precision (per class) = of all passengers PREDICTED to be in that
      class, how many actually are -> measures how trustworthy a positive
      prediction is.
    Recall (per class) = of all passengers ACTUALLY in that class, how
      many were found -> measures how complete the detection of that class
      is.
    A model biased towards the majority class (not survived) will show high
    accuracy and high recall for "not survived" but lower recall for
    "survived", exactly as seen above.

Q2. Feature importances (information gain contribution):
who_man         0.595229
class_Third     0.244727
class_First     0.051799
embarked_S      0.039487
alone           0.039092
embarked_C      0.016161
class_Second    0.008309
who_woman       0.002661
    -> The tree splits first/most strongly on 'who_man' ('who'=man/woman/child is effectively a refined version of sex), consistent with the course's own finding that sex is the dominant predictor of survival, refined afterwards by passenger class.

======================================================================
ASSIGNMENT II: DECISION TREE ON TIC-TAC-TOE WITH HYPERPARAMETER TUNING
======================================================================
Q1. Loaded tic-tac-toe.data: 958 rows, missing values: 0
    All 9 attributes are already non-numerical (one of x/o/b per board square) -> no numerical features to exclude; used as is.

Q2. Testing 64 hyperparameter combinations (min_samples_split x min_samples_leaf x min_impurity_decrease/'minimal gain')

Top 10 configurations by F1-score:
 min_samples_split  min_samples_leaf  min_impurity_decrease  accuracy       f1  n_leaves  depth
                 2                 1                   0.00  0.940972 0.954667        72     13
                 2                 5                   0.00  0.934028 0.949333        38      8
                10                 5                   0.00  0.934028 0.949333        38      8
                10                 1                   0.00  0.927083 0.943700        39      8
                20                 5                   0.00  0.920139 0.938005        30      7
                20                 1                   0.00  0.920139 0.938005        30      7
                 2                10                   0.00  0.906250 0.930591        28      6
                10                10                   0.00  0.906250 0.930591        28      6
                20                10                   0.00  0.906250 0.930591        28      6
                10                 5                   0.01  0.829861 0.884706        14      6

Q3. Best configuration by F1-score: min_samples_split=2, min_samples_leaf=1, min_impurity_decrease=0.0 -> accuracy=94.10%, F1=0.955, tree has 72 leaves / depth 13
    Unconstrained growth (min_impurity_decrease=0, leaf size 1) wins on raw F1 here, at the cost of a large, overfitting-prone tree. A more interpretable runner-up barely sacrifices performance, e.g. 2/10/0.00 reaches 90.6% accuracy with only 28 leaves - illustrating the classic accuracy-vs-interpretability trade-off pre-pruning controls. A non-zero minimal gain (0.05-0.10) clearly over-prunes on this dataset: it collapses the tree so much that most of the genuinely necessary 'three-in-a-row' splits get discarded and F1 drops sharply.

Feature importances (cumulative information gain) of the best tree:
bottom-left_o      0.116094
bottom-right_o     0.094382
top-left_o         0.087994
bottom-left_x      0.084454
middle-right_o     0.076588
middle-middle_o    0.072019

Root-node split (the single highest information-gain split): 'middle-middle_o'
    -> exactly as in the course's own result, the very first split is on the CENTRE square (middle-middle) - it belongs to 4 of the 8 possible winning lines and is therefore the most informative single square. Cumulative importance over the whole tree is dominated by the corner squares instead (3 winning lines each, but many more nodes deeper in the tree depend on them), while edge squares (2 winning lines each) contribute the least either way.
    Despite pruning, the full best tree still has 72 leaves at depth 13 - too large to visualize/interpret completely, which matches the course's own observation that ID3 trees on this dataset remain large even after pruning (scikit-learn, unlike AI Studio's ID3 operator, would additionally let us cap max_depth directly to force a compact tree).
```
