# Exercise I - Data Exploration: Results

```

======================================================================
A. DATA CLEANING ASSIGNMENT - Iris dataset
======================================================================
Q1. Dataset imported: 150 observations, 4 numerical variables + 1 species label
Q2. Variables included: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'species']
    -> 4 numerical variables (sepal length/width, petal length/width) describing flower geometry, 1 categorical label (species: [np.str_('setosa'), np.str_('versicolor'), np.str_('virginica')])
    -> 150 observations (50 samples per species)
Q3. Renamed columns -> {'a1': 'sepal length (cm)', 'a2': 'sepal width (cm)', 'a3': 'petal length (cm)', 'a4': 'petal width (cm)'}
    a1   a2   a3   a4 species
0  5.1  3.5  1.4  0.2  setosa
1  4.9  3.0  1.4  0.2  setosa
2  4.7  3.2  1.3  0.2  setosa
3  4.6  3.1  1.5  0.2  setosa
4  5.0  3.6  1.4  0.2  setosa
Q4. Missing values per column:
a1         0
a2         0
a3         0
a4         0
species    0
    -> Total missing values: 0 (the Iris dataset is clean, no imputation necessary)

======================================================================
B. EXPLORATORY DATA ANALYSIS ASSIGNMENT - Iris dataset
======================================================================
Q1. Saved outputs/q1_histogram_binsizes.png
    -> bins=5: too coarse, the bimodal/overlapping structure of the species disappears
    -> bins=10: a good trade-off, the overall shape (roughly normal, slightly right-skewed) is visible
    -> bins=100: too fine, almost every bin holds 0-2 points -> the histogram becomes noisy and random variation looks like structure
Q2. Saved outputs/q2_boxplot_sepal_length_by_species.png
            min    25%  50%  75%  max
species                              
setosa      4.3  4.800  5.0  5.2  5.8
versicolor  4.9  5.600  5.9  6.3  7.0
virginica   4.9  6.225  6.5  6.9  7.9
    -> setosa has clearly the smallest sepal length and little overlap with the other two species
    -> versicolor and virginica overlap considerably in sepal length, so this attribute alone cannot cleanly separate them (unlike petal length/width, see scatter matrix)
Q3. Saved outputs/q3_scatter_matrix.png
Correlation matrix:
      a1    a2    a3    a4
a1  1.00 -0.12  0.87  0.82
a2 -0.12  1.00 -0.43 -0.37
a3  0.87 -0.43  1.00  0.96
a4  0.82 -0.37  0.96  1.00
    -> a3 (petal length) and a4 (petal width) are almost perfectly correlated (r=0.96) and together separate the three species almost perfectly -> most useful features for classification
    -> a1 (sepal length) correlates fairly strongly with a3/a4 too, while a2 (sepal width) is only weakly / negatively related to the others and shows the most overlap between species

======================================================================
C. FEATURE ENGINEERING ASSIGNMENT - Titanic dataset
======================================================================
Titanic dataset loaded (online: True), 891 rows

Q1. Encoding strategy
    sex -> ONE-HOT encoding (no natural order between male/female):
   sex_female  sex_male
0       False      True
1        True     False
2        True     False
3        True     False
4       False      True
    pclass -> ORDINAL / Map encoding (1st > 2nd > 3rd class is a natural order, so we explicitly map 1,2,3 instead of letting a generic encoder assign arbitrary integers):
0    3
1    1
2    3
3    1
4    3

Q2. fare skewness before log-transform: 4.770 (strongly right-skewed: most passengers paid low fares, a few paid very high fares)
    fare skewness after log-transform: 0.901 -> much closer to a symmetric / normal shape
    min-max scaled fare range: [0.000, 1.000] (raw), [0.000, 1.000] (after log-transform)
    -> log-transform fixes the skewness, min-max scaling afterwards puts the values into a comparable [0,1] range for modelling
```
