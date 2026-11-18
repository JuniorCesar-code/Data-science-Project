# Exercise V - Linear & Logistic Regression: Results

```
======================================================================
PART 1, Q1: MULTIPLE LINEAR REGRESSION (ALL FEATURES EXCEPT SEX)
======================================================================
Loaded diabetes.txt: 442 rows, columns: ['AGE', 'SEX', 'BMI', 'BP', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'Y']
Features used (SEX excluded as instructed): ['AGE', 'BMI', 'BP', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6']

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      Y   R-squared:                       0.501
Model:                            OLS   Adj. R-squared:                  0.490
Method:                 Least Squares   F-statistic:                     48.11
Date:                Wed, 18 Nov 2026   Prob (F-statistic):           8.80e-60
Time:                        11:21:28   Log-Likelihood:                -2393.7
No. Observations:                 442   AIC:                             4807.
Df Residuals:                     432   BIC:                             4848.
Df Model:                           9                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       -363.8987     68.142     -5.340      0.000    -497.829    -229.968
AGE           -0.1205      0.220     -0.549      0.583      -0.552       0.311
BMI            6.0041      0.721      8.322      0.000       4.586       7.422
BP             0.9505      0.225      4.227      0.000       0.509       1.392
S1            -0.9808      0.582     -1.685      0.093      -2.125       0.163
S2             0.6585      0.539      1.221      0.223      -0.401       1.718
S3             0.5136      0.794      0.646      0.518      -1.048       2.075
S4             4.6599      6.037      0.772      0.441      -7.206      16.526
S5            68.9473     15.927      4.329      0.000      37.643     100.252
S6             0.2026      0.277      0.731      0.465      -0.342       0.747
==============================================================================
Omnibus:                        5.601   Durbin-Watson:                   2.018
Prob(Omnibus):                  0.061   Jarque-Bera (JB):                4.046
Skew:                           0.094   Prob(JB):                        0.132
Kurtosis:                       2.571   Cond. No.                     7.19e+03
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 7.19e+03. This might indicate that there are
strong multicollinearity or other numerical problems.

Interpretation:
  R-squared = 0.5006 -> the model explains 50.1% of the variance in the diabetes progression score Y
  Adjusted R-squared = 0.4902 (slightly lower, correcting for the number of predictors used)

  Coefficients, sorted by absolute magnitude:
S5     68.947342
BMI     6.004066
S4      4.659881
S1     -0.980784
BP      0.950508
S2      0.658496
S3      0.513628
S6      0.202625
AGE    -0.120515

  Statistically significant predictors (p < 0.05): ['BMI', 'BP', 'S5']
  BMI and S5 show by far the largest positive coefficients: higher body-mass index and higher S5 (log of serum triglycerides level) are associated with a markedly worse (higher) diabetes progression score, holding the other variables constant. BP (blood pressure) also has a clear positive effect. Several S-variables (S1-S4, S6) are highly correlated with each other (multicollinearity), which is why some of them are not individually significant despite the model's good overall fit.

======================================================================
PART 1, Q2: REGULARIZED REGRESSION (L1 vs. L2, LAMBDA SEARCH)
======================================================================
L1 (alpha=1, Lasso): best lambda found via 10-fold CV = 1.5199
L2 (alpha=0, Ridge): best lambda found via 10-fold CV = 31.2572

Coefficient comparison (standardized features, so coefficients are directly comparable in magnitude):
         OLS (no regularization)  L1 / Lasso (alpha=1)  L2 / Ridge (alpha=0)
feature                                                                     
AGE                       -1.578                -0.000                -0.993
BMI                       26.497                26.039                25.128
BP                        13.132                11.787                12.776
S1                       -33.905                -4.329                -3.961
S2                        20.004                -0.000                -3.397
S3                         6.636                -6.794                -6.280
S4                         6.007                 0.000                 3.711
S5                        35.977                24.736                22.847
S6                         2.327                 1.267                 3.168

Interpretation:
  L1/Lasso sets 3 coefficient(s) exactly to zero (['AGE', 'S2', 'S4']) -> Lasso performs automatic feature selection by shrinking the least useful / most collinear predictors all the way to 0.
  L2/Ridge shrinks every coefficient towards zero but keeps all of them non-zero and tends to spread the effect across correlated predictors (e.g. S1-S4) more evenly than OLS, which can assign them large, unstable, sometimes even sign-flipped coefficients.
  Both regularized models therefore have smaller coefficient magnitudes overall than the unregularized OLS fit, trading a little bias for a lot less variance - this is most visible for BMI/BP/S5 which dominate the OLS fit but are pulled in by both penalties.

======================================================================
PART 2: LOGISTIC REGRESSION ON SEX, PASSENGER CLASS, PASSENGER FARE
======================================================================
Titanic loaded (online=True), 891 rows

                           Logit Regression Results                           
==============================================================================
Dep. Variable:               survived   No. Observations:                  891
Model:                          Logit   Df Residuals:                      886
Method:                           MLE   Df Model:                            4
Date:                Wed, 18 Nov 2026   Pseudo R-squ.:                  0.3038
Time:                        11:21:34   Log-Likelihood:                -413.06
converged:                       True   LL-Null:                       -593.33
Covariance Type:            nonrobust   LLR p-value:                 9.314e-77
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const          2.1476      0.278      7.716      0.000       1.602       2.693
sex_male      -2.6246      0.185    -14.170      0.000      -2.988      -2.262
pclass_2      -0.7366      0.271     -2.717      0.007      -1.268      -0.205
pclass_3      -1.7903      0.252     -7.111      0.000      -2.284      -1.297
fare           0.0018      0.002      0.864      0.388      -0.002       0.006
==============================================================================

Q1/Q2. Coefficients and odds ratios:
          coefficient (log-odds)  odds_ratio  p_value
const                     2.1476      8.5639   0.0000
sex_male                 -2.6246      0.0725   0.0000
pclass_2                 -0.7366      0.4787   0.0066
pclass_3                 -1.7903      0.1669   0.0000
fare                      0.0018      1.0018   0.3878

Interpretation:
  sex_male=1 (male): multiplies the odds of survival by 0.0725 relative to female passengers -> being male decreases the odds of survival by 92.8% - the single strongest effect in the model, matching 'women and children first'.
  pclass_2=1: multiplies the odds of survival by 0.4787 relative to the baseline (1st class) -> 52.1% decrease in survival odds for that passenger class.
  pclass_3=1: multiplies the odds of survival by 0.1669 relative to the baseline (1st class) -> 83.3% decrease in survival odds for that passenger class.
  fare: each additional $1 of fare multiplies the odds of survival by 1.0018 (0.2% increase per dollar) -> on its own, higher fare would suggest slightly higher survival odds, but this effect is NOT statistically significant (p=0.388 > 0.05) once sex and class are already in the model - fare's influence on survival is largely already captured by passenger class.
```
