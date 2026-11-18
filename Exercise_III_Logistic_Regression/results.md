# Exercise III - Logistic Regression: Results

```
======================================================================
1. SIGMOID FUNCTION  sigma(z) = e^z / (1 + e^z)
======================================================================
Saved outputs/sigmoid_function.png

======================================================================
2. ODDS / LOGIT / ODDS RATIO - worked examples
======================================================================
Rot-Weiss Essen: odds-in-favor = 4:2 -> odds = 2.00 (matches slide value 2.0)
  probability of winning = wins/(wins+losses) = 0.6667 (matches slide value ~0.66)
  check odds from probability: p/(1-p) = 2.00 (matches slide value 2.0)
  logit(odds=1/4) = log(0.25) = -1.39
  logit(odds=4/1) = log(4.00) = 1.39

Odds ratio example (gene mutation vs cancer):
  odds(exposed) = 23/117 = 0.1966
  odds(control) = 6/210 = 0.0286
  odds ratio = 6.88 (matches slide value 6.88)
  -> the odds of having cancer are ~6.88x higher for someone with the mutated gene

======================================================================
3. COEFFICIENT INTERPRETATION EXAMPLE (slide values beta0=-0.884, beta1=0.012 for passenger fare)
======================================================================
logit = beta0 + beta1*fare = -0.884 + 0.012*50 = -0.284
p(survival | fare=$50) = sigma(logit) = 0.429
  Note: the slide claims p=0.73 for this fare, but plugging the slide's own beta0/beta1 into the sigmoid gives p=0.429 - the slide's 0.73 example and its coefficients are not numerically consistent (likely a different fare value, e.g. ~$157, was used when that example was drawn: sigma(-0.884+0.012*157)=sigma(1.0)=0.73).
Odds ratio for fare = exp(beta1) = exp(0.012) = 1.0121 (matches slide value 1.012)
Percentage change in odds per $1 fare increase = (OR - 1) * 100 = 1.21% (matches slide value ~1.2%)
Interpretation of intercept beta0=-0.884: when fare = $0, the log-odds of survival are -0.884 (not practically meaningful, since no passenger paid $0)

======================================================================
4. HANDS-ON LOGISTIC REGRESSION ON TIC-TAC-TOE DATASET
======================================================================
Loaded tic-tac-toe.data: 958 rows, target 'class' distribution: {'positive': 626, 'negative': 332}
Logistic regression test accuracy: 0.986

Top 5 features increasing the odds of an 'x' win most:
        feature  coefficient  odds_ratio
middle-middle_x     3.145127   23.222617
  bottom-left_x     2.873360   17.696373
    top-right_x     2.765093   15.880510
 bottom-right_x     2.753802   15.702225
     top-left_x     2.741445   15.509387

Top 5 features decreasing the odds of an 'x' win most:
        feature  coefficient  odds_ratio
 bottom-right_o    -2.901218    0.054956
     top-left_o    -2.921425    0.053857
    top-right_o    -2.968685    0.051371
  bottom-left_o    -3.039342    0.047866
middle-middle_o    -3.241154    0.039119

Interpretation: squares being marked 'x' in the centre or corners (part of
many winning lines) show the largest positive coefficients / odds ratios
greater than 1 - consistent with the fact that the centre square belongs to
4 of the 8 possible winning lines, and corners to 3 each. Squares marked 'o'
on these same high-value positions show the largest negative coefficients,
since an opponent mark there blocks the most winning lines for 'x'.

```
