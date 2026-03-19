# Credit Risk and Portfolio Construction using Machine Learning

Predicting loan default and optimizing investment strategy on LendingClub data.

---

## Overview

The starting question for this project was simple: if you were an anonymous investor on a peer-to-peer lending platform, how would you decide which loans to fund?

LendingClub assigns every loan a grade from A to G. You could just follow their grades and call it a day. But that raises a more interesting question: can you build something better than what LendingClub already gives you? And even if you can predict default well, does that actually translate into better investment returns?

This project works through both questions across three phases: exploratory data analysis, predictive modeling, and portfolio construction with backtesting. The short answer is that predicting default is tractable, predicting returns is hard, and the relationship between the two is more complicated than it looks.

---

## Project Structure

```
credit-risk-portfolio-analytics/
├── README.md
├── requirements.txt
├── data/
│   └── lending_club_loans.csv
└── notebooks/
    ├── 01_investor_framing_and_eda.ipynb
    ├── 02_feature_engineering_and_returns.ipynb
    └── 03_modeling_and_investment_strategies.ipynb
```

The notebooks are meant to be read in order. Each one builds on the previous, and there are analytical decision points throughout where you have to make a choice and justify it.

---

## Data

The dataset comes from LendingClub's public loan data, covering loans issued from approximately 2010 to 2019. The full dataset contains around 2.5 million loan records with 150+ features per loan.

To download the data:
1. Go to LendingClub's public data page. Note that LendingClub ceased operations as a P2P platform in 2020, but historical loan data remains publicly available through Kaggle and other sources.
2. Download the loan-level CSV files.
3. Place them in the `data/` directory.

The notebooks expect a combined file called `lending_club_loans.csv`. See `data/README.md` for the exact steps to merge quarterly files.

**Key features used:**

| Feature | Type | Description |
|---|---|---|
| loan_amnt | Numeric | Requested loan amount |
| int_rate | Numeric | Interest rate on the loan |
| grade / sub_grade | Categorical | LendingClub's risk grade (derived feature -- see Phase 3) |
| annual_inc | Numeric | Borrower's self-reported annual income |
| dti | Numeric | Debt-to-income ratio |
| fico_range_low/high | Numeric | Borrower FICO score range |
| delinq_2yrs | Numeric | Delinquencies in the past 2 years |
| revol_util | Numeric | Revolving credit utilization |
| home_ownership | Categorical | RENT / OWN / MORTGAGE |
| emp_length | Categorical | Employment length in years |
| loan_status | Target | Fully Paid / Charged Off / Default / etc. |

---

## Phase 1: Investor Framing and EDA

**Notebook:** `01_investor_framing_and_eda.ipynb`

The first notebook is about understanding the data before touching any models. The framing matters here because we are approaching this as an investor trying to maximize risk-adjusted returns, not as a data scientist trying to minimize classification error.

**Key findings:**

Grade B and C loans make up the majority of the portfolio (~28% each), with Grade A representing about 17% and the riskier grades (E, F, G) forming a small tail. Default rates follow the expected gradient:

| Grade | Default Rate | Avg. Interest Rate |
|---|---|---|
| A | 9.1% | 7.2% |
| B | 17.6% | 10.9% |
| C | 28.1% | 14.2% |
| D | 36.6% | 18.1% |
| E | 43.6% | 21.3% |
| F | 49.3% | 24.9% |
| G | 54.6% | 27.5% |

The interesting question is not whether higher grades default less -- of course they do. It is whether higher interest rates on riskier loans compensate for the additional default risk. The answer depends heavily on how you define return, which leads directly into Phase 2.

We define four return metrics that capture different investor assumptions:

- **ret_PESS (Pessimistic):** No reinvestment of recovered principal. Investor simply loses defaulted principal.
- **ret_OPT (Optimistic):** Immediate reinvestment at the risk-free rate on recovered amounts.
- **ret_INTa / ret_INTb (Intermediate):** Return calculated as a function of the interest rate and default timing, under different reinvestment rate assumptions.

Under ret_PESS, Grades C through G produce negative average returns. Under ret_INTb, Grade G loans look attractive. This is not a contradiction -- it reflects genuine uncertainty about the right return model, and it drives the investment strategy design in Phase 3.

---

## Phase 2: Feature Engineering and Return Calculation

**Notebook:** `02_feature_engineering_and_returns.ipynb`

Phase 2 does the preparatory work for modeling: cleaning the data, engineering features, handling the temporal structure, and computing the return variables used as regression targets in Phase 3.

**Notable decisions:**

- `cr_hist` is computed as the length of the borrower's credit history at the time the loan was issued (the difference between `issue_d` and `earliest_cr_line` in months), not the raw date. A credit line opened in 1980 looks very different on a 2010 application versus a 2015 one.
- Percentage fields like `int_rate` and `revol_util` require string cleaning before numeric conversion.
- Outliers in `annual_inc`, `dti`, and `revol_bal` are removed as likely data errors. Outliers in `delinq_2yrs` and `pub_rec` are retained -- a borrower with 20 delinquencies in two years is a genuine data point worth keeping.
- The train/test split is assigned once at the start and held fixed across all models, so every model sees the exact same training and test data.

---

## Phase 3: Modeling and Investment Strategy Backtesting

**Notebook:** `03_modeling_and_investment_strategies.ipynb`

This is the core of the project. It covers default prediction, a data leakage investigation, temporal stability analysis, return regression, and investment strategy backtesting.

### Default Prediction

Six classifiers are trained and evaluated on the task of predicting whether a loan will default. Models are evaluated using AUC-ROC rather than accuracy -- with an ~80% non-default rate, accuracy is a misleading metric. A model that predicts no default for every loan would score 80% accuracy while being completely useless.

**Pass 1: CV, with grade, subset of data**

| Model | Recall | AUC | Brier Score |
|---|---|---|---|
| Naive Bayes | 0.0000 | 0.67 | 0.5363 |
| L1 Logistic Regression | 0.0390 | 0.71 | 0.1454 |
| L2 Logistic Regression | 0.0331 | 0.71 | 0.1453 |
| Decision Tree | 0.0989 | 0.66 | 0.1579 |
| **Random Forest** | **0.1637** | **0.81** | **0.1470** |
| MLP | 0.0408 | 0.70 | 0.1463 |

**Pass 2: CV, only grade, subset of data**

| Model | Recall | AUC | Brier Score |
|---|---|---|---|
| Naive Bayes | -- | -- | -- |
| L1 Logistic Regression | 0.0000 | 0.60 | 0.1577 |
| L2 Logistic Regression | 0.0000 | 0.63 | 0.1542 |
| Decision Tree | 0.0000 | 0.67 | 0.1503 |
| Random Forest | 0.0000 | 0.67 | 0.1503 |
| MLP | 0.0408 | 0.70 | 0.1463 |

> Using grade as the sole feature collapses recall to zero for all models except MLP, confirming that grade alone is not sufficient for default prediction. Tree-based models selecting unconstrained depth (max_depth=None) in this pass is a sign of overfitting to limited signal.

**Pass 3: CV, with grade, entire dataset (Random Forest)**

| Model | Recall | AUC | Brier Score |
|---|---|---|---|
| Random Forest | 0.1637 | 0.81 | 0.1470 |

> Pass 3 results match Pass 1 almost exactly, indicating the subset used in Pass 1 was already representative of the full dataset. Optimal parameters land on the edge of the search grid, suggesting the model could benefit from a wider hyperparameter search.

### Data Leakage Investigation

One of the more interesting findings in this project is what happens when you remove LendingClub-derived features (grade, sub_grade) from the feature set.

When we train a logistic regression on grade alone, it achieves an AUC of 0.65 -- nearly as good as the full model. This tells you something important: most of the predictive signal in the full model was coming from LendingClub's own risk assessment, not raw borrower characteristics. The practical implication is that if you want a model that genuinely adds information beyond what LendingClub already knows, you need to use raw borrower features only, or find alternative data sources.

### Temporal Stability Test

To check whether the model holds up over time, we trained on an earlier time period (2010--2014) and tested on a later one (2015--2017), simulating real-world deployment conditions. Random train/test splits are optimistic by design -- they mix loans from the same period into both sets, which is not how a deployed model would work.

| Metric | Pass 1 (random split) | Temporal split |
|---|---|---|
| Recall | 0.1637 | 0.2467 |
| AUC | 0.81 | 0.81 |
| Brier Score | 0.1470 | 0.1628 |
| Best max_depth | 30 | 10 |
| Best n_estimators | 150 | 100 |

AUC holds perfectly across the temporal split, and recall actually improves. The shift toward a shallower tree (max_depth=10 vs 30) reflects the model preferring less complexity when generalizing to future data, which is a sensible result. Overall, a strong stability outcome.

### Return Regression

Four regression models are trained to predict loan returns across all four return definitions:

| Return Column | Lasso R2 | Ridge R2 | MLP R2 | Random Forest R2 |
|---|---|---|---|---|
| ret_PESS | 0.0042 | 0.0124 | 0.0114 | 0.0113 |
| ret_OPT | -0.00002 | 0.0007 | -0.0112 | 0.0007 |
| ret_INTa | 0.0088 | 0.0138 | 0.0128 | 0.0124 |
| ret_INTb | 0.0092 | 0.0111 | 0.0104 | 0.0099 |

R2 scores are near zero across all models and return definitions. Ridge is the best overall and was selected as the regression model for the investment strategy backtesting. The weak regression results are not surprising -- predicting loan returns requires knowing things like whether a borrower will lose their job or repay early, which are not observable at loan issuance. The classification model is doing the real work.

### Investment Strategy Backtesting

Four strategies are tested on the held-out test set:

- **Random:** Invest in 1,000 randomly selected loans.
- **Default-based:** Rank loans by predicted default probability (ascending) and invest in the 1,000 with the lowest predicted default risk.
- **Return-based:** Rank loans by Ridge-predicted return (descending) and invest in the top 1,000.
- **Default-return-based:** Use the Random Forest to predict default probability, and two separate Ridge models (one for defaulted, one for non-defaulted loans) to predict returns. Compute a probability-weighted expected return per loan and invest in the top 1,000.

| Strategy | ret_PESS | ret_OPT | ret_INTa | ret_INTb |
|---|---|---|---|---|
| Random | -0.0065 | 0.0421 | 0.4066 | 1.2652 |
| Default-based | **0.0127** | 0.0377 | 0.4079 | 1.2509 |
| Return-based | 0.0127 | 0.0410 | 0.4196 | 1.2312 |
| Default-return-based | 0.0127 | 0.0368 | **0.4252** | 1.2596 |

Default-based is the most reliable strategy for risk-averse investors. It turns the pessimistic return from a loss (-0.0065) into a gain (0.0127) and performs consistently across all return definitions. The Default-return-based strategy edges ahead on ret_INTa, but the margin is small given the weak regression R2 scores. Random investing loses money under pessimistic return assumptions, which is the clearest argument for using a model at all.

### Portfolio Size Sensitivity

The Default-based strategy was tested across portfolio sizes ranging from 1,000 to 9,000 loans to understand how performance scales with portfolio size:

| Portfolio Size | Return (%) |
|---|---|
| 1,000 | 1.27 |
| 2,000 | 0.90 |
| 3,000 | 0.64 |
| 4,000 | 0.65 |
| 5,000 | 0.66 |
| 6,000 | 0.49 |
| 7,000 | 0.41 |
| 8,000 | 0.26 |
| 9,000 | 0.13 |

Returns decline as portfolio size grows, which is exactly what you would expect from a ranking-based strategy. The model's highest-confidence predictions are concentrated at the top of the ranking; as the portfolio expands, progressively lower-confidence loans are included, pulling down average returns. The sharpest drop is between 1,000 and 3,000 loans, suggesting the pool of genuinely high-confidence safe loans is relatively small. Importantly, returns remain positive at all portfolio sizes tested, confirming that the classifier adds genuine value even at scale. A portfolio of 1,000 to 2,000 loans represents the best trade-off between return quality and diversification under this strategy.

---

## Key Takeaways

**LendingClub's grade is hard to beat.** Once you strip out their derived features, model performance drops noticeably. The grade variable alone captures most of the predictive signal in the dataset, which makes sense -- LendingClub has access to data not in the public download and has been building underwriting models for years.

**Predicting default and maximizing returns are related but different problems.** The Default-based strategy performs best under pessimistic return assumptions. The Return-based strategy does better under optimistic ones. The best approach depends on your return model and risk tolerance.

**Temporal validity matters more than most analyses acknowledge.** Random train/test splits produce cleaner numbers but overestimate real-world performance. Temporal splits are messier but more honest about what a deployed model would actually do.

**Regression is hard.** All return regression models had near-zero R2 scores. Predicting the exact return of a loan is much harder than predicting whether it will default -- too many time-dependent factors affect the actual return amount. The classification model is doing the real work in the investment strategies.

---

## Technical Details

**Python version:** 3.9+

**Key dependencies:**
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
jupyter>=1.0.0
```

Install with:
```bash
pip install -r requirements.txt
```

**Runtime note:** The Random Forest model with full cross-validation takes approximately 10--15 minutes to run on a standard laptop. The MLP regression runs are the slowest individual cells at around 3--4 minutes per return column.

---

## What I Would Do Differently

The return regression models are the weakest part of this project. Survival analysis approaches like Cox proportional hazards are better suited to the temporal structure of loan default and would give a more principled way to model when default occurs, not just whether it does. Treating this as a standard regression problem loses information about default timing that matters a lot for return calculation.

The feature set is also limited to what LendingClub makes publicly available. Alternative data -- payment behavior on other accounts, employment verification, spending patterns -- would likely improve predictive power for the stripped model (the one without LendingClub-derived features) meaningfully.

Finally, the investment strategy backtesting assumes you can invest in any loan in the test set. In practice, loans fill up quickly and you face adverse selection -- the loans still available when you arrive are ones other investors passed on. Modeling this selection effect would make the strategy analysis considerably more realistic.

---

## Context

This project was completed as part of the Machine Learning for Problem Solving course at Carnegie Mellon University (95-828). The analytical framework, including the three-phase structure and return metric definitions, was developed collaboratively with my project team. The modeling implementation, data leakage investigation, and investment strategy backtesting are my own work.
