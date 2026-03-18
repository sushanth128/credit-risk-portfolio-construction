# Credit Risk & Portfolio Analytics
### Predicting Loan Default and Optimizing Investment Strategy on LendingClub Data

---

## Overview

This project started with a simple question: if you were an anonymous investor on a peer-to-peer lending platform, how would you decide which loans to fund?

The answer turns out to be more interesting than it looks. LendingClub assigns every loan a grade from A to G, and you could just follow their grades — Grade A loans default less, Grade G loans default more. But that raises a more interesting question: can you build something better than what LendingClub already gives you? And even if you can predict default well, does that actually translate to better investment returns?

This project works through both questions across three phases — from exploratory data analysis through predictive modeling to portfolio construction and backtesting.

---

## The Problem

Peer-to-peer lending is an unusual asset class. As a lender, you're essentially making an unsecured loan to an individual borrower. You earn the interest rate if they repay, and you lose principal if they default. The platform (LendingClub in this case) has already done some risk assessment and assigned a grade, but that grade is based on their proprietary model — not yours.

The core analytical challenge is that loans are temporal entities with 36 or 60 month terms. Default doesn't happen at a single point in time — some loans default immediately, some default years later, and some get repaid early before the term ends. Any model that ignores this temporal structure will produce optimistic results that don't hold in practice.

There's also a data leakage problem that isn't obvious at first glance. LendingClub's dataset includes features like `grade` and `sub_grade` that are derived from their own internal model — not raw borrower data. If you include those features in your model, you're essentially asking "can LendingClub's model predict default?" which is a very different question from "can you build a model that predicts default from raw borrower characteristics?" This distinction matters enormously for understanding where your model's predictive power actually comes from.

---

## Project Structure

```
credit-risk-portfolio-analytics/
├── README.md
├── requirements.txt
├── data/
│   └── README.md          ← instructions to download LendingClub data
└── notebooks/
    ├── 01_investor_framing_and_eda.ipynb
    ├── 02_feature_engineering_and_returns.ipynb
    └── 03_modeling_and_investment_strategies.ipynb
```

The three notebooks are designed to be read in order. Each one builds on the previous, and there are deliberate analytical decision points throughout where you have to make a choice and justify it.

---

## Data

The dataset comes from LendingClub's public loan data, covering loans issued from approximately 2010 to 2019. The full dataset contains around 2.5 million loan records with 150+ features per loan.

**To download the data:**
1. Go to [LendingClub's public data page](https://www.lendingclub.com/info/download-data.action) — note that LendingClub ceased operations as a P2P platform in 2020, but historical loan data remains publicly available through various sources including Kaggle
2. Download the loan-level data files (CSV format)
3. Place them in the `data/` directory
4. The notebooks expect a combined file called `lending_club_loans.csv` — see `data/README.md` for exact preprocessing steps to merge quarterly files

**Key features used:**

| Feature | Type | Description |
|---|---|---|
| `loan_amnt` | Numeric | Requested loan amount |
| `int_rate` | Numeric | Interest rate on the loan |
| `grade` / `sub_grade` | Categorical | LendingClub's risk assessment (derived feature — see Phase 3 discussion) |
| `annual_inc` | Numeric | Borrower's annual income |
| `dti` | Numeric | Debt-to-income ratio |
| `fico_range_low/high` | Numeric | Borrower's FICO credit score range |
| `delinq_2yrs` | Numeric | Delinquencies in past 2 years |
| `revol_util` | Numeric | Revolving credit utilization rate |
| `home_ownership` | Categorical | RENT / OWN / MORTGAGE |
| `emp_length` | Categorical | Employment length in years |
| `loan_status` | Target | Fully Paid / Charged Off / Default / etc. |

---

## Phase 1: Investor Framing and EDA

**Notebook:** `01_investor_framing_and_eda.ipynb`

The first notebook is about understanding the data before touching any models. The framing matters here — we're approaching this as an investor trying to maximize risk-adjusted returns, not as a data scientist trying to minimize classification error. That distinction shapes every decision that follows.

**Key findings from EDA:**

The grade distribution tells the first important story. Grade B and C loans make up the majority of the portfolio (~28% and ~28% respectively), with Grade A representing about 17% and the riskier grades (E, F, G) representing a small tail. This isn't surprising — LendingClub's origination model naturally filters out the most extreme risks.

Default rates follow the expected gradient:

| Grade | Default Rate | Avg. Interest Rate |
|---|---|---|
| A | 9.1% | 7.2% |
| B | 17.6% | 10.9% |
| C | 28.1% | 14.2% |
| D | 36.6% | 18.1% |
| E | 43.6% | 21.3% |
| F | 49.3% | 24.9% |
| G | 54.6% | 27.5% |

The interesting question isn't whether higher grades default less — of course they do. It's whether the higher interest rates on riskier loans compensate for the additional default risk. The answer depends heavily on how you define "return."

We define four return metrics that capture different investor assumptions:
- **M1 (Pessimistic):** No reinvestment of recovered principal — investor simply loses defaulted principal
- **M2 (Optimistic):** Immediate reinvestment at the risk-free rate on recovered amounts
- **M3 (Interest-based):** Return calculated as a function of the interest rate and default timing

Under M1, Grades C through G actually produce negative average returns — the default losses exceed the interest income. Under M3 with a 2% reinvestment rate, Grade G loans look attractive. This isn't a contradiction — it reflects genuine uncertainty about what the right return model is, and it drives the investment strategy design in Phase 3.

**On feature correlations:** Several features are highly correlated in ways that create downstream issues. `int_rate` and `grade` are strongly correlated because LendingClub sets interest rates based on their internal grade model. `fico_range_high` and `fico_range_low` are by definition almost perfectly correlated. `loan_amnt` and `installment` move together. These correlations inform the feature selection decisions in Phase 2.

---

## Phase 2: Feature Engineering and Return Calculation

**Notebook:** `02_feature_engineering_and_returns.ipynb`

Phase 2 does the preparatory work for modeling: cleaning the data, engineering features, handling the temporal structure, and computing the return variables that serve as regression targets in Phase 3.

**Feature engineering decisions:**

The credit history length feature (`cr_hist`) deserves specific mention. Rather than using `earliest_cr_line` directly, we compute the length of the borrower's credit history at the time the loan was issued — the difference between `issue_d` and `earliest_cr_line` in months. This is a meaningful distinction: a 1980 credit line looks very different on a 2010 loan application versus a 2015 one.

Percentage fields (`int_rate`, `revol_util`) require string cleaning before numeric conversion. Date fields need careful handling — it's important to use the loan issuance date as the reference point, not the data download date, to avoid introducing look-ahead bias.

**On outlier handling:** We take a domain-informed approach. For `annual_inc`, `dti`, `revol_bal`, and `open_acc`, outliers are flagged and removed because they likely represent data entry errors or edge cases that won't generalize. But for `delinq_2yrs`, `pub_rec`, and the recovery fields, we retain outliers — a borrower with 20 delinquencies in the past two years is a genuine data point worth keeping, not noise.

**The temporal structure issue:** Loans in this dataset span from 2010 to 2019, and credit conditions changed substantially over that period. A model trained on 2010 loans is being asked to generalize to borrowers in a very different macroeconomic environment in 2017. This is addressed directly in Phase 3's temporal stability analysis, but it's worth flagging early: random train/test splits will overestimate model performance in a way that temporal splits will not.

---

## Phase 3: Modeling and Investment Strategy Backtesting

**Notebook:** `03_modeling_and_investment_strategies.ipynb`

This is the core of the project. It covers default prediction, the data leakage investigation, temporal stability analysis, and investment strategy backtesting.

### Default Prediction

We train and evaluate six classifiers on the task of predicting whether a loan will default:

| Model | AUC (with LC features) | AUC (without LC features) |
|---|---|---|
| Random Classifier (baseline) | 0.50 | 0.50 |
| Naïve Bayes | 0.67 | 0.65 |
| L1 Logistic Regression | 0.70 | 0.61 |
| L2 Logistic Regression | 0.70 | 0.62 |
| Decision Tree | 0.66 | 0.60 |
| Random Forest | 0.70 | **0.67** |

All models are evaluated using AUC-ROC rather than accuracy. The class imbalance (about 19% default rate in our sample) makes accuracy a misleading metric — a model that predicts "no default" for every loan would achieve ~80% accuracy while being completely useless.

### The Data Leakage Investigation

One of the more interesting findings in this project is what happens when you remove LendingClub-derived features (`grade`, `sub_grade`, `verification_status`, `dti`, `loan_status`) from the feature set.

When we train a logistic regression on `grade` alone, it achieves an AUC of 0.65 — nearly as good as our full model. This tells us something important: most of the predictive signal in the full model was coming from LendingClub's own risk assessment, not from raw borrower characteristics. Once we remove those features, model performance drops significantly and the ranking changes — Random Forest becomes the clear winner at AUC 0.67.

This has a practical implication: if you want to build a model that genuinely adds information beyond what LendingClub already knows, you need to either use raw borrower characteristics only, or find alternative data sources that LendingClub isn't already using.

All results reported below use the model without LendingClub-derived features.

### Temporal Stability

A model trained on 2009 loan data and tested on 2017 loans achieves 78.9% accuracy and AUC 0.60. The same architecture trained on 2016 data achieves 84.6% accuracy and AUC 0.65 on the same 2017 test set. The model degrades meaningfully over time.

This is expected — credit conditions, macroeconomic environment, and borrower behavior all shift over time. In a real deployment, this model would require periodic retraining on recent data. The 2009-trained model isn't useless (it still beats random), but it's materially worse than a recently-trained version.

### Investment Strategy Backtesting

We test four investment strategies using 100 independent train/test splits to average out variance:

**Random:** Invest in 1,000 randomly selected loans from the test set.

**Default-based:** Rank loans by predicted default probability (ascending) and invest in the 1,000 with the lowest predicted default risk.

**Return-based:** Train a regression model to predict expected return for each loan and invest in the top 1,000 by predicted return.

**Default-return-based (DefRet):** Train two separate regression models — one for defaulted loans, one for non-defaulted loans. Compute expected return as a probability-weighted combination and invest in the top 1,000.

Results averaged across 100 runs:

| Strategy | M1 Return | M2 Return | M3 (1.4%) Return | M3 (2%) Return |
|---|---|---|---|---|
| Random | -0.0042 | 0.0430 | 0.4189 | 1.2661 |
| Default-based | **0.0133** | **0.0454** | 0.4193 | 1.2505 |
| Return-based | -0.0089 | 0.0416 | 0.4195 | **1.2710** |
| DefRet | 0.0089 | 0.0403 | **0.4213** | 1.2510 |
| Best (hindsight) | 0.0133 | 0.0454 | 0.4213 | 1.2710 |

**Default-based** is the most reliable strategy — it outperforms random under M1 and M2, and matches the theoretical best under M1. The DefRet strategy performs best under M3 (1.4%) and is consistently solid across all return definitions. Random investing loses money under the pessimistic (M1) return assumption.

### Portfolio Size Sensitivity

Returns peak at a portfolio size of 4,000–5,000 loans and decline as you invest in more. This is intuitive: the model ranks the best loans first, and as you expand the portfolio you're gradually including lower-quality loans that the model ranked as less attractive. At around 6,000+ loans, the incremental loans start dragging down average portfolio returns.

---

## Key Takeaways

A few things stood out from working through this project:

**LendingClub's grade is hard to beat.** Once you strip out their derived features, your model's predictive power drops noticeably. The grade variable alone captures most of the signal in the dataset. This makes sense — LendingClub has access to more data than what's in the public download, and their underwriting model is built on years of loan performance history.

**Predicting default and maximizing returns are related but different problems.** The Default-based strategy (which minimizes default risk) performs well under pessimistic return assumptions, but the Return-based strategy does better under optimistic ones. The best approach depends on your return model and risk appetite.

**Temporal validity matters more than most analyses acknowledge.** It's tempting to use random train/test splits because they produce cleaner results. Temporal splits are messier and produce lower performance numbers — but they're more honest about what would actually happen if you deployed this model.

**Regression is hard.** All regression models for return prediction had near-zero R² scores. Predicting the exact return of a loan is much harder than predicting whether it will default — too many time-dependent factors affect the actual return amount. The classification model (default vs. no default) is doing the real work.

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

**Runtime note:** The 100-iteration cross-validation loops in Phase 3 are computationally intensive. The Random Forest model with full cross-validation takes approximately 3–4 hours to run on a standard laptop. Reduce the seed range (e.g., `range(0, 10)` instead of `range(0, 101)`) for a faster exploratory run.

---

## What I'd Do Differently

A few things I'd change with more time:

The return regression models are weak. I'd explore survival analysis approaches (like Cox proportional hazards) that are better suited to the temporal structure of loan default. Treating this as a classification problem loses information about *when* default occurs, which matters a lot for return calculation.

The feature set is limited to what LendingClub makes publicly available. Alternative data — payment behavior on other accounts, spending patterns, employment verification — would likely improve predictive power meaningfully for the stripped model.

The investment strategy backtesting assumes you can invest in any loan in the test set at the modeled return. In practice, loans fill up quickly and you'd face adverse selection — the loans still available when you arrive are the ones other investors passed on. Modeling this selection effect would make the strategy analysis more realistic.

---

## Context

This project was completed as part of the Machine Learning for Problem Solving course at Carnegie Mellon University (95-828). The analytical framework — particularly the three-phase structure and the return metric definitions — was developed collaboratively with my project team. The modeling implementation, data leakage investigation, and investment strategy backtesting are my own work.
