# Company Bankruptcy Prediction

Machine learning classifier that flags companies at risk of bankruptcy from their financial ratios. Built as part of a data science On Job Training program, using a public dataset.

**Shubham Adate** | [LinkedIn](https://www.linkedin.com/in/adate-shubham) | [GitHub](https://github.com/shubhamadatework-git)

## Problem
A financial institution wants an early-warning signal for company bankruptcies to support risk assessment and lending decisions.

## Data
- 6,819 companies, 95 financial indicators (profitability, liquidity, leverage, efficiency) and a `Bankrupt?` label
- Highly imbalanced: only about 3% of companies are bankrupt (44 of the 1,364 companies in the test set)
- Source: [Taiwanese Bankruptcy Prediction (UCI)](https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction). See `data/README.md`.

## Approach
1. Exploratory analysis and data checks; outliers were kept on purpose because they reflect real financial volatility
2. Feature selection using correlation with the target
3. 60/20/20 train / validation / test split
4. Compared 7 algorithms: Logistic Regression, KNN, SVM, Decision Tree, Random Forest, Gradient Boosting, XGBoost
5. Evaluated with accuracy, cross-validation and class-level precision, recall and F1

## Results
| Model | Accuracy |
|---|---|
| **Random Forest** | **96.99%** (cross-validation 97.14%) |
| KNN / XGBoost | about 96.7 to 96.9% |
| SVM / Gradient Boosting / Decision Tree | slightly below 96.5% |
| Logistic Regression | 95.96% |

**Read this with the class imbalance in mind.** Because about 97% of companies are not bankrupt, accuracy is mostly driven by the majority class. On the held-out test set the Random Forest's results for the **bankrupt class** were:

| Metric (bankrupt class) | Value |
|---|---|
| Precision | 0.56 |
| Recall | 0.32 |
| F1 | 0.41 |
| Confusion matrix | 14 bankrupt companies caught, 30 missed, 11 false alarms |

Strongest predictors: Return on Assets (ROA), operating profit rate, net income to total assets, debt ratio and current liability to assets, and cash flow to total assets.

## Limitations and next steps
- Recall on bankrupt companies is low; the model misses more than half of them
- Next steps: threshold tuning, precision-recall AUC, class weights or resampling applied inside cross-validation, and testing on data from other markets or periods

## Run it
```bash
pip install -r requirements.txt
jupyter notebook company-bankruptcy-prediction.ipynb
```

**Tech:** Python, pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn, seaborn, matplotlib
