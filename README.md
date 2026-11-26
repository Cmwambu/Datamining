# Predicting the Perception of University Leaders from Social Media Information
**Case study: University of Eastern Africa, Baraton (UEAB) | Course: Data Mining**

Group members: Ian Bett (SBETIA2411), Crystie Topaz (SOCHCR2411), Cassimy Mwambu (SMWAMA2412)

## Overview
A supervised classification model that predicts whether a UEAB student or staff member holds a
**positive, neutral or negative** perception of campus leaders, based on social media use and
information habits. It also ranks the factors that most influence that perception.
The work follows CRISP-DM. Algorithms compared: Logistic Regression (baseline), Decision Tree,
Random Forest and Support Vector Machine.

## Repository structure
```
train_model.py          # cleaning, modelling, evaluation, feature importance
generate_demo_data.py   # SYNTHETIC data, only for testing that the code runs
requirements.txt
data/                   # put survey_responses.csv here (not committed: privacy)
results/                # generated: metrics table, confusion matrix, plots, saved model
```

## Data
Primary data from a structured questionnaire (~370 respondents, stratified sample, Yamane's formula
at 5% margin of error). Save it as `data/survey_responses.csv`, one row per respondent, with columns:

| Group | Columns |
|---|---|
| Demographics | `gender`, `age_group`, `role`, `year_or_position` |
| Social media use | `uses_whatsapp`, `uses_facebook`, `uses_x`, `uses_instagram`, `uses_tiktok`, `hours_per_day`, `purpose_of_use` |
| Information habits | `main_info_source`, `trust_in_source`, `misinfo_exposure`, `checks_official_channels`, `follows_leaders_online`, `engages_with_posts` |
| Target items (Likert 1-5) | `perception_trust`, `perception_credibility`, `perception_competence`, `perception_approachability` |

The target class is derived from the mean of the four Likert items (below 2.5 negative, 2.5 to 3.5
neutral, above 3.5 positive). These four items are removed from the features to prevent leakage.
Adjust column names at the top of `train_model.py` if your questionnaire differs.
Participation was voluntary and anonymous, with no personal identifiers stored (Kenya Data Protection Act, 2019).

## Pipeline
1. Remove duplicate and incomplete records; median/mode imputation for missing feature values
2. IQR and box-plot outlier check
3. One-hot encoding of categorical features, scaling of numeric features
4. Stratified 80/20 train/test split
5. SMOTE on training data only (applied automatically if classes are imbalanced)
6. 10-fold cross-validated grid search per model
7. Evaluation on the test set: accuracy, precision, recall, F1, ROC-AUC, confusion matrix
8. Permutation feature importance on the best model

## How to run
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# put your real data in data/survey_responses.csv, then:
python train_model.py
```
To only test the code, run `python generate_demo_data.py` first. Its output is synthetic and
its metrics must not be reported.

## Results
See `results/model_comparison.csv`, `results/confusion_matrix.png` and `results/feature_importance.png`.
(Add your final scores here after running on the real data.)

## Reference
Vosoughi, S., Roy, D., & Aral, S. (2018). The spread of true and false news online. *Science, 359*(6380), 1146-1151.
