# FIFA World Cup 2026 Predictor

A machine learning project that predicts which teams will reach the quarter-finals and beyond at the 2026 FIFA World Cup, based on historical national team performance data from 2002-2022. Done purely for learning and educational purposes.

## Dataset

Uses the [FIFA World Cup Team Dataset](https://www.kaggle.com/datasets/harrachimustapha/fifa-world-cup-team-dataset) from Kaggle, which includes team-level stats such as FIFA ranking/points, squad market value, squad age, recent win/loss record, and historical World Cup progression (round of 16, quarter-finals, semi-finals, finals, titles).

Download the dataset and place the `fifa-world-cup-team-dataset` folder in the project root before running the notebook.

## What the notebook does

1. **EDA** - explores the train (2002-2022) and test (2026) sets: missing values, progression rates by continent, correlation between features, FIFA rank vs. how far teams advance, and distribution shifts between train and test (e.g. the 2018 FIFA ranking system change, 2026's three host nations).
2. **Feature engineering** - imputes missing squad market values, derives goal difference and win rate over the last 4 years, one-hot encodes continent.
3. **Modeling** - frames the problem as four separate binary classification tasks (`quarter_finalist`, `semi_finalist`, `finalist`, `winner`), each solved with the best of Logistic Regression, Random Forest, XGBoost, and LightGBM, selected via chronological validation (training on earlier editions, validating on 2010/2014/2018/2022) and Top-K accuracy.
4. **2026 predictions** - retrains the chosen models on all available data (2002-2022) and produces a predicted bracket and full ranking table for all 48 teams in the 2026 tournament, along with validation metrics and feature importances per stage.

## Requirements

The notebook installs its own dependencies on first run: `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `lightgbm`.
