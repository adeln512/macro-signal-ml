# Macroeconomic Signal Prediction with Machine Learning

This project studies whether commonly used macroeconomic indicators contain predictive information about future changes in core inflation.


## Overview

Using monthly U.S. macroeconomic data (labor market conditions, monetary policy, industrial output, and yield curve information), I construct a supervised learning dataset where the task is to predict whether 6-month annualized core inflation will be higher or lower six months into the future.

The goal of the project is not to maximize short-term predictive accuracy, but to:
- build a leak-free, reproducible macro ML pipeline
- evaluate whether simple models capture economically meaningful signals
- interpret learned relationships in light of macroeconomic theory


##  Data

All data are publicly available monthly time series from the Federal Reserve Economic Data (FRED) database.

Indicators used:
- Core CPI (excluding food & energy)
- Unemployment rate
- Effective federal funds rate
- 2-year Treasury yield
- 10-year Treasury yield
- Industrial production index

Daily Treasury yields were aggregated to monthly averages.
All series were aligned to a common monthly date index and merged into a single dataset spanning 1976–2025.


## Feature Engineering & Labels

Core inflation is defined as 6-month annualized inflation, computed from the core CPI index:
- inflation_6m(t)  = 200 * ((cpi / cpi_lag6) - 1)

The prediction label is forward-looking:
- 1 if inflation at (t+6) is higher than inflation at t
- 0 otherwise

To model delayed macroeconomic effects, each predictor is transformed into lagged features at 1, 3, 6, and 12 months.
An additional yield curve spread feature (10Y − 2Y) is constructed and lagged as well.

All feature construction is performed using only information available at time t, ensuring no look-ahead bias.


## Modeling Approach

I begin with logistic regression, chosen for its interpretability and suitability for economic analysis.

Key modeling choices:
- Time-based train/test split (training on earlier decades, testing on later periods)
- Feature standardization using training data only
- L2-regularized logistic regression to control overfitting

Rather than treating the model as a black box, coefficient signs and magnitudes are analyzed to assess whether learned relationships align with known macroeconomic mechanisms.


## Results & Interpretations

Predictive accuracy on the test set is modest (≈48%), which is expected given the difficulty of macroeconomic forecasting and the absence of high-frequency or expectation-based data.

More importantly, the learned coefficients exhibit economically meaningful patterns, including:
- yield curve inversion predicting future disinflation
- labor market slack reducing inflation pressure
- delayed effects of monetary tightening
- policy endogeneity reflected in longer-lag interest rate coefficients

These results suggest that even simple linear models can recover recognizable macroeconomic structure when constructed carefully.


## Limitations

- Macroeconomic forecasting is inherently noisy; accuracy alone is not a sufficient evaluation metric.
- The model does not incorporate expectations data, financial market volatility, or global variables.
- Logistic regression cannot capture nonlinear interactions between macro variables.

These results suggest that even simple linear models can recover recognizable macroeconomic structure when constructed carefully and evaluated honestly.


## How to Run
From the project root:
- pip install -r requirements.txt
- python -m models.train

This will train the logistic regression model and print evaluation metrics and coefficient summaries.


## Project Structure

macro-signal-ml/
├── data/
│   ├── raw/               # FRED CSV files
│   └── processed/         # merged monthly dataset
├── features/              # feature & label construction
├── models/                # model training scripts
├── eval/                  # evaluation utilities
├── notebooks/             # exploratory analysis
├── README.md
├── requirements.txt


## Future Work

- Compare linear models with nonlinear approaches (e.g., random forests)
- Incorporate expectations-based indicators
- Evaluate regime-specific performance (inflationary vs disinflationary periods)
- Extend to multi-class or regression formulations