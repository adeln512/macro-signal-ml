import pandas as pd # type: ignore
import numpy as np # type: ignore
from eval.report import print_report, save_confusion_matrix, save_roc_pr_curves, summarize_coefficients

from features.make_dataset import make_dataset
macro_df = pd.read_csv(
    "data/data/processed/macro_monthly.csv",
    parse_dates=["observation_date"]
)
macro_df = macro_df.set_index("observation_date").sort_index()

X, y, dates, full_df = make_dataset(macro_df)

train_end = "2014-12-01"
X_train = X.loc[dates <= train_end]
y_train = y.loc[dates <= train_end]

X_test = X.loc[dates > train_end]
y_test = y.loc[dates > train_end]
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

from sklearn.preprocessing import StandardScaler # type: ignore

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression # type: ignore

model = LogisticRegression(penalty = "l2", C = 1.0, solver = "lbfgs", max_iter = 1000)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]
print_report(y_test, y_pred, y_prob=y_prob, header="Logistic Regression (Test)")
save_confusion_matrix(y_test, y_pred, "eval/figures/logreg_cm.png")
save_roc_pr_curves(y_test, y_prob, "eval/figures", prefix="logreg")


from sklearn.metrics import accuracy_score, classification_report # type: ignore

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

coef_summary = summarize_coefficients(model.coef_[0], list(X.columns), top_k=5)
print("\nMost negative coefficients:")
print(pd.Series(coef_summary["most_negative"]))
print("\nMost positive coefficients:")
print(pd.Series(coef_summary["most_positive"]))
