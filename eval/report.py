from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np # type: ignore
import pandas as pd # type: ignore

from sklearn.metrics import ( # type: ignore
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

import matplotlib.pyplot as plt # type: ignore


@dataclass
class EvalResults:
    accuracy: float
    precision: float
    recall: float
    f1: float
    support_0: int
    support_1: int
    roc_auc: Optional[float] = None
    avg_precision: Optional[float] = None


def evaluate_classification(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    pos_label: int = 1,
) -> EvalResults:
    """
    Compute core classification metrics.
    y_prob should be the predicted probability for class=1 (same length as y_true).
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=pos_label, zero_division=0
    )

    support_0 = int((y_true == 0).sum())
    support_1 = int((y_true == 1).sum())

    roc_auc = None
    avg_prec = None
    if y_prob is not None:
        
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            roc_auc = None
        try:
            avg_prec = float(average_precision_score(y_true, y_prob))
        except Exception:
            avg_prec = None

    return EvalResults(
        accuracy=acc,
        precision=float(prec),
        recall=float(rec),
        f1=float(f1),
        support_0=support_0,
        support_1=support_1,
        roc_auc=roc_auc,
        avg_precision=avg_prec,
    )


def print_report(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    header: str = "Evaluation",
) -> None:
    """Pretty console output."""
    print(f"\n=== {header} ===")
    print(f"n={len(y_true)} | class0={int((y_true==0).sum())} | class1={int((y_true==1).sum())}")
    if y_prob is not None:
        print("Probabilities provided: yes")
    else:
        print("Probabilities provided: no")

    results = evaluate_classification(y_true, y_pred, y_prob=y_prob)
    print(f"Accuracy: {results.accuracy:.4f}")
    print(f"Precision (class=1): {results.precision:.4f}")
    print(f"Recall    (class=1): {results.recall:.4f}")
    print(f"F1        (class=1): {results.f1:.4f}")
    if results.roc_auc is not None:
        print(f"ROC AUC:   {results.roc_auc:.4f}")
    if results.avg_precision is not None:
        print(f"PR AUC:    {results.avg_precision:.4f}")

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, zero_division=0))


def save_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    out_path: str | Path,
    title: str = "Confusion Matrix",
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])

    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, str(val), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def save_roc_pr_curves(
    y_true: pd.Series,
    y_prob: np.ndarray,
    out_dir: str | Path,
    prefix: str = "logreg",
) -> None:
    """
    Saves ROC and Precision-Recall curves. Requires probabilities for class=1.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_roc.png", dpi=200)
    plt.close(fig)

    
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    fig = plt.figure()
    plt.plot(rec, prec)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_pr.png", dpi=200)
    plt.close(fig)


def summarize_coefficients(
    coef: np.ndarray,
    feature_names: list[str],
    top_k: int = 8,
) -> Dict[str, Any]:
    """
    Returns top positive/negative coefficients for interpretability.
    coef should be shape (n_features,).
    """
    s = pd.Series(coef, index=feature_names).sort_values()
    return {
        "most_negative": s.head(top_k).to_dict(),
        "most_positive": s.tail(top_k).to_dict(),
    }
