"""
evaluate.py - shared metrics, bootstrap confidence intervals and plots.
Used by the baselines now and by the transformer / SHAP steps later, so every
model is scored in exactly the same way.
"""
import numpy as np
from matplotlib.figure import Figure   # no pyplot: no global backend side effects
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.preprocessing import label_binarize

LABEL_NAMES = ["other", "fraud", "churn"]   # index = label_id from data.py
LABEL_IDS = [0, 1, 2]


def compute_metrics(y_true, y_pred, scores=None) -> dict:
    """Accuracy, macro P/R/F1, per-class P/R/F1 and (if scores given) macro PR-AUC.

    scores: array of shape (n_samples, 3), class probabilities or decision values.
    """
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    out = {"accuracy": accuracy_score(y_true, y_pred),
           "macro_precision": p, "macro_recall": r, "macro_f1": f}
    pc, rc, fc, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABEL_IDS, zero_division=0)
    for i, name in enumerate(LABEL_NAMES):
        out[f"precision_{name}"] = pc[i]
        out[f"recall_{name}"] = rc[i]
        out[f"f1_{name}"] = fc[i]
    if scores is not None:
        Y = label_binarize(y_true, classes=LABEL_IDS)
        out["pr_auc_macro"] = average_precision_score(Y, scores, average="macro")
    return out


def bootstrap_macro_f1_ci(y_true, y_pred, n_boot=1000, seed=42, alpha=0.05):
    """95% bootstrap confidence interval for macro-F1 on the test set."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
    return (float(np.percentile(stats, 100 * alpha / 2)),
            float(np.percentile(stats, 100 * (1 - alpha / 2))))


def plot_confusion(y_true, y_pred, path, title=""):
    """Row-normalised confusion matrix (each row sums to 1 = recall per class)."""
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_IDS, normalize="true")
    fig = Figure(figsize=(4.5, 4))
    ax = fig.subplots()
    im = ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(3), LABEL_NAMES)
    ax.set_yticks(range(3), LABEL_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


# ---------------------------------------------------------------------------
# Comparing two models on the SAME test set (paired tests)
# ---------------------------------------------------------------------------
def mcnemar_exact(y_true, pred_a, pred_b) -> dict:
    """Exact McNemar test on correct/incorrect decisions of two models.

    Only the cases where exactly one model is right matter. Under the null
    hypothesis (equal error rates) each such case is a coin flip, so the test
    is a two-sided binomial test. Note: this compares plain correctness
    (accuracy-style), not macro-F1; use paired_bootstrap_f1_diff for macro-F1.
    """
    from scipy.stats import binomtest
    y_true = np.asarray(y_true)
    a_ok = np.asarray(pred_a) == y_true
    b_ok = np.asarray(pred_b) == y_true
    only_a = int((a_ok & ~b_ok).sum())
    only_b = int((~a_ok & b_ok).sum())
    n = only_a + only_b
    p = 1.0 if n == 0 else float(binomtest(min(only_a, only_b), n, 0.5).pvalue)
    return {"only_a_correct": only_a, "only_b_correct": only_b, "p_value": p}


def paired_bootstrap_f1_diff(y_true, pred_a, pred_b, n_boot=1000, seed=42):
    """Bootstrap 95% CI for macro-F1(B) - macro-F1(A) on one shared test set.

    Resamples test rows (same rows for both models), so the model difference is
    measured on paired data. If the CI excludes 0, B and A differ reliably.
    """
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        fa = f1_score(y_true[idx], pred_a[idx], average="macro", zero_division=0)
        fb = f1_score(y_true[idx], pred_b[idx], average="macro", zero_division=0)
        diffs.append(fb - fa)
    point = (f1_score(y_true, pred_b, average="macro", zero_division=0)
             - f1_score(y_true, pred_a, average="macro", zero_division=0))
    return {"diff": float(point),
            "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5))}
