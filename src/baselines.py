"""
baselines.py - TF-IDF + Logistic Regression / Linear SVM (and optional Word2Vec).

Run from the repo root:
    python -m src.baselines                 # TF-IDF + LR + SVM
    python -m src.baselines --w2v           # also Word2Vec-average + LR

Method notes (for the write-up):
  * Features are fitted on the TRAIN split only.
  * Hyperparameter C is chosen on the VALIDATION split (a time-ordered split
    must not be tuned with shuffled cross-validation).
  * The TEST split is touched once per model, after the choice is made.
  * Class weights are "balanced" because fraud/churn are minority classes.
"""
import argparse
import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.evaluate import (LABEL_NAMES, bootstrap_macro_f1_ci, compute_metrics,
                          plot_confusion)

SEED = 42
GRIDS = {"tfidf_logreg": [1, 10, 100],
         "tfidf_svm": [0.1, 0.5, 1.0],
         "w2v_logreg": [0.1, 1, 10]}


def load_splits(data_dir):
    return {s: pd.read_parquet(Path(data_dir) / f"{s}.parquet")
            for s in ("train", "val", "test")}


def make_model(name, C):
    if name.endswith("logreg"):
        return LogisticRegression(C=C, max_iter=1000, class_weight="balanced",
                                  random_state=SEED)
    return LinearSVC(C=C, class_weight="balanced", random_state=SEED)


def get_scores(model, X):
    """(n, 3) scores: probabilities for LR, decision values for SVM."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return model.decision_function(X)


def run_model(name, Xtr, ytr, Xva, yva, Xte, yte):
    """Pick C on validation macro-F1, then score once on test."""
    best = None
    for C in GRIDS[name]:
        t0 = time.time()
        m = make_model(name, C).fit(Xtr, ytr)
        val_f1 = compute_metrics(yva, m.predict(Xva))["macro_f1"]
        print(f"  [{name}] C={C:<6} val macro-F1={val_f1:.4f}  ({time.time()-t0:.0f}s)")
        if best is None or val_f1 > best[0]:
            best = (val_f1, C, m)
    val_f1, C, model = best
    pred = model.predict(Xte)
    scores = get_scores(model, Xte)
    metrics = compute_metrics(yte, pred, scores)
    lo, hi = bootstrap_macro_f1_ci(yte, pred)
    row = {"model": name, "best_C": C, "val_macro_f1": val_f1, **metrics,
           "macro_f1_ci_low": lo, "macro_f1_ci_high": hi}
    return row, model, pred


def tokenize(text, drop_stop):
    toks = re.findall(r"[a-z']+", text.lower())
    return [t for t in toks if not (drop_stop and t in ENGLISH_STOP_WORDS)]


def doc_vectors(token_lists, wv):
    dim = wv.vector_size
    out = np.zeros((len(token_lists), dim), dtype=np.float32)
    for i, toks in enumerate(token_lists):
        vecs = [wv[t] for t in toks if t in wv]
        if vecs:
            out[i] = np.mean(vecs, axis=0)
    return out


def main(args):
    res = Path(args.results_dir)
    (res / "models").mkdir(parents=True, exist_ok=True)
    (res / "figures").mkdir(parents=True, exist_ok=True)
    (res / "predictions").mkdir(parents=True, exist_ok=True)

    d = load_splits(args.data_dir)
    ytr, yva, yte = (d[s]["label_id"].to_numpy() for s in ("train", "val", "test"))
    print({s: len(v) for s, v in d.items()})

    rows = []

    # ---------------- TF-IDF features (fit on train only) ----------------
    vec = TfidfVectorizer(
        lowercase=True,
        stop_words=None if args.keep_stopwords else "english",
        ngram_range=(1, 2), min_df=5, max_df=0.9,
        max_features=args.max_features, sublinear_tf=True)
    t0 = time.time()
    Xtr = vec.fit_transform(d["train"]["text"])
    Xva, Xte = vec.transform(d["val"]["text"]), vec.transform(d["test"]["text"])
    print(f"TF-IDF: {Xtr.shape[1]:,} features ({time.time()-t0:.0f}s)")
    joblib.dump(vec, res / "models" / "tfidf_vectorizer.joblib")

    fitted = {}
    for name in ("tfidf_logreg", "tfidf_svm"):
        row, model, pred = run_model(name, Xtr, ytr, Xva, yva, Xte, yte)
        rows.append(row)
        fitted[name] = model
        joblib.dump(model, res / "models" / f"{name}.joblib")
        plot_confusion(yte, pred, res / "figures" / f"cm_{name}.png", name)
        pd.DataFrame({"Complaint ID": d["test"]["Complaint ID"].to_numpy(),
                      "y_true": yte, "y_pred": pred}
                     ).to_parquet(res / "predictions" / f"{name}_test.parquet", index=False)

    # Top terms per class from Logistic Regression coefficients. These are the
    # "global explanation" to compare with SHAP later.
    lr, terms = fitted["tfidf_logreg"], np.array(vec.get_feature_names_out())
    top = []
    for k, cls in enumerate(LABEL_NAMES):
        for j in np.argsort(lr.coef_[k])[::-1][:25]:
            top.append({"class": cls, "term": terms[j], "coef": lr.coef_[k][j]})
    pd.DataFrame(top).to_csv(res / "lr_top_terms.csv", index=False)

    # ---------------- Optional: Word2Vec average + LR ----------------
    if args.w2v:
        try:
            from gensim.models import Word2Vec
        except ImportError:
            raise SystemExit("gensim is not installed: pip install gensim")
        drop = not args.keep_stopwords
        tok = {s: [tokenize(t, drop) for t in d[s]["text"]] for s in d}
        t0 = time.time()
        # workers=1 keeps training reproducible for a fixed seed
        w2v = Word2Vec(tok["train"], vector_size=200, window=5, min_count=5,
                       workers=1, epochs=5, seed=SEED)
        print(f"Word2Vec trained on train split only ({time.time()-t0:.0f}s)")
        Wtr, Wva, Wte = (doc_vectors(tok[s], w2v.wv) for s in ("train", "val", "test"))
        row, model, pred = run_model("w2v_logreg", Wtr, ytr, Wva, yva, Wte, yte)
        rows.append(row)
        joblib.dump(model, res / "models" / "w2v_logreg.joblib")
        w2v.save(str(res / "models" / "word2vec.model"))
        plot_confusion(yte, pred, res / "figures" / "cm_w2v_logreg.png", "w2v_logreg")
        pd.DataFrame({"Complaint ID": d["test"]["Complaint ID"].to_numpy(),
                      "y_true": yte, "y_pred": pred}
                     ).to_parquet(res / "predictions" / "w2v_logreg_test.parquet", index=False)

    out = pd.DataFrame(rows)
    out.to_csv(res / "baseline_results.csv", index=False)
    with open(res / "baseline_config.json", "w") as f:
        json.dump({"seed": SEED, "grids": GRIDS, "max_features": args.max_features,
                   "keep_stopwords": args.keep_stopwords}, f, indent=2)
    show = ["model", "best_C", "val_macro_f1", "accuracy", "macro_f1",
            "macro_f1_ci_low", "macro_f1_ci_high", "pr_auc_macro",
            "f1_fraud", "f1_churn"]
    print("\nTEST RESULTS")
    print(out[show].round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--max-features", type=int, default=200_000)
    ap.add_argument("--keep-stopwords", action="store_true")
    ap.add_argument("--w2v", action="store_true")
    main(ap.parse_args())
