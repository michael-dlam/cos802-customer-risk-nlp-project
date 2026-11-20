"""
data.py - load CFPB complaints, define risk labels, clean text, split.

Usage (from repo root):
    python -m src.data --explore  data/raw/complaints.csv    # step 1: inspect Issue values
    python -m src.data --build    data/raw/complaints.csv    # step 2: build labelled splits
"""
import argparse
import re
from pathlib import Path

import pandas as pd

USECOLS = [
    "Date received", "Product", "Sub-product", "Issue", "Sub-issue",
    "Consumer complaint narrative", "Company response to consumer",
    "Timely response?", "Complaint ID",
]

# Column names in the archived parquet mirror -> names used in this module
PARQUET_RENAME = {
    "date_received": "Date received",
    "product": "Product",
    "sub_product": "Sub-product",
    "issue": "Issue",
    "sub_issue": "Sub-issue",
    "narrative": "Consumer complaint narrative",
    "company_response": "Company response to consumer",
    "timely_response": "Timely response?",
    "complaint_id": "Complaint ID",
}

# ---------------------------------------------------------------------------
# LABEL DEFINITION
# Labels come ONLY from the structured Issue / Sub-issue fields, never from the
# narrative, so the models must learn risk from the free text itself.
# Priority order: fraud > churn > other (classes are mutually exclusive).
# !! Run --explore first and check these patterns against the real Issue values.
# ---------------------------------------------------------------------------

# Scope: banking-type products only. Credit-reporting and debt-collection
# complaints are about credit bureaus / collectors, not bank customers, and
# they dominate recent CFPB data (about 90% of the sample), so they are excluded.
BANKING_PRODUCTS = {
    "Checking or savings account",
    "Bank account or service",
    "Credit card",
    "Credit card or prepaid card",
    "Prepaid card",
    "Money transfer, virtual currency, or money service",
    "Payday loan, title loan, personal loan, or advance loan",
    "Vehicle loan or lease",
    "Mortgage",
}

FRAUD_PATTERN = re.compile(
    r"fraud|scam|unauthorized|identity theft|stolen|without my consent",
    re.I,
)
# Issues whose names contain fraud-like words but are NOT fraud events
# (service complaints or lost items), found by inspecting the real Issue counts.
NOT_FRAUD_ISSUES = {
    "credit monitoring or identity theft protection services",
    "problem with fraud alerts or security freezes",
    "lost or stolen money order",
    "lost or stolen check",
    "lost or stolen refund",
}
CHURN_ISSUES = {"closing an account", "closing your account"}
CHURN_SUBISSUE_PATTERN = re.compile(
    r"company closed|can.t close|closed account|trouble closing", re.I
)

LABELS = {"other": 0, "fraud": 1, "churn": 2}


def assign_label(issue: str, sub_issue: str) -> str:
    issue, sub_issue = str(issue), str(sub_issue)
    if (issue.strip().lower() not in NOT_FRAUD_ISSUES
            and FRAUD_PATTERN.search(f"{issue} {sub_issue}")):
        return "fraud"
    if issue.strip().lower() in CHURN_ISSUES or CHURN_SUBISSUE_PATTERN.search(sub_issue):
        return "churn"
    return "other"


# ---------------------------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------------------------
REDACTION = re.compile(r"\b(X{2,}|x{2,})\b|\{\$[\d.,]+\}")  # XXXX, {$12.00}


def clean_text(s: str) -> str:
    s = REDACTION.sub(" ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# ---------------------------------------------------------------------------
# LOADING
# ---------------------------------------------------------------------------
def load_raw(path: str, min_year: int = 2020) -> pd.DataFrame:
    if str(path).endswith(".parquet"):
        # Archived mirror (complaints.parquet). Filter while reading so the full
        # 17M-row file never has to fit in memory.
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        wanted = {**PARQUET_RENAME}
        file_cols = pq.ParquetFile(path).schema_arrow.names
        missing = [c for c in wanted if c not in file_cols]
        if missing:
            raise SystemExit(f"Parquet file is missing expected columns: {missing}\n"
                             f"Columns found: {file_cols}")
        keep = (pc.field("product").isin(sorted(BANKING_PRODUCTS))
                & pc.field("narrative").is_valid())
        df = (pq.read_table(path, columns=list(wanted), filters=keep)
                .to_pandas().rename(columns=wanted))
    else:
        header = pd.read_csv(path, nrows=0).columns
        missing = [c for c in USECOLS if c not in header]
        if "Consumer complaint narrative" in missing:
            raise SystemExit(
                "This CSV has no 'Consumer complaint narrative' column, so there is "
                "no text to classify. The CFPB removed narratives from its live "
                "database in September 2026; use an archived copy (parquet)."
            )
        if missing:
            raise SystemExit(f"CSV is missing expected columns: {missing}")
        df = pd.read_csv(path, usecols=USECOLS, low_memory=False)
        df = df.dropna(subset=["Consumer complaint narrative"])

    df = df[df["Consumer complaint narrative"].astype(str).str.strip() != ""]
    df["date"] = pd.to_datetime(df["Date received"])
    df = df[(df["date"].dt.year >= min_year) & df["Product"].isin(BANKING_PRODUCTS)].copy()
    return df


def explore(path: str) -> None:
    df = load_raw(path)
    print(f"Rows with narrative since 2020: {len(df):,}\n")
    print("Products:\n", df["Product"].value_counts().head(15), "\n")
    print("Top 60 Issues:\n", df["Issue"].value_counts().head(60), "\n")
    df["label"] = [assign_label(i, str(s)) for i, s in zip(df["Issue"], df["Sub-issue"])]
    print("Label distribution:\n", df["label"].value_counts(normalize=True).round(3), "\n")
    for lab in ("fraud", "churn"):
        print(f"--- Issues mapped to '{lab}' ---")
        print(df.loc[df["label"] == lab, "Issue"].value_counts().head(25), "\n")
    df["n_words"] = df["Consumer complaint narrative"].str.split().str.len()
    print("Narrative length (words):\n", df["n_words"].describe())


# ---------------------------------------------------------------------------
# BUILD SPLITS
# ---------------------------------------------------------------------------
def subsample_by_class(df: pd.DataFrame, n_total: int, seed: int) -> pd.DataFrame:
    """Stratified subsample: keep ~n_total rows with the same class proportions."""
    if len(df) <= n_total:
        return df
    return df.groupby("label").sample(frac=n_total / len(df), random_state=seed)


def time_split(df: pd.DataFrame, train_frac: float = 0.70, val_end: float = 0.85) -> dict:
    """Split by time: oldest 70% -> train, next 15% -> validation, newest 15% -> test."""
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    a, b = int(train_frac * n), int(val_end * n)
    return {"train": df.iloc[:a], "val": df.iloc[a:b], "test": df.iloc[b:]}


def build(path: str, out_dir: str = "data/processed",
          n_total: int = 60_000, seed: int = 42) -> None:
    df = load_raw(path)
    df["label"] = [assign_label(i, str(s)) for i, s in zip(df["Issue"], df["Sub-issue"])]
    df["label_id"] = df["label"].map(LABELS)
    df["text"] = df["Consumer complaint narrative"].map(clean_text)
    df = df[df["text"].str.split().str.len() >= 10]  # drop near-empty narratives

    df = subsample_by_class(df, n_total, seed)     # keeps class ratios; makes fine-tuning feasible
    splits = time_split(df)                        # time-based, not random

    keep = ["Complaint ID", "date", "Product", "Issue", "text", "label", "label_id"]
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part[keep].to_parquet(out / f"{name}.parquet", index=False)
        print(f"{name}: {len(part):,} rows | "
              f"{part['date'].min().date()} -> {part['date'].max().date()}")
        print(part["label"].value_counts(normalize=True).round(3).to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--explore", metavar="CSV")
    ap.add_argument("--build", metavar="CSV")
    a = ap.parse_args()
    if a.explore:
        explore(a.explore)
    elif a.build:
        build(a.build)
    else:
        ap.print_help()
