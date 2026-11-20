# Detecting customer risk in bank complaints

COS802 (University of Pretoria) project, Msimisi Dlamini (u18276319).

**Research question.** How do baseline classifiers compare with transformer-based models in detecting customer risk (fraud, churn, other) from bank complaints, and how can SHAP explain their predictions?

The project classifies CFPB consumer complaint narratives into three classes (`other`, `fraud`, `churn`), compares TF-IDF and Word2Vec baselines with a fine-tuned DistilBERT, and explains predictions with SHAP plus a deletion-based faithfulness test.

## Headline results (single run, seed 42; preliminary)

| Model | Training rows | Macro-F1 (95% CI) |
|---|---|---|
| TF-IDF + logistic regression | 42,000 | 0.667 (0.651 to 0.683) |
| TF-IDF + linear SVM | 42,000 | 0.666 (0.650 to 0.682) |
| Word2Vec average + logistic regression | 42,000 | 0.560 (0.547 to 0.573) |
| TF-IDF + logistic regression, matched | 14,999 | 0.653 (0.636 to 0.668) |
| TF-IDF + linear SVM, matched | 14,999 | 0.652 (0.635 to 0.668) |
| DistilBERT (`distilbert-base-uncased`) | 14,999 | 0.667 (0.652 to 0.682) |

"Matched" baselines are retrained on exactly the same 14,999 rows as DistilBERT so the comparison is not confounded by training-set size. The full table is in `results/final_results_table.csv` after you run the notebook.

## Folder structure

```
customer-risk-nlp/
├── README.md
├── requirements.txt          # dependencies
├── requirements-lock.txt     # exact versions of the original run (create with pip freeze, see below)
├── notebooks/
│   └── customer_risk_pipeline.ipynb   # the full pipeline, start here
├── src/                      # the same logic as tested modules (command-line runs)
│   ├── data.py               # loading, label rules, cleaning, time split
│   ├── baselines.py          # TF-IDF and Word2Vec baselines
│   ├── transformer.py        # DistilBERT fine-tuning
│   └── evaluate.py           # metrics, bootstrap CIs, McNemar, paired bootstrap
├── data/
│   ├── raw/                  # downloaded complaints.parquet (not committed)
│   └── processed/            # train/val/test parquet files (built by the pipeline)
└── results/                  # tables, predictions, figures, saved models, SHAP caches
```

## Setup

Tested with Python 3.12 (gensim has no installer for Python 3.14).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Versions used for the reported results: Python 3.12.16, torch 2.11.0, transformers 5.22.2, gensim 4.4.0, pandas 3.1.0, numpy 2.5.4, scikit-learn 1.9.1. To reproduce with the exact versions, install from `requirements-lock.txt`. If you are re-creating the environment, record it with `pip freeze > requirements-lock.txt`.

Apple GPU note: on macOS 14 with PyTorch 2.15 the GPU backend (MPS) crashed. `torch==2.11.0` works. Use `--device cpu` (command line) or `device="cpu"` (notebook) if you hit this.

## Data

The CFPB Consumer Complaint Database is the source:
https://www.consumerfinance.gov/data-research/consumer-complaints/

The CFPB removed complaint narratives from its live database in September 2026, so this project uses an archived copy (CC0), which covers 2011-12-01 to 2026-08-11:
https://huggingface.co/datasets/vivekkopthsd/cfpb-consumer-complaints

Download `complaints.parquet` into `data/raw/`:

```bash
pip install huggingface_hub
hf download vivekkopthsd/cfpb-consumer-complaints complaints.parquet --repo-type dataset --local-dir data/raw
```

The notebook expects `data/raw/complaints.parquet`.

## How to run

### Option 1: the notebook (recommended)

Open `notebooks/customer_risk_pipeline.ipynb` and run it top to bottom. The stages are: setup, load and filter data, label rules, cleaning, time split, baselines, DistilBERT, matched baselines and significance tests, SHAP and faithfulness, final results table.

Switches near the top of the relevant stages:

| Switch | Meaning |
|---|---|
| `RERUN_BASELINES` | `False` loads saved baseline results; `True` retrains them |
| `TRANSFORMER_MODE` | `"train"` fine-tunes DistilBERT; `"smoke"` is a tiny end-to-end check; `"off"` loads saved results only |
| `CFG.max_train` | training rows for DistilBERT. `15000` was used; `0` uses all 42,000 (slower) |
| `CFG.seeds` | list of random seeds, for example `[42, 43, 44]` |
| `RERUN_SHAP` | `True` recomputes SHAP even if cached explanations exist |

### Option 2: command line (same code as modules)

Run from the repository root:

```bash
python -m src.data --explore data/raw/complaints.parquet   # inspect Issue values and label counts
python -m src.data --build   data/raw/complaints.parquet   # build data/processed/{train,val,test}.parquet
python -m src.baselines --w2v                               # TF-IDF and Word2Vec baselines
python -m src.transformer --smoke                           # quick check
python -m src.transformer --max-train 15000 --epochs 2 --seeds 42
```

## Approximate run times (Apple Silicon laptop)

| Step | Time |
|---|---|
| Baselines | minutes |
| DistilBERT, 14,999 rows, 2 epochs, one seed | about 42 minutes (GPU) |
| SHAP on 60 test complaints for the transformer | about 4 minutes |

## Method summary

- **Scope:** banking products only (credit reporting and debt collection excluded), complaints from 2020 with a narrative of at least 10 words: 618,939 complaints. A stratified 60,000-row sample is split by time 70/15/15 (train to 2025-02-13, validation to 2025-10-09, test to 2026-07-21).
- **Labels:** from the structured Issue and Sub-issue fields only, never from the narrative. Priority: fraud, then churn, then other. Some fraud-like issues that are service complaints (for example credit monitoring) are excluded from fraud.
- **Baselines:** TF-IDF (word 1 to 2 grams, sublinear tf, stop words removed) with logistic regression or LinearSVC; Word2Vec averages with logistic regression. Class weights balanced; C tuned on the validation set.
- **DistilBERT:** max length 256 tokens, 2 epochs, batch size 16, learning rate 2e-5, class-weighted cross-entropy, best epoch chosen on validation.
- **Evaluation:** macro-F1 (headline), per-class precision, recall and F1, PR-AUC, bootstrap 95% confidence intervals, McNemar exact test, paired bootstrap of macro-F1 differences.
- **Explainability:** SHAP Partition explainer with word-level masking on 60 test complaints (20 per class), plus a deletion test comparing removal of top SHAP words with random words.

## Limitations

- Single seed for DistilBERT (more seeds are supported through `CFG.seeds`).
- Texts are truncated at 256 tokens (about 32% of training texts are longer).
- Labels are proxies from the Issue field and contain noise.
- DistilBERT is trained on a 14,999-row subsample for compute reasons.
- The proposal's Twitter and Trustpilot sources were not used (paid API access and no risk labels).

## Reproducibility

Random seeds are fixed, the time split is deterministic, and splits and predictions are saved under `data/processed/` and `results/`. Data are public (CC0). Cite the CFPB as the original source and describe the Hugging Face mirror as an archived copy.
