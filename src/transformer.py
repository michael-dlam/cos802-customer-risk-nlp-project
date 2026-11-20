"""
transformer.py - fine-tune DistilBERT (or BERT / RoBERTa) for risk classification.

Plain PyTorch loop; runs on CUDA, Apple-silicon GPU (MPS) or CPU, whichever exists.

Typical use (from the repo root):
    python -m src.transformer --smoke          # ~2-5 min: checks everything works
    python -m src.transformer --benchmark      # times 20 steps, projects run time
    python -m src.transformer --max-train 20000 --epochs 2 --seeds 42 43 44

Method notes (match the baselines so results are directly comparable):
  * same train/val/test parquet files, same metrics (src/evaluate.py)
  * class-weighted cross-entropy ("balanced", like the baselines)
  * best epoch chosen on VALIDATION macro-F1; TEST scored once at the end
  * the training subset (if --max-train is used) is fixed across seeds; seeds
    change only weight initialisation of the classifier head and batch order
"""
import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # Apple GPU: fall back to CPU for rare ops
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from src.evaluate import (LABEL_NAMES, bootstrap_macro_f1_ci,  # noqa: E402
                          compute_metrics, plot_confusion)

NUM_LABELS = 3


# ----------------------------------------------------------------------------
# helpers that do not need torch (these are unit-testable on their own)
# ----------------------------------------------------------------------------
def stratified_subsample(df, n, seed):
    """Keep ~n rows with the same class proportions. n<=0 means keep all."""
    if n <= 0 or n >= len(df):
        return df.reset_index(drop=True)
    return (df.groupby("label_id").sample(frac=n / len(df), random_state=seed)
              .reset_index(drop=True))


def balanced_class_weights(y, k=NUM_LABELS):
    """n / (k * n_c): the same 'balanced' rule scikit-learn used for the baselines."""
    counts = np.bincount(y, minlength=k).astype(float)
    return len(y) / (k * np.maximum(counts, 1.0))


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------------
# torch helpers
# ----------------------------------------------------------------------------
def pick_device(choice="auto"):
    """choice: 'auto' (cuda > mps > cpu), or force 'cpu' / 'mps' / 'cuda'."""
    import torch
    if choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def encode(tokenizer, texts, max_len):
    """Token ids only, truncated to max_len. Padding happens per batch."""
    return tokenizer(list(texts), truncation=True, max_length=max_len)["input_ids"]


def make_batch(ids_list, labels, idx, pad_id, device):
    import torch
    seqs = [ids_list[i] for i in idx]
    L = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), L), dtype=torch.long)
    for r, s in enumerate(seqs):
        input_ids[r, :len(s)] = torch.tensor(s, dtype=torch.long)
        attn[r, :len(s)] = 1
    y = torch.tensor(np.asarray(labels)[idx], dtype=torch.long)
    return input_ids.to(device), attn.to(device), y.to(device)


def predict_logits(model, ids_list, pad_id, device, batch_size=64):
    import torch
    model.eval()
    order = np.argsort([len(s) for s in ids_list])        # length-sorted = less padding
    out = np.zeros((len(ids_list), NUM_LABELS), dtype=np.float32)
    dummy = np.zeros(len(ids_list), dtype=int)
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            x, m, _ = make_batch(ids_list, dummy, idx, pad_id, device)
            out[idx] = model(input_ids=x, attention_mask=m).logits.float().cpu().numpy()
    return out


def load_model(name, device):
    from transformers import AutoModelForSequenceClassification
    id2label = {i: n for i, n in enumerate(LABEL_NAMES)}
    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=NUM_LABELS, id2label=id2label,
        label2id={n: i for i, n in id2label.items()})
    return model.to(device)


# ----------------------------------------------------------------------------
# benchmark: time a few training steps and project the full run
# ----------------------------------------------------------------------------
def run_benchmark(args, ids_tr, ytr, n_eval, pad_id, device):
    import torch
    set_seed(0)
    model = load_model(args.model, device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    rng = np.random.default_rng(0)
    times = []
    for step in range(23):                                  # 3 warm-up + 20 timed
        idx = rng.choice(len(ids_tr), args.batch_size, replace=False)
        x, m, y = make_batch(ids_tr, ytr, idx, pad_id, device)
        sync(device)
        t0 = time.time()
        loss = loss_fn(model(input_ids=x, attention_mask=m).logits, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        sync(device)
        if step >= 3:
            times.append(time.time() - t0)
    spb = float(np.mean(times))
    steps = int(np.ceil(len(ids_tr) / args.batch_size))
    train_min = spb * steps / 60
    eval_min = spb / 3 * n_eval / args.batch_size / 60      # forward-only is ~1/3 the cost
    print(f"\ndevice: {device} | {spb:.2f} s/step at batch size {args.batch_size}, "
          f"max_len {args.max_len}")
    print(f"training examples: {len(ids_tr):,} -> about {train_min:.0f} min per epoch")
    print(f"evaluation (val each epoch + test once): about {eval_min:.1f} min per pass over "
          f"{n_eval:,} examples")
    total = args.epochs * train_min + (args.epochs + 1) * eval_min
    print(f"projected time for ONE seed with {args.epochs} epochs: ~{total:.0f} min "
          f"(rough estimate, +/-30%)")


# ----------------------------------------------------------------------------
# one full training run (one seed)
# ----------------------------------------------------------------------------
def train_one_seed(args, seed, name, ids, labels, test_df, pad_id, device, res):
    import torch
    from transformers import get_linear_schedule_with_warmup

    ids_tr, ids_va, ids_te = ids["train"], ids["val"], ids["test"]
    ytr, yva, yte = labels["train"], labels["val"], labels["test"]

    set_seed(seed)
    model = load_model(args.model, device)
    weights = torch.tensor(balanced_class_weights(ytr), dtype=torch.float32).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = int(np.ceil(len(ids_tr) / args.batch_size))
    total_steps = steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total_steps), total_steps)
    rng = np.random.default_rng(seed)

    best_f1, best_epoch, best_state = -1.0, 0, None
    t_start = time.time()
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(len(ids_tr))
        for start in range(0, len(perm), args.batch_size):
            idx = perm[start:start + args.batch_size]
            x, m, y = make_batch(ids_tr, ytr, idx, pad_id, device)
            loss = loss_fn(model(input_ids=x, attention_mask=m).logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 50 == 0 or step == total_steps:
                el = time.time() - t_start
                eta = el / step * (total_steps - step)
                print(f"  [{name}] epoch {epoch} step {step}/{total_steps} "
                      f"loss {loss.item():.4f} | {el/60:.1f} min elapsed, ~{eta/60:.1f} min left")
        val_pred = predict_logits(model, ids_va, pad_id, device).argmax(1)
        val_f1 = compute_metrics(yva, val_pred)["macro_f1"]
        print(f"  [{name}] epoch {epoch} validation macro-F1 = {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    logits = predict_logits(model, ids_te, pad_id, device)
    probs, pred = softmax(logits), logits.argmax(1)
    metrics = compute_metrics(yte, pred, probs)
    lo, hi = bootstrap_macro_f1_ci(yte, pred)

    (res / "predictions").mkdir(parents=True, exist_ok=True)
    (res / "figures").mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"Complaint ID": test_df["Complaint ID"].to_numpy(),
                        "y_true": yte, "y_pred": pred})
    for k, n in enumerate(LABEL_NAMES):
        out[f"p_{n}"] = probs[:, k]
    out.to_parquet(res / "predictions" / f"{name}_test.parquet", index=False)
    plot_confusion(yte, pred, res / "figures" / f"cm_{name}.png", name)
    if not args.no_save_model:
        mdir = res / "models" / name
        model.save_pretrained(mdir)
    return {"model": name, "seed": seed, "best_epoch": best_epoch,
            "val_macro_f1": best_f1, **metrics,
            "macro_f1_ci_low": lo, "macro_f1_ci_high": hi,
            "train_minutes": (time.time() - t_start) / 60}


# ----------------------------------------------------------------------------
def main(args):
    from transformers import AutoTokenizer

    if args.smoke:                                   # tiny end-to-end check
        args.max_train, args.max_val, args.max_test = 600, 300, 300
        args.epochs, args.seeds = 1, args.seeds[:1]
        args.results_dir = str(Path(args.results_dir) / "smoke")

    device = pick_device(args.device)
    print(f"device: {device}")
    res = Path(args.results_dir)
    res.mkdir(parents=True, exist_ok=True)

    d = {s: pd.read_parquet(Path(args.data_dir) / f"{s}.parquet")
         for s in ("train", "val", "test")}
    d["train"] = stratified_subsample(d["train"], args.max_train, args.subsample_seed)
    d["val"] = stratified_subsample(d["val"], args.max_val, args.subsample_seed)
    d["test"] = stratified_subsample(d["test"], args.max_test, args.subsample_seed)
    print({s: len(v) for s, v in d.items()})

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    t0 = time.time()
    ids = {s: encode(tokenizer, d[s]["text"], args.max_len) for s in d}
    labels = {s: d[s]["label_id"].to_numpy() for s in d}
    lens = np.array([len(x) for x in ids["train"]])
    print(f"tokenised in {time.time()-t0:.0f}s | {np.mean(lens >= args.max_len):.0%} of "
          f"training texts hit the {args.max_len}-token limit")
    pad_id = tokenizer.pad_token_id

    if args.benchmark:
        run_benchmark(args, ids["train"], labels["train"],
                      len(ids["val"]) + len(ids["test"]), pad_id, device)
        return

    short = Path(args.model).name
    rows = []
    for seed in args.seeds:
        name = f"{short}_s{seed}" + ("_smoke" if args.smoke else "")
        print(f"\n=== {name} ===")
        rows.append(train_one_seed(args, seed, name, ids, labels, d["test"],
                                   pad_id, device, res))
        # save after every seed so a crash never loses finished runs
        csv = res / "transformer_results.csv"
        new = pd.DataFrame(rows)
        if csv.exists():
            old = pd.read_csv(csv)
            new = pd.concat([old[~old["model"].isin(new["model"])], new])
        new.to_csv(csv, index=False)

    out = pd.DataFrame(rows)
    show = ["model", "best_epoch", "val_macro_f1", "accuracy", "macro_f1",
            "macro_f1_ci_low", "macro_f1_ci_high", "pr_auc_macro", "f1_fraud", "f1_churn"]
    print("\nTEST RESULTS")
    print(out[show].round(4).to_string(index=False))
    if len(out) > 1:
        print(f"\nmacro-F1 over {len(out)} seeds: {out['macro_f1'].mean():.4f} "
              f"+/- {out['macro_f1'].std(ddof=1):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--max-train", type=int, default=0, help="0 = use all training rows")
    ap.add_argument("--max-val", type=int, default=0)
    ap.add_argument("--max-test", type=int, default=0)
    ap.add_argument("--subsample-seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"],
                    help="force a device; use cpu if the Apple GPU (mps) crashes")
    ap.add_argument("--no-save-model", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    main(ap.parse_args())
