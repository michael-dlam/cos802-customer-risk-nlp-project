# Stage 10 and 11 cells: paste in this order, after Stage 9.
# Each block is one cell. Cell types: [CODE] or [MARKDOWN].

####################################################################################################
# CELL 1 [MARKDOWN]
####################################################################################################
# ## 10. Explainability: which words drive a risk prediction?
# 
# A score alone does not tell a risk manager *why* a complaint was flagged. **SHAP** answers that for one complaint at a time. It
# hides words and re-asks the model, then shares the prediction out among the words so that **the word contributions add up to
# the model's output**. A positive value for a word means it pushed the prediction toward that class.
# 
# Design choices that keep the comparison fair:
# 
# - **One method for both models.** The same SHAP explainer, the same word-level masking (words are hidden, one word is one unit), and
#   the same complaints are used for the TF-IDF Logistic Regression and the transformer. Differences between the explanations then come from
#   the models, not from the tool.
# - **A small, balanced sample.** SHAP needs many model calls per complaint, which is slow for a transformer. We explain `N_EXPLAIN`
#   complaints, equal numbers per true class (not representative of the real class mix, but good for seeing what drives fraud and churn).
# - **The window the transformer reads.** Complaints are cut at 195 words (about 256 tokens), so both models see the same text.
# - **Approximation.** SHAP is estimated with a limited number of evaluations (`MAX_EVALS`); more evaluations are more accurate and slower.
# 
# Three questions follow: *what does a single explanation look like*, *do the two models rely on the same words*, and *can we trust the
# explanations* (a faithfulness test in 10.6).

####################################################################################################
# CELL 2 [CODE]
####################################################################################################
import pickle, textwrap
import joblib
import shap
from collections import defaultdict

N_EXPLAIN = 60          # complaints to explain, equal per true class. Start small; raise it if time allows
MAX_EVALS = 300         # model calls per complaint: more = more accurate, slower
RERUN_SHAP = False      # True: recompute even if saved explanations exist
MAX_WORDS = 195         # about 256 tokens, the window the transformer reads

def clip_words(text, n=MAX_WORDS):
    return " ".join(text.split()[:n])

per_class = N_EXPLAIN // 3
sample = pd.concat([g.sample(per_class, random_state=SEED) for _, g in splits["test"].groupby("label_id")]
                   ).reset_index(drop=True)
texts = [clip_words(t) for t in sample["text"]]
y_true_s = sample["label_id"].to_numpy()
print(f"Explaining {len(texts)} test complaints ({per_class} per class: {', '.join(LABEL_NAMES)})")

####################################################################################################
# CELL 3 [MARKDOWN]
####################################################################################################
# ### 10.1 The models to explain
# 
# SHAP only needs a function that turns a list of texts into class probabilities. For Logistic Regression that is the TF-IDF vectoriser
# plus the classifier from Stage 6 (loaded from disk if saved, otherwise refitted with the same settings, which takes seconds).

####################################################################################################
# CELL 4 [CODE]
####################################################################################################
vec_path, lr_path = f"{RESULTS_DIR}/models/tfidf_vectorizer.joblib", f"{RESULTS_DIR}/models/tfidf_logreg.joblib"
if os.path.exists(vec_path) and os.path.exists(lr_path):
    vec_x, lr_x = joblib.load(vec_path), joblib.load(lr_path)
    print("Loaded the saved TF-IDF vectoriser and Logistic Regression")
else:
    best_C = float(base.loc[base["model"] == "tfidf_logreg", "best_C"].iloc[0])
    vec_x = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2),
                            min_df=5, max_df=0.9, max_features=200_000, sublinear_tf=True)
    lr_x = LogisticRegression(C=best_C, max_iter=1000, class_weight="balanced", random_state=SEED
                              ).fit(vec_x.fit_transform(splits["train"]["text"]), ytr)
    print(f"Refitted Logistic Regression (C = {best_C})")

def f_lr(texts):
    return lr_x.predict_proba(vec_x.transform([str(t) for t in texts]))

# Consistency check: this must reproduce the Logistic Regression test score from Stage 6
check = compute_metrics(yte, f_lr(splits["test"]["text"]).argmax(1))["macro_f1"]
print(f"Test macro-F1 of this model: {check:.4f}  (Stage 6 reported {base.loc[base['model'] == 'tfidf_logreg', 'macro_f1'].iloc[0]:.4f})")

####################################################################################################
# CELL 5 [CODE]
####################################################################################################
def make_transformer_proba(model, tokenizer, device, max_len, batch_size=32):
    import torch
    model.eval()
    def f(texts):
        texts, out = [str(t) for t in texts], []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = tokenizer(texts[i:i + batch_size], truncation=True, max_length=max_len,
                                padding=True, return_tensors="pt").to(device)
                out.append(torch.softmax(model(**enc).logits.float(), dim=-1).cpu().numpy())
        return np.vstack(out)
    return f

models = {"tfidf_logreg": f_lr}
tf_dir = Path(RESULTS_DIR) / "models" / best_tf if HAVE_TF else None      # best_tf comes from Stage 9
if tf_dir is not None and tf_dir.exists():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok_x = AutoTokenizer.from_pretrained(CFG.model)
    dev_x = pick_device(CFG.device)
    tmodel = AutoModelForSequenceClassification.from_pretrained(str(tf_dir)).to(dev_x)
    models[best_tf] = make_transformer_proba(tmodel, tok_x, dev_x, CFG.max_len)
    print(f"Loaded the fine-tuned transformer {best_tf} on {dev_x}")
else:
    print("No saved transformer found in results/models/: only Logistic Regression will be explained.")

####################################################################################################
# CELL 6 [MARKDOWN]
####################################################################################################
# ### 10.2 Computing the explanations
# 
# `explain` runs SHAP on each complaint and keeps the words and their contributions to **all three classes**. It saves the result, because
# SHAP is slow and should not be recomputed by accident. The timing line shows how long the full sample will take; interrupt the cell
# and lower `N_EXPLAIN` if it is too long.

####################################################################################################
# CELL 7 [CODE]
####################################################################################################
def explain(name, f):
    cache = f"{RESULTS_DIR}/shap_{name}_n{N_EXPLAIN}_e{MAX_EVALS}.pkl"
    if os.path.exists(cache) and not RERUN_SHAP:
        print(f"[{name}] loading saved explanations")
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    explainer = shap.Explainer(f, shap.maskers.Text(r"\W+"), output_names=LABEL_NAMES)   # word-level masking
    exs, t0 = [], time.time()
    for i, t in enumerate(texts):
        e = explainer([t], max_evals=MAX_EVALS, batch_size=32, silent=True)[0]
        exs.append({"tokens": list(e.data), "values": np.asarray(e.values)})
        if i in (0, 4) or (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  [{name}] {i + 1}/{len(texts)} | {el / 60:.1f} min elapsed, ~{el / (i + 1) * (len(texts) - i - 1) / 60:.1f} min left")
    out = {"examples": exs, "probs": f(texts)}
    with open(cache, "wb") as fh:
        pickle.dump(out, fh)
    return out

explained = {name: explain(name, f) for name, f in models.items()}
for name, out in explained.items():
    pred = out["probs"].argmax(1)
    print(f"{name}: predicts {[int((pred == k).sum()) for k in range(3)]} complaints as {LABEL_NAMES} (true: {[per_class] * 3})")

####################################################################################################
# CELL 8 [MARKDOWN]
####################################################################################################
# ### 10.3 One complaint, in detail
# 
# For a single complaint: what each model predicted, and the words that pushed it toward that class (positive) or away from it (negative).

####################################################################################################
# CELL 9 [CODE]
####################################################################################################
def clean_word(tok):
    return re.sub(r"\W+", "", tok.lower())

def top_words(ex, k, n=8):
    toks, vals = [clean_word(t) for t in ex["tokens"]], ex["values"][:, k]
    order = np.argsort(-vals)
    pos = [(toks[j], round(float(vals[j]), 3)) for j in order[:n] if vals[j] > 0]
    neg = [(toks[j], round(float(vals[j]), 3)) for j in order[::-1][:5] if vals[j] < 0]
    return pos, neg

def show_example(i):
    print(f"TRUE CLASS: {LABEL_NAMES[y_true_s[i]]} | Issue: {sample.loc[i, 'Issue']}")
    print(textwrap.fill(texts[i][:600], 100) + " ...\n")
    for name, out in explained.items():
        p = out["probs"][i]; k = int(p.argmax())
        pos, neg = top_words(out["examples"][i], k)
        print(f"[{name}] predicts '{LABEL_NAMES[k]}' (probabilities {np.round(p, 2)})")
        print("   pushes toward it:", pos)
        print("   pushes away:     ", neg, "\n")

for cls in (1, 2):                                   # one fraud and one churn complaint
    show_example(int(np.where(y_true_s == cls)[0][0]))
    print("-" * 100)

####################################################################################################
# CELL 10 [MARKDOWN]
####################################################################################################
# ### 10.4 Do the models rely on the same words?
# 
# Across the complaints a model flagged as fraud (or churn), which words have the highest *average* contribution? `min_count` drops words
# seen too rarely to trust. Then compare with Logistic Regression's own top terms from Stage 6.4, and with the transformer.
# 
# With a sample this small the rankings are **indicative, not definitive**.

####################################################################################################
# CELL 11 [CODE]
####################################################################################################
def aggregate(name, class_id, min_count=3):
    out = explained[name]
    tot, cnt = defaultdict(float), defaultdict(int)
    for ex, p in zip(out["examples"], out["probs"].argmax(1)):
        if p != class_id:
            continue
        for tok, v in zip(ex["tokens"], ex["values"][:, class_id]):
            w = clean_word(tok)
            if len(w) >= 3:
                tot[w] += v; cnt[w] += 1
    d = pd.DataFrame({"word": list(tot), "mean_shap": [tot[w] / cnt[w] for w in tot], "n": [cnt[w] for w in tot]})
    return d[d["n"] >= min_count].sort_values("mean_shap", ascending=False).reset_index(drop=True)

terms = pd.read_csv(f"{RESULTS_DIR}/lr_top_terms.csv")
def coef_words(cls):                                  # unigrams from Logistic Regression's top terms
    return {w for t in terms[terms["class"] == cls]["term"] for w in t.split() if len(w) >= 3}

fig, axes = plt.subplots(2, len(explained), figsize=(5.5 * len(explained), 7), squeeze=False)
for r, (cls, k) in enumerate([("fraud", 1), ("churn", 2)]):
    ref, tops = coef_words(cls), {}
    for c, name in enumerate(explained):
        agg = aggregate(name, k)
        tops[name] = set(agg.head(15)["word"])
        print(f"--- {cls} | {name}: top words by average SHAP (flagged complaints: {int((explained[name]['probs'].argmax(1) == k).sum())}) ---")
        display(agg.head(12).round(3))
        if len(agg):
            axes[r][c].barh(agg.head(12)["word"][::-1], agg.head(12)["mean_shap"][::-1])
        axes[r][c].set_title(f"{cls}: {name}")
        axes[r][c].set_xlabel("mean SHAP value")
    for name, t in tops.items():
        print(f"{cls}: {len(t & ref)} of {name}'s top-15 SHAP words are among Logistic Regression's own top-25 terms")
    if len(tops) == 2:
        a, b = list(tops.values())
        print(f"{cls}: the two models share {len(a & b)} of their top-15 SHAP words\n")
fig.tight_layout(); fig.savefig(f"{FIG_DIR}/shap_top_words.png", dpi=150); plt.show()

####################################################################################################
# CELL 12 [MARKDOWN]
####################################################################################################
# > **✍️ Your reasoning:** Are the top words sensible for fraud and for churn? Do Logistic Regression and the transformer rely on the
# > same words, or does the transformer use different evidence? Do the words match Logistic Regression's own coefficients (a sanity check that
# > SHAP is behaving)? Is anything surprising or spurious? Remember the sample is small.

####################################################################################################
# CELL 13 [MARKDOWN]
####################################################################################################
# ### 10.5 Can the explanations be trusted? A faithfulness (deletion) test
# 
# An explanation is only useful if it points at words the model really depends on. Test: for complaints the model flagged as a risk class,
# **delete the k words SHAP rated most important** for that class and see how much the model's probability for that class falls. Compare
# with deleting **k random words**. If SHAP is faithful, deleting its top words should hurt much more than deleting random ones.

####################################################################################################
# CELL 14 [CODE]
####################################################################################################
def deletion_test(f, out, ks=(1, 3, 5, 10), n_random=5, seed=SEED):
    rng = np.random.default_rng(seed)
    probs, exs = out["probs"], out["examples"]
    pred = probs.argmax(1)
    idx = [i for i in range(len(exs)) if pred[i] != 0]                 # only complaints flagged as fraud or churn
    if not idx:
        return pd.DataFrame()
    cls, p0 = pred[idx], probs[idx, pred[idx]]
    rows = []
    for k in ks:
        top_texts, rand_texts = [], []
        for i in idx:
            toks, vals = exs[i]["tokens"], exs[i]["values"][:, pred[i]]
            kk = min(k, len(toks))
            drop_top = set(np.argsort(-vals)[:kk])
            top_texts.append("".join(t for j, t in enumerate(toks) if j not in drop_top))
            for _ in range(n_random):
                drop_rand = set(rng.choice(len(toks), kk, replace=False))
                rand_texts.append("".join(t for j, t in enumerate(toks) if j not in drop_rand))
        p_top_all, p_rand_all = f(top_texts), f(rand_texts)
        p_top = p_top_all[np.arange(len(idx)), cls]
        p_rand = p_rand_all[np.arange(len(p_rand_all)), np.repeat(cls, n_random)].reshape(len(idx), n_random).mean(1)
        rows.append({"k_words_removed": k, "n_complaints": len(idx),
                     "mean_drop_SHAP_top": float((p0 - p_top).mean()),
                     "mean_drop_random": float((p0 - p_rand).mean()),
                     "share_SHAP_hurts_more": float(((p0 - p_top) > (p0 - p_rand)).mean()),
                     "pred_flips_SHAP_top": float((p_top_all.argmax(1) != cls).mean())})
    return pd.DataFrame(rows)

faith = {name: deletion_test(models[name], explained[name]) for name in explained}
for name, d in faith.items():
    print(f"--- {name}: drop in the predicted class's probability after removing words ---")
    display(d.round(3))

fig, axes = plt.subplots(1, len(faith), figsize=(5.5 * len(faith), 4), squeeze=False)
for ax, (name, d) in zip(axes[0], faith.items()):
    if len(d):
        ax.plot(d["k_words_removed"], d["mean_drop_SHAP_top"], marker="o", label="remove top SHAP words")
        ax.plot(d["k_words_removed"], d["mean_drop_random"], marker="o", label="remove random words")
    ax.set_title(name); ax.set_xlabel("words removed"); ax.set_ylabel("mean drop in class probability"); ax.legend()
fig.tight_layout(); fig.savefig(f"{FIG_DIR}/shap_faithfulness.png", dpi=150); plt.show()

####################################################################################################
# CELL 15 [MARKDOWN]
####################################################################################################
# > **✍️ Your reasoning:** Is the drop after removing SHAP's top words larger than after removing random words, for each model? What does that
# > say about whether the explanations can be trusted? Does one model's explanations look more faithful than the other's, and what could cause that?
# >
# > **Limits to state honestly:** a small, class-balanced sample; SHAP approximated with limited evaluations; words are hidden with a
# > placeholder in SHAP but deleted in this test; only one faithfulness check; one trained transformer (one seed).

####################################################################################################
# CELL 16 [MARKDOWN]
####################################################################################################
# ## 11. Consolidated results, limitations and conclusions
# 
# ### 11.1 One table for the report
# 
# Every model on the same test set, with how much training data it saw. It is saved to `results/final_results_table.csv`.

####################################################################################################
# CELL 17 [CODE]
####################################################################################################
rows_seen = {m: (len(subset) if ("matched" in m or m.startswith("distilbert")) else len(ytr)) for m in allres["model"]}
final = allres.copy()
final.insert(1, "train_rows", final["model"].map(rows_seen))
final["macro_f1 (95% CI)"] = [f"{m:.3f} ({lo:.3f} to {hi:.3f})" for m, lo, hi in
                              zip(final["macro_f1"], final["macro_f1_ci_low"], final["macro_f1_ci_high"])]
final = final[["model", "train_rows", "macro_f1 (95% CI)", "accuracy", "pr_auc_macro", "f1_fraud", "f1_churn", "val_macro_f1"]]
final.to_csv(f"{RESULTS_DIR}/final_results_table.csv", index=False)
display(final.round(3))
print("Saved to", f"{RESULTS_DIR}/final_results_table.csv", "| transformer rows come from one seed unless several were trained")

####################################################################################################
# CELL 18 [MARKDOWN]
####################################################################################################
# ### 11.2 Limitations and conclusions
# 
# > **✍️ Your reasoning:** Write these yourself, in your own words, using the evidence above. Points to cover:
# >
# > - **Labels:** built from the Issue category the customer chose, so partly category prediction; some label noise (Stage 7), which probably caps every model.
# > - **Triage, not early warning:** complaints arrive after the event.
# > - **Data:** an archived snapshot (the live CFPB database no longer has narratives); only opt-in complaints; banking products only.
# > - **Drift:** the class mix changes over time, so the test set is harder than validation.
# > - **Compute:** a subsample for fine-tuning, one or few seeds, 256-token limit (about a third of training texts are longer); DistilBERT instead of a larger model.
# > - **Fair comparison:** the transformer saw fewer rows than the full-data baselines, so matched baselines were added; the SVM's best `C` sat at the edge of its grid.
# > - **Explainability:** a small, balanced sample; SHAP approximated with limited evaluations; one faithfulness test.
# > - **Scope change:** the proposal's Twitter data was not used (the API is now paid and limited); a free bank-tweet dataset could serve as a cross-domain test in future work.
# >
# > **Conclusion:** answer the research question directly, in a few sentences, using your numbers and not overstating them.
