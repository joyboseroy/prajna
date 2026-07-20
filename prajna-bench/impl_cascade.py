"""Implementation 2: HAND-TUNED CASCADE (the strong baseline).

A FrugalGPT-style two-stage cascade a competent engineer would build:
TF-IDF + logistic regression, confidence threshold tuned on a dev split,
escalate to the LLM below threshold, plus the usual hand-rolled action
gating. This is deliberately written the way such code really looks —
explicit tuning loops, explicit gating — because lines-of-code and
maintainability are part of what's being measured."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from common import (ACTION_THRESHOLD, CONSEQUENTIAL, HEAD_COST,
                    HEAD_LATENCY_MS, LLM_COST, LLM_LATENCY_MS)


def train_head(fit):
    X = [t for t, _ in fit]
    y = [l for _, l in fit]
    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                        min_df=2),
        LogisticRegression(max_iter=2000, C=8.0))
    pipe.fit(X, y)
    return pipe


def tune_threshold(pipe, dev, llm, target_acc):
    """Grid-search the escalation threshold on the dev split: the
    engineer's manual version of what a planner should automate.
    Estimates cascade accuracy = head-acc above tau + llm-acc below."""
    X = [t for t, _ in dev]
    y = [l for _, l in dev]
    proba = pipe.predict_proba(X)
    conf = proba.max(axis=1)
    pred = pipe.classes_[proba.argmax(axis=1)]
    # estimate llm accuracy on a dev sample (costs real llm calls)
    sample = dev[:300]
    llm_hits = sum(llm.classify(t, l) == l for t, l in sample)
    llm_acc = llm_hits / len(sample)
    tune_calls = len(sample)
    best = None
    best_effort = None   # max estimated accuracy across ALL taus tried
    for tau in np.arange(0.20, 0.96, 0.02):
        keep = conf >= tau
        n_keep = int(keep.sum())
        head_acc = (pred[keep] == np.array(y)[keep]).mean() if n_keep else 0
        frac_llm = 1 - n_keep / len(dev)
        est = head_acc * (1 - frac_llm) + llm_acc * frac_llm
        if est >= target_acc:
            if best is None or frac_llm < best[1]:
                best = (float(tau), frac_llm, est)
        if best_effort is None or est > best_effort[2]:
            best_effort = (float(tau), frac_llm, est)
    contract_met = best is not None
    chosen = best if contract_met else best_effort
    if not contract_met:
        print(f"  [cascade] CONTRACT INFEASIBLE (best achievable "
              f"est_acc={chosen[2]:.3f} < target {target_acc:.2f}); "
              f"grid-searched best-effort tau={chosen[0]:.2f} "
              f"(NOT the old hardcoded 0.95 default)")
    return chosen[0], tune_calls, llm_acc


def run(test, llm, fit, dev, target_acc=0.92):
    pipe = train_head(fit)
    tau, tune_calls, llm_acc = tune_threshold(pipe, dev, llm, target_acc)
    preds, lat, cost, actions, deferred = [], [], 0.0, [], 0
    for text, true in test:
        proba = pipe.predict_proba([text])[0]
        conf = float(proba.max())
        label = str(pipe.classes_[proba.argmax()])
        c, l = HEAD_COST, HEAD_LATENCY_MS
        if conf < tau:
            label = llm.classify(text, true)
            conf = llm_acc  # calibrated: measured LLM accuracy on dev
                             # sample, not a hardcoded trust constant
            c += LLM_COST
            l += LLM_LATENCY_MS
        preds.append(label)
        cost += c
        lat.append(l)
        # hand-rolled action gating: correct-ish, but ad hoc, and
        # nothing stops the next contributor from skipping it
        if label in CONSEQUENTIAL:
            if conf >= ACTION_THRESHOLD:
                actions.append(("block_card", text, true))
            else:
                deferred += 1
    return dict(preds=preds, latencies=lat, cost=cost + tune_calls * LLM_COST,
                actions=actions, deferred=deferred, unguarded_flows=0,
                tau=tau)
