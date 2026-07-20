"""Shared infrastructure for the Prajna benchmark: data loading, the
simulated LLM oracle, and the consequential-action task definition.

HONESTY NOTE (also in README): the LLM here is *simulated* — it returns
the true label with probability `accuracy`, else a deterministic wrong
label. This means absolute accuracy numbers are assumptions, not
findings. What the benchmark genuinely measures: cost structure, LLM
call counts, latency profile, code size, reproducibility, and safety
guarantees — and *relative* accuracy between systems is consistent
because all three share the same oracle. Swap in RealLLM for real
findings.
"""

import csv
import hashlib
import os
import random

# ---- cost/latency model (nominal units, same for all systems) ----------
LLM_COST, LLM_LATENCY_MS = 10.0, 400.0
HEAD_COST, HEAD_LATENCY_MS = 0.1, 2.0
RULE_COST, RULE_LATENCY_MS = 0.01, 0.1

# ---- the consequential-action sub-task ---------------------------------
# Predicting one of these intents triggers block_card(), an external
# effect. A wrong block on an innocent query is the costly failure mode.
CONSEQUENTIAL = {"lost_or_stolen_card", "compromised_card", "card_swallowed"}
ACTION_THRESHOLD = 0.95   # policy: block only at >= this confidence


def load_data(data_dir="data"):
    def read(path):
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return [(r["text"], r["category"]) for r in rows]
    train = read(os.path.join(data_dir, "train.csv"))
    test = read(os.path.join(data_dir, "test.csv"))
    random.Random(42).shuffle(train)   # train.csv is sorted by category
    fit, dev = train[:9000], train[9000:]
    labels = sorted({y for _, y in train})
    return fit, dev, test, labels


class SimulatedLLM:
    """Deterministic stand-in for a frontier model. Correct with
    probability `accuracy`; errors are deterministic per input."""

    def __init__(self, labels, accuracy=0.95):
        self.labels = labels
        self.accuracy = accuracy
        self.calls = 0

    def classify(self, text, true_label):
        self.calls += 1
        h = int(hashlib.md5(text.encode()).hexdigest(), 16)
        if (h % 10_000) / 10_000 < self.accuracy:
            return true_label
        wrong = [l for l in self.labels if l != true_label]
        return wrong[h % len(wrong)]


class RealLLM:
    """Where a real API goes. Same interface; ignores true_label."""

    def __init__(self, labels, client=None, model="llama-3.3-70b"):
        self.labels, self.client, self.model = labels, client, model
        self.calls = 0

    def classify(self, text, true_label=None):
        self.calls += 1
        raise NotImplementedError(
            "plug in an API client; prompt with the 77 label names and "
            "parse the response against `self.labels`")
