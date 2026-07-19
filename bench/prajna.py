"""
Prajna — a prototype embedded DSL for AI-native programming.

Synthesis of three design sketches (Synapse, Prajna, and the graded-type
proposal). Core commitments:

  1. Belief types: soft values carry (value, confidence, source, cost).
  2. Grounding: soft -> crisp flow requires an explicit, checkable gate.
  3. Semantic functions: declared by examples + contract; the "compiler"
     lowers them to the cheapest implementation that passes the contract
     (rule synthesis -> distilled statistical head -> LLM oracle).
  4. Deterministic replay: every soft call is memoised against
     (function, implementation-version, input); identical runs are
     bit-identical.
  5. SDM semantic cache: a sparse distributed memory addressed by
     rank-order N-of-M codes sits in front of the model, answering
     near-match queries in microseconds.
  6. Hot-path distillation: traces produced by expensive levels are used
     to re-lower the function to a cheaper level over time.

This is a semantics prototype, not a production system: the "LLM" level
is an injectable oracle so the whole thing runs offline and
deterministically.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# 1. Belief: the soft value type
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Belief:
    """A soft value. Cannot silently become crisp: use ground()."""
    value: Any
    confidence: float          # in [0, 1]
    source: str                # which implementation produced it
    cost: float = 0.0          # abstract cost units of producing it

    def __repr__(self) -> str:
        return (f"Belief({self.value!r}, conf={self.confidence:.2f}, "
                f"source={self.source}, cost={self.cost:g})")


class GroundingError(Exception):
    """Raised when a soft value fails to pass into a crisp context."""


def ground(b: Belief,
           by: Optional[Callable[[Any], bool]] = None,
           threshold: float = 0.0,
           fallback: Optional[Callable[[], Any]] = None) -> Any:
    """The soft->crisp gate. Validator and/or confidence threshold.

    Either returns a plain (crisp) value or fails *deterministically*.
    Hallucination becomes a visible, catchable error class.
    """
    ok = b.confidence >= threshold and (by is None or bool(by(b.value)))
    if ok:
        return b.value
    if fallback is not None:
        return fallback()
    raise GroundingError(
        f"cannot ground {b!r}: threshold={threshold}, "
        f"validator={'passed' if by is None else 'failed'}")


# --------------------------------------------------------------------------
# 2. Rank-order N-of-M encoder  (rank-order codes over a hashed token space)
# --------------------------------------------------------------------------

class RankOrderEncoder:
    """Encode text as the rank-ordered top-N of M hashed feature dims.

    A code is a tuple of dim indices ordered by activation strength —
    an N-of-M rank-order code in the Furber/Bose sense: information is
    carried by *which* N lines fire and in *what order*, not by values.
    """

    def __init__(self, M: int = 2048, N: int = 24, alpha: float = 0.9):
        self.M, self.N, self.alpha = M, N, alpha
        # significance weights by rank: w_r = alpha^r
        self._w = alpha ** np.arange(N)
        self._wnorm = float(np.sum(self._w * self._w))

    def _features(self, text: str) -> np.ndarray:
        acts = np.zeros(self.M)
        toks = re.findall(r"[a-z0-9']+", text.lower())
        grams = toks + [" ".join(p) for p in zip(toks, toks[1:])]
        for g in grams:
            h = int(hashlib.md5(g.encode()).hexdigest(), 16)
            acts[h % self.M] += 1.0 + 0.1 * ((h >> 16) % 7)  # tie-break
        return acts

    def encode(self, text: str) -> Tuple[int, ...]:
        acts = self._features(text)
        n = min(self.N, int(np.count_nonzero(acts)))
        order = np.argsort(-acts, kind="stable")[:n]
        return tuple(int(i) for i in order)

    def similarity(self, a: Sequence[int], b: Sequence[int]) -> float:
        """Rank-weighted overlap in [0,1]: shared dims score by the
        product of their rank significances (early ranks matter most)."""
        rb = {d: i for i, d in enumerate(b)}
        s = 0.0
        for i, d in enumerate(a):
            j = rb.get(d)
            if j is not None:
                s += self._w[i] * self._w[j]
        return s / self._wnorm


# --------------------------------------------------------------------------
# 3. SDM semantic cache
# --------------------------------------------------------------------------

class SDMCache:
    """A small sparse distributed memory used as a semantic cache.

    Hard locations are random rank-order codes. A write activates all
    locations within an activation radius of the address and superimposes
    the payload (a label id) onto their counters. A read activates the
    same neighbourhood and takes a confidence-weighted vote.

    This gives *near-match* retrieval: a paraphrase of a previously seen
    input activates an overlapping set of hard locations and recovers the
    stored answer without touching any model.
    """

    def __init__(self, encoder: RankOrderEncoder,
                 n_locations: int = 512,
                 activation: float = 0.12,
                 seed: int = 7):
        self.enc = encoder
        self.activation = activation
        rng = np.random.default_rng(seed)
        # random hard addresses: random N-of-M rank-order codes
        self.addresses: List[Tuple[int, ...]] = [
            tuple(int(x) for x in rng.choice(encoder.M, size=encoder.N,
                                             replace=False))
            for _ in range(n_locations)
        ]
        self.labels: List[str] = []            # label id -> label
        self._label_ix: Dict[str, int] = {}
        self.counters = np.zeros((n_locations, 0))
        self.hits = 0
        self.misses = 0

    def _label_id(self, label: str) -> int:
        if label not in self._label_ix:
            self._label_ix[label] = len(self.labels)
            self.labels.append(label)
            self.counters = np.hstack(
                [self.counters, np.zeros((self.counters.shape[0], 1))])
        return self._label_ix[label]

    def _activated(self, code: Tuple[int, ...]) -> List[Tuple[int, float]]:
        out = []
        for i, addr in enumerate(self.addresses):
            s = self.enc.similarity(code, addr)
            if s >= self.activation:
                out.append((i, s))
        return out

    def store(self, text: str, label: str) -> int:
        code = self.enc.encode(text)
        j = self._label_id(label)
        act = self._activated(code)
        for i, s in act:
            self.counters[i, j] += s
        return len(act)

    def lookup(self, text: str, min_conf: float = 0.55) -> Optional[Belief]:
        if not self.labels:
            self.misses += 1
            return None
        code = self.enc.encode(text)
        act = self._activated(code)
        if not act:
            self.misses += 1
            return None
        votes = np.zeros(len(self.labels))
        for i, s in act:
            votes += s * self.counters[i]
        total = float(votes.sum())
        if total <= 0:
            self.misses += 1
            return None
        j = int(np.argmax(votes))
        conf = float(votes[j] / total)
        if conf < min_conf:
            self.misses += 1
            return None
        self.hits += 1
        return Belief(self.labels[j], conf, source="sdm_cache", cost=0.001)


# --------------------------------------------------------------------------
# 4. Implementation levels for semantic functions
# --------------------------------------------------------------------------

class RuleSynthesizer:
    """Level 0: synthesise discriminative keyword rules from examples."""
    name = "rules"
    cost = 0.01

    def fit(self, X: List[str], y: List[str]) -> "RuleSynthesizer":
        toks_by_label: Dict[str, Dict[str, int]] = {}
        df: Dict[str, set] = {}
        for text, label in zip(X, y):
            toks = set(re.findall(r"[a-z']+", text.lower()))
            toks_by_label.setdefault(label, {})
            for t in toks:
                toks_by_label[label][t] = toks_by_label[label].get(t, 0) + 1
                df.setdefault(t, set()).add(label)
        # keep tokens exclusive to one label
        self.rules: Dict[str, set] = {
            lab: {t for t, c in cnt.items() if len(df[t]) == 1}
            for lab, cnt in toks_by_label.items()
        }
        self.default = max(set(y), key=y.count)
        return self

    def predict(self, text: str) -> Belief:
        toks = set(re.findall(r"[a-z']+", text.lower()))
        scores = {lab: len(toks & kws) for lab, kws in self.rules.items()}
        best = max(scores, key=scores.get)
        hits = scores[best]
        if hits == 0:
            return Belief(self.default, 0.34, self.name, self.cost)
        margin = hits - sorted(scores.values())[-2] if len(scores) > 1 else hits
        conf = min(0.95, 0.55 + 0.15 * margin)
        return Belief(best, conf, self.name, self.cost)


class DistilledHead:
    """Level 1: char/word n-gram hashing + logistic regression.

    Stands in for 'a small learned head over frozen embeddings' — the
    thing hot paths get distilled into.
    """
    name = "distilled_head"
    cost = 0.1

    def fit(self, X: List[str], y: List[str]) -> "DistilledHead":
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        self.pipe = make_pipeline(
            HashingVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                              n_features=2 ** 14, alternate_sign=False),
            LogisticRegression(max_iter=2000, C=4.0))
        self.pipe.fit(X, y)
        return self

    def predict(self, text: str) -> Belief:
        proba = self.pipe.predict_proba([text])[0]
        j = int(np.argmax(proba))
        return Belief(str(self.pipe.classes_[j]), float(proba[j]),
                      self.name, self.cost)


class LLMOracle:
    """Level 2: the 'frontier model'. Injectable so the prototype runs
    offline; in a real system this is an API call. Expensive."""
    name = "llm"
    cost = 10.0

    def __init__(self, oracle: Callable[[str], str], confidence: float = 0.97):
        self.oracle = oracle
        self.confidence = confidence
        self.calls = 0

    def fit(self, X, y):
        return self

    def predict(self, text: str) -> Belief:
        self.calls += 1
        return Belief(self.oracle(text), self.confidence, self.name, self.cost)


# --------------------------------------------------------------------------
# 5. Deterministic replay
# --------------------------------------------------------------------------

class TraceStore:
    """Memoise every soft call against (fn, impl-version, input).
    Same program + same inputs => bit-identical run. Doubles as the
    dataset that drives re-lowering (distillation)."""

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.data: Dict[str, dict] = {}
        if path and os.path.exists(path):
            with open(path) as f:
                self.data = json.load(f)

    @staticmethod
    def key(fn: str, impl_version: str, x: str) -> str:
        return hashlib.sha256(f"{fn}|{impl_version}|{x}".encode()).hexdigest()

    def get(self, k: str) -> Optional[Belief]:
        r = self.data.get(k)
        if r is None:
            return None
        return Belief(r["value"], r["confidence"], r["source"] + "+replay",
                      cost=0.0)

    def put(self, k: str, fn: str, x: str, b: Belief) -> None:
        self.data[k] = {"fn": fn, "input": x, "value": b.value,
                        "confidence": b.confidence, "source": b.source}
        if self.path:
            with open(self.path, "w") as f:
                json.dump(self.data, f, indent=1)

    def traces_for(self, fn: str, min_conf: float = 0.9) -> List[Tuple[str, str]]:
        return [(r["input"], r["value"]) for r in self.data.values()
                if r["fn"] == fn and r["confidence"] >= min_conf]


# --------------------------------------------------------------------------
# 6. The sem-function: contract-driven lowering + runtime
# --------------------------------------------------------------------------

@dataclass
class Contract:
    accuracy: float = 0.9        # required holdout accuracy
    min_confidence: float = 0.0  # runtime: below this, escalate a level


class SemFunction:
    def __init__(self, name: str, examples: List[Tuple[str, str]],
                 contract: Contract, llm: LLMOracle,
                 trace_store: TraceStore,
                 use_sdm_cache: bool = True,
                 verbose: bool = True):
        self.name = name
        self.examples = list(examples)
        self.contract = contract
        self.llm = llm
        self.traces = trace_store
        self.verbose = verbose
        self.encoder = RankOrderEncoder()
        self.cache = SDMCache(self.encoder) if use_sdm_cache else None
        self.impl = None
        self.impl_version = "v0"
        self.ladder = [RuleSynthesizer, DistilledHead]
        self.compile()

    # ---- "compilation": model lowering -----------------------------------
    def _holdout_accuracy(self, cls, X, y) -> float:
        """Leave-one-out on the examples (they're few)."""
        correct = 0
        for i in range(len(X)):
            Xt = X[:i] + X[i + 1:]
            yt = y[:i] + y[i + 1:]
            if len(set(yt)) < 2:
                return 0.0
            m = cls().fit(Xt, yt)
            if m.predict(X[i]).value == y[i]:
                correct += 1
        return correct / len(X)

    def compile(self, extra: Optional[List[Tuple[str, str]]] = None) -> None:
        data = self.examples + (extra or [])
        X = [t for t, _ in data]
        y = [l for _, l in data]
        for cls in self.ladder:
            acc = self._holdout_accuracy(cls, X, y)
            if self.verbose:
                print(f"  [lower] {self.name}: trying {cls.name:>14s} "
                      f"-> LOO acc {acc:.2f} "
                      f"(contract {self.contract.accuracy:.2f}) "
                      f"on {len(X)} examples")
            if acc >= self.contract.accuracy:
                self.impl = cls().fit(X, y)
                self.impl_version = (f"{cls.name}-"
                                     f"{hashlib.md5(json.dumps(data).encode()).hexdigest()[:8]}")
                if self.verbose:
                    print(f"  [lower] {self.name}: LINKED {cls.name} "
                          f"(cost {cls.cost}/call)")
                return
        self.impl = self.llm
        self.impl_version = f"llm-{self.llm.confidence}"
        if self.verbose:
            print(f"  [lower] {self.name}: no cheap level met contract; "
                  f"LINKED llm (cost {self.llm.cost}/call)")

    def relower(self) -> bool:
        """Hot-path distillation: fold high-confidence traces back in and
        try to link a cheaper implementation."""
        extra = [t for t in self.traces.traces_for(self.name)
                 if t not in self.examples]
        if not extra:
            return False
        before = self.impl
        if self.verbose:
            print(f"  [relower] {self.name}: retrying ladder with "
                  f"{len(extra)} distilled traces")
        self.compile(extra=extra)
        return self.impl is not before

    # ---- runtime ----------------------------------------------------------
    def __call__(self, x: str) -> Belief:
        # 1. deterministic replay
        k = TraceStore.key(self.name, self.impl_version, x)
        hit = self.traces.get(k)
        if hit is not None:
            return hit
        # 2. SDM semantic cache (near-match, sub-model latency)
        if self.cache is not None:
            c = self.cache.lookup(x)
            if c is not None:
                return c
        # 3. linked implementation, escalating on low confidence
        b = self.impl.predict(x)
        if (b.confidence < self.contract.min_confidence
                and self.impl is not self.llm):
            esc = self.llm.predict(x)
            esc = Belief(esc.value, esc.confidence,
                         f"{b.source}->escalated:{esc.source}",
                         b.cost + esc.cost)
            b = esc
        # 4. record: trace for replay/distillation, cache for near-matches
        self.traces.put(k, self.name, x, b)
        if self.cache is not None and b.confidence >= 0.75:
            self.cache.store(x, b.value)
        return b


def sem(name: str, examples: List[Tuple[str, str]], llm: LLMOracle,
        accuracy: float = 0.9, min_confidence: float = 0.0,
        trace_store: Optional[TraceStore] = None,
        use_sdm_cache: bool = True,
        verbose: bool = True) -> SemFunction:
    """The DSL entry point: declare a semantic function by contract."""
    return SemFunction(name, examples, Contract(accuracy, min_confidence),
                       llm, trace_store or TraceStore(),
                       use_sdm_cache=use_sdm_cache, verbose=verbose)
