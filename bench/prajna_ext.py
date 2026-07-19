"""prajna_ext: the Inference Planner (candidate for prajna core v0.3).

Extends SemFunction with the semantics the benchmark argues for: the
contract specifies END-TO-END accuracy *including escalation*, and the
compiler — not the programmer — selects both the implementation level
and the escalation threshold, by estimating cascade accuracy on a dev
split and minimising expected cost subject to the contract. This
automates exactly the tuning loop the hand-built cascade does manually.

Also adds a TfidfHead capability so the planner's catalog matches what
a competent engineer would hand-build (fair comparison)."""

import numpy as np

from prajna import Belief, DistilledHead, RuleSynthesizer, SemFunction

from common import (HEAD_COST, HEAD_LATENCY_MS, LLM_COST, RULE_COST)


class TfidfHead:
    name = "tfidf_head"
    cost = HEAD_COST

    def fit(self, X, y):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        self.pipe = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
            LogisticRegression(max_iter=2000, C=8.0))
        self.pipe.fit(X, y)
        return self

    def predict(self, text):
        p = self.pipe.predict_proba([text])[0]
        j = int(np.argmax(p))
        return Belief(str(self.pipe.classes_[j]), float(p[j]),
                      self.name, self.cost)

    def predict_batch(self, X):
        p = self.pipe.predict_proba(X)
        return self.pipe.classes_[p.argmax(1)], p.max(1)


class PlannedSemFunction(SemFunction):
    """Contract-driven planning over (level, escalation threshold)."""

    def __init__(self, name, fit_set, dev_set, contract_accuracy, llm,
                 trace_store, verbose=True):
        self._fit_set, self._dev_set = fit_set, dev_set
        self._planner_llm = llm          # oracle with .classify()
        self.plan = None
        # bypass parent's compile-on-init; we plan explicitly
        self.name = name
        self.examples = fit_set
        self.traces = trace_store
        self.verbose = verbose
        self.cache = None                # SDM cache off: separate claim
        self.ladder = [RuleSynthesizer, DistilledHead, TfidfHead]
        self.contract_accuracy = contract_accuracy
        self._compile_plan()

    def _compile_plan(self):
        X = [t for t, _ in self._fit_set]
        y = [l for _, l in self._fit_set]
        Xd = [t for t, _ in self._dev_set]
        yd = np.array([l for _, l in self._dev_set])
        # one-time estimate of LLM accuracy on a dev sample
        sample = self._dev_set[:300]
        llm_acc = (sum(self._planner_llm.classify(t, l) == l
                       for t, l in sample) / len(sample))
        self.plan_compile_cost = len(sample) * LLM_COST
        best = None
        self.plan_candidates = []
        for cls in self.ladder:
            impl = cls().fit(X, y)
            if hasattr(impl, "predict_batch"):
                pred, conf = impl.predict_batch(Xd)
            else:
                bs = [impl.predict(t) for t in Xd]
                pred = np.array([b.value for b in bs])
                conf = np.array([b.confidence for b in bs])
            level_best = None
            solo_acc = float((pred == yd).mean())
            for tau in np.arange(0.20, 0.96, 0.02):
                keep = conf >= tau
                nk = int(keep.sum())
                head_acc = float((pred[keep] == yd[keep]).mean()) if nk else 0
                frac_llm = 1 - nk / len(yd)
                est = head_acc * (1 - frac_llm) + llm_acc * frac_llm
                exp_cost = impl.cost + frac_llm * LLM_COST
                cand = dict(impl=impl, level=cls.name, tau=float(tau),
                            exp_cost=exp_cost, est_acc=est,
                            frac_llm=frac_llm, solo_acc=solo_acc,
                            feasible=est >= self.contract_accuracy)
                if cand["feasible"] and (level_best is None
                                         or exp_cost < level_best["exp_cost"]):
                    level_best = cand
                if cand["feasible"] and (best is None
                                         or exp_cost < best["exp_cost"]):
                    best = cand
            if level_best is None:
                level_best = dict(level=cls.name, tau=None, est_acc=None,
                                  solo_acc=solo_acc, feasible=False,
                                  exp_cost=None, frac_llm=None)
            self.plan_candidates.append(level_best)
        self.llm_est_acc = llm_acc
        if best is None:   # nothing meets contract: link the LLM alone
            best = dict(impl=None, level="llm", tau=1.0,
                        exp_cost=LLM_COST, est_acc=llm_acc, frac_llm=1.0)
        self.plan = best
        self.impl_version = f"planned-{best['level']}-tau{best['tau']:.2f}"
        if self.verbose:
            print(f"  [plan] {self.name}: linked {best['level']} "
                  f"tau={best['tau']:.2f} est_acc={best['est_acc']:.3f} "
                  f"est_llm_frac={best['frac_llm']:.2f} "
                  f"exp_cost={best['exp_cost']:.2f}/call")

    def __call__(self, text, _true=None):
        k = self.traces.key(self.name, self.impl_version, text)
        hit = self.traces.get(k)
        if hit is not None:
            return hit
        p = self.plan
        if p["impl"] is not None:
            b = p["impl"].predict(text)
            if b.confidence < p["tau"]:
                v = self._planner_llm.classify(text, _true)
                b = Belief(v, 0.95, f"{b.source}->escalated:llm",
                           b.cost + LLM_COST)
        else:
            v = self._planner_llm.classify(text, _true)
            b = Belief(v, 0.95, "llm", LLM_COST)
        self.traces.put(k, self.name, text, b)
        return b


    def explain(self, effects=None):
        """Cognitive execution plan, in the spirit of SQL EXPLAIN."""
        p = self.plan
        out = [f"COGNITIVE EXECUTION PLAN: {self.name}",
               "",
               "Contract:",
               f"  estimated end-to-end accuracy >= "
               f"{self.contract_accuracy:.2f}   [STATISTICAL]",
               f"  objective: minimise expected cost per call",
               "",
               "Candidates considered (best feasible tau per level):"]
        for c in self.plan_candidates:
            if c["feasible"]:
                mark = "chosen " if c["level"] == p["level"] and                     abs(c["tau"] - p["tau"]) < 1e-9 else "feasible"
                out.append(
                    f"  [{mark}] {c['level']:<15} solo_acc={c['solo_acc']:.3f}"
                    f"  tau={c['tau']:.2f}  est_acc={c['est_acc']:.3f}"
                    f"  escalation={c['frac_llm']:.1%}"
                    f"  exp_cost={c['exp_cost']:.2f}")
            else:
                out.append(
                    f"  [rejected] {c['level']:<13} solo_acc={c['solo_acc']:.3f}"
                    f"  no threshold meets contract")
        out += ["",
                "Plan:",
                f"  Stage 1: {p['level']}  (cost "
                f"{p['impl'].cost if p['impl'] else '-'}/call)",
                f"  Escalate when confidence < {p['tau']:.2f}",
                f"  Stage 2: llm  (cost {LLM_COST}/call, "
                f"dev-estimated accuracy {self.llm_est_acc:.3f})",
                "",
                "Expected (dev-split estimates, statistical not guaranteed):",
                f"  accuracy   {p['est_acc']:.3f}",
                f"  escalation {p['frac_llm']:.1%}",
                f"  cost/call  {p['exp_cost']:.2f}",
                "",
                f"Replay: traces keyed by impl_version="
                f"{self.impl_version}"]
        if effects:
            out += ["", "Materialization / external effects:"]
            for name, e in effects.items():
                out.append(f"  {name}: requires ground("
                           f"threshold>={e['threshold']}, "
                           f"validator={e.get('validator', 'label set')}) "
                           f"else fallback -> {e['fallback']}")
        return "\n".join(out)
