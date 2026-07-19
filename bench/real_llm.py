"""Real-model backend for the Prajna benchmark (Groq, Llama 3.3 70B).

Drop into bench/ alongside common.py. Usage:

    export GROQ_API_KEY=...
    python3 real_llm.py                 # cascade + prajna (~900 calls)
    python3 real_llm.py --with-naive    # also naive (~3,000 more calls)

Design notes:
- Same interface as SimulatedLLM.classify(text, true_label), but the
  true label is IGNORED: the real run does not peek.
- Every response is cached to disk (llm_cache.json) keyed by a hash of
  (model, text). Interrupt and rerun freely; you pay for each unique
  message once, and cached reruns are deterministic even though
  temperature-0 API sampling is not strictly guaranteed to be.
- Parse strategy: exact label match, else unique substring match, else
  one retry with a stricter prompt, else nearest label by token overlap
  (counted as a parse failure in stats, never a crash).
- 429/5xx handled with exponential backoff. Groq free-tier rate limits
  are low; the pause between calls is configurable.
"""

import argparse
import hashlib
import json
import os
import re
import time

import requests

from common import load_data

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"
CACHE_PATH = "llm_cache.json"


class RealLLM:
    def __init__(self, labels, model=MODEL, cache_path=CACHE_PATH,
                 pause_s=0.5, api_key=None):
        self.labels = list(labels)
        self.label_set = set(self.labels)
        self.model = model
        self.pause_s = pause_s
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("set GROQ_API_KEY")
        self.calls = 0            # counts non-cached API round trips
        self.served = 0           # counts classify() invocations
        self.parse_failures = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cache_path = cache_path
        self.cache = {}
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                self.cache = json.load(f)
        self._sys = (
            "You are an intent classifier for banking customer messages. "
            "Reply with exactly one label from this list and nothing "
            "else:\n" + "\n".join(self.labels))

    # ---- transport -----------------------------------------------------
    def _post(self, messages, max_tokens=20):
        for attempt in range(6):
            r = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                json={"model": self.model, "messages": messages,
                      "temperature": 0, "max_tokens": max_tokens},
                timeout=60)
            if r.status_code == 200:
                d = r.json()
                u = d.get("usage", {})
                self.prompt_tokens += u.get("prompt_tokens", 0)
                self.completion_tokens += u.get("completion_tokens", 0)
                return d["choices"][0]["message"]["content"]
            if r.status_code in (429, 500, 502, 503):
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Groq HTTP {r.status_code}: {r.text[:200]}")
        raise RuntimeError("Groq: exhausted retries (rate limit?)")

    # ---- parsing -------------------------------------------------------
    def _parse(self, raw):
        t = raw.strip().strip(".\"'` ").lower().replace(" ", "_")
        if t in self.label_set:
            return t
        hits = [l for l in self.labels if l in t]
        if len(hits) == 1:
            return hits[0]
        return None

    def _nearest(self, raw):
        toks = set(re.findall(r"[a-z]+", raw.lower()))
        return max(self.labels,
                   key=lambda l: len(toks & set(l.split("_"))))

    # ---- public --------------------------------------------------------
    def classify(self, text, true_label=None):   # true_label ignored
        self.served += 1
        k = hashlib.md5(f"{self.model}|{text}".encode()).hexdigest()
        if k in self.cache:
            return self.cache[k]
        msgs = [{"role": "system", "content": self._sys},
                {"role": "user", "content": text}]
        self.calls += 1
        label = self._parse(self._post(msgs))
        if label is None:                        # one strict retry
            msgs.append({"role": "user", "content":
                         "Reply with exactly one label from the list, "
                         "with no other words."})
            self.calls += 1
            raw = self._post(msgs)
            label = self._parse(raw)
            if label is None:
                self.parse_failures += 1
                label = self._nearest(raw)
        self.cache[k] = label
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f)
        time.sleep(self.pause_s)
        return label

    def stats(self):
        return (f"api_calls={self.calls} served={self.served} "
                f"parse_failures={self.parse_failures} "
                f"tokens={self.prompt_tokens}+{self.completion_tokens}")


def main():
    import numpy as np
    import impl_cascade
    import impl_prajna
    import impl_naive

    ap = argparse.ArgumentParser()
    ap.add_argument("--with-naive", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="run on first N test items (smoke test)")
    args = ap.parse_args()

    fit, dev, test, labels = load_data()
    if args.limit:
        test = test[:args.limit]
    y = [l for _, l in test]

    from common import CONSEQUENTIAL

    def report(name, res, llm):
        acc = float(np.mean([p == t for p, t in zip(res["preds"], y)]))
        wrong = sum(1 for _, _, t in res["actions"] if t not in CONSEQUENTIAL)
        print(f"{name:<14} acc={acc:.3f} llm_served={llm.served} "
              f"wrong_blocks={wrong} deferred={res['deferred']}  "
              f"[{llm.stats()}]")

    llm = RealLLM(labels)
    report("hand_cascade", impl_cascade.run(test, llm, fit, dev), llm)

    llm = RealLLM(labels)
    report("prajna", impl_prajna.run(test, llm, fit, dev), llm)

    if args.with_naive:
        llm = RealLLM(labels)
        report("naive_llm", impl_naive.run(test, llm), llm)


if __name__ == "__main__":
    main()
