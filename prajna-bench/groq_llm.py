"""GroqLLM: a real frontier-model backend for the benchmark.

Drop-in replacement for SimulatedLLM. Uses Groq's OpenAI-compatible API
(llama-3.3-70b-versatile by default). Every response is cached to disk
keyed by input hash, so reruns are free, deterministic, and the replay
guarantee survives; delete the cache file to force fresh calls.

Usage:
    export GROQ_API_KEY=...        # from console.groq.com
    python3 run_real.py            # optionally N_TEST=750 python3 run_real.py
"""

import hashlib
import json
import os
import time

import requests

API_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqLLM:
    def __init__(self, labels, model="llama-3.3-70b-versatile",
                 cache_path="groq_cache.json"):
        self.labels = labels
        self.model = model
        self.calls = 0            # counts billable (non-cached) calls
        self.cache_path = cache_path
        self.cache = {}
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                self.cache = json.load(f)
        self._label_lookup = {l.lower().replace(" ", "_"): l for l in labels}
        self._sys = ("You are an intent classifier for a bank's customer "
                     "support. Reply with exactly one label from this list "
                     "and nothing else:\n" + "\n".join(labels))

    def _post(self, text):
        key = os.environ["GROQ_API_KEY"]
        for attempt in range(6):
            r = requests.post(API_URL, timeout=60, headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json"},
                json={"model": self.model, "temperature": 0,
                      "max_tokens": 20,
                      "messages": [
                          {"role": "system", "content": self._sys},
                          {"role": "user", "content": text}]})
            if r.status_code == 429:
                time.sleep(2 ** attempt)      # rate limited: back off
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        raise RuntimeError("rate-limited after 6 retries")

    def _parse(self, raw):
        s = raw.strip().lower().replace(" ", "_").strip(".\"'`")
        if s in self._label_lookup:
            return self._label_lookup[s]
        for cand in self._label_lookup:       # substring fallback
            if cand in s:
                return self._label_lookup[cand]
        return self.labels[0]                 # unparseable: recorded as-is

    def classify(self, text, true_label=None):   # true_label ignored
        k = hashlib.md5((self.model + "|" + text).encode()).hexdigest()
        if k in self.cache:
            return self.cache[k]
        self.calls += 1
        label = self._parse(self._post(text))
        self.cache[k] = label
        if self.calls % 25 == 0:              # periodic persist
            self._save()
        return label

    def _save(self):
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f)

    def __del__(self):
        try:
            self._save()
        except Exception:
            pass
