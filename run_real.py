"""Real-model benchmark run. Same three systems, same metrics, real LLM.

  export GROQ_API_KEY=...
  python3 run_real.py                 # full 3,080-item test set
  N_TEST=750 python3 run_real.py      # subsampled (rate-limit friendly)

Reuses the metric functions from run_benchmark. The GroqLLM disk cache
makes reruns free and reproducible.
"""
import os
import random

import impl_cascade
import impl_naive
import impl_prajna
from common import load_data
from groq_llm import GroqLLM
from run_benchmark import evaluate, loc

TARGET = float(os.environ.get("CONTRACT", "0.85"))   # see README note


def main():
    fit, dev, test, labels = load_data()
    n = int(os.environ.get("N_TEST", len(test)))
    if n < len(test):
        test = random.Random(7).sample(test, n)
    print(f"REAL-MODEL RUN  test={len(test)}  contract={TARGET}\n")
    rows = []
    for name, fn, kw in [
            ("naive_llm", impl_naive.run, {}),
            ("hand_cascade", impl_cascade.run,
             dict(fit=fit, dev=dev, target_acc=TARGET)),
            ("prajna", impl_prajna.run,
             dict(fit=fit, dev=dev, target_acc=TARGET))]:
        llm = GroqLLM(labels)
        before = llm.calls
        r = fn(test, llm, **kw)
        m = evaluate(name, r, test)
        m["llm_calls"] = llm.calls - before
        m["loc"] = loc(f"impl_{name.split('_')[0] if name != 'hand_cascade' else 'cascade'}.py")
        llm._save()
        rows.append((m, r))
        print(f"{name}: acc={m['acc']:.3f} billable_llm_calls={m['llm_calls']} "
              f"cost={m['cost']:.0f} wrong_blocks={m['actions_wrong']} "
              f"deferred={m['deferred']}")
    fn_obj = rows[-1][1].get("fn")
    if fn_obj is not None:
        print("\n" + fn_obj.explain())


if __name__ == "__main__":
    main()
