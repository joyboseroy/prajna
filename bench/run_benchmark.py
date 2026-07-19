"""Run the full benchmark: three implementations, one metrics table."""

import hashlib

import numpy as np

import impl_cascade
import impl_naive
import impl_prajna
from common import CONSEQUENTIAL, SimulatedLLM, load_data


def loc(path):
    n = 0
    for line in open(path):
        s = line.strip()
        if s and not s.startswith("#") and not s.startswith('"""') \
                and not s.startswith("'''"):
            n += 1
    return n


def action_outcomes(res, name):
    right = sum(1 for _, _, true in res["actions"] if true in CONSEQUENTIAL)
    wrong = len(res["actions"]) - right
    return right, wrong, res["deferred"]


def evaluate(name, res, test):
    y = [l for _, l in test]
    acc = float(np.mean([p == t for p, t in zip(res["preds"], y)]))
    lat = np.array(res["latencies"])
    right, wrong, deferred = action_outcomes(res, name)
    return dict(name=name, acc=acc, cost=res["cost"],
                p50=float(np.percentile(lat, 50)),
                p95=float(np.percentile(lat, 95)),
                actions_right=right, actions_wrong=wrong,
                deferred=deferred, unguarded=res["unguarded_flows"])


def main():
    fit, dev, test, labels = load_data()
    print(f"Banking77: fit={len(fit)} dev={len(dev)} test={len(test)} "
          f"labels={len(labels)}\n")

    rows = []

    llm = SimulatedLLM(labels)
    r = impl_naive.run(test, llm)
    m = evaluate("naive_llm", r, test); m["llm_calls"] = llm.calls
    m["loc"] = loc("impl_naive.py"); rows.append(m)

    llm = SimulatedLLM(labels)
    r = impl_cascade.run(test, llm, fit, dev)
    m = evaluate("hand_cascade", r, test); m["llm_calls"] = llm.calls
    m["loc"] = loc("impl_cascade.py"); m["tau"] = r["tau"]; rows.append(m)

    llm = SimulatedLLM(labels)
    r = impl_prajna.run(test, llm, fit, dev)
    m = evaluate("prajna", r, test); m["llm_calls"] = llm.calls
    m["loc"] = loc("impl_prajna.py"); rows.append(m)

    # deterministic replay check: rerun prajna with the same trace store
    fn = r["fn"]
    h1 = hashlib.md5("".join(r["preds"]).encode()).hexdigest()
    preds2 = [fn(t, l).value for t, l in test]
    h2 = hashlib.md5("".join(preds2).encode()).hexdigest()
    extra_llm = llm.calls - m["llm_calls"]
    replay = "bit-identical, 0 extra LLM calls" if (
        h1 == h2 and extra_llm == 0) else "FAILED"

    hdr = (f"{'system':<14}{'acc':>7}{'LLM calls':>11}{'cost':>10}"
           f"{'p50 ms':>8}{'p95 ms':>8}{'LOC':>6}"
           f"{'act ok/wrong/defer':>20}{'unguarded':>11}")
    print(hdr); print("-" * len(hdr))
    for m in rows:
        print(f"{m['name']:<14}{m['acc']:>7.3f}{m['llm_calls']:>11}"
              f"{m['cost']:>10.0f}{m['p50']:>8.1f}{m['p95']:>8.1f}"
              f"{m['loc']:>6}"
              f"{str(m['actions_right'])+'/'+str(m['actions_wrong'])+'/'+str(m['deferred']):>20}"
              f"{m['unguarded']:>11}")
    print(f"\nprajna replay: {replay}")
    print(f"cascade hand-tuned tau={rows[1].get('tau'):.2f} vs "
          f"prajna planned tau={fn.plan['tau']:.2f} "
          f"(level={fn.plan['level']})")


if __name__ == "__main__":
    main()
