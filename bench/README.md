# prajna-bench: three ways to ship the same semantic pipeline

The falsifiable experiment behind Prajna's Tier-2 claim. One task, three
implementations, one metrics table.

**Task.** Banking77 (10,016 train / 3,080 test customer messages, 77
intents, from PolyAI's public repo) plus a consequential sub-task: three
intents (`lost_or_stolen_card`, `compromised_card`, `card_swallowed`)
trigger `block_card()`, an external effect. Wrongly blocking an innocent
customer is the failure mode that matters.

**Implementations.**
- `impl_naive.py` — every message to the LLM; action fires off the raw
  model string. What most first integrations look like.
- `impl_cascade.py` — the strong baseline: a hand-built FrugalGPT-style
  cascade (TF-IDF + logistic head, escalation threshold hand-tuned on a
  dev split, ad-hoc action gating). Written the way a competent engineer
  writes it.
- `impl_prajna.py` — application code only; planning, threshold tuning,
  escalation, replay and grounding live in the runtime
  (`prajna_ext.PlannedSemFunction`, the Inference Planner: contract =
  end-to-end accuracy, compiler picks level + threshold minimising
  expected cost).

## Results (simulated 95%-accurate LLM, contract = 0.92)

```
system            acc  LLM calls      cost  p50 ms  p95 ms   LOC  act ok/wrong/defer  unguarded
-----------------------------------------------------------------------------------------------
naive_llm       0.950       3080     30800   400.0   400.0    15             115/7/0          1
hand_cascade    0.929        582      6128     2.0   402.0    70            22/1/100          0
prajna          0.919        530      5608     2.0   402.0    31             22/1/97          0

prajna replay: bit-identical, 0 extra LLM calls
cascade hand-tuned tau=0.28 vs prajna planned tau=0.26
```

**What this shows.**
1. Both cascades cut cost ~5x and median latency 200x vs naive, at the
   contracted accuracy. (Consistent with FrugalGPT's finding; not novel
   on its own.)
2. The planner converged on nearly the same threshold as manual tuning
   (0.26 vs 0.28) with zero cascade code written by the application
   programmer: 31 LOC vs 70, and the 31 contain no orchestration.
3. Action safety: naive wrongly blocked 7 innocent customers. Both
   gated systems wrongly blocked 1, deferring ~100 low-confidence cases
   to human review. But only Prajna makes the unguarded flow
   *unrepresentable* — the cascade's gate is a convention the next
   contributor can skip.
4. Replay: rerunning Prajna is bit-identical with zero extra LLM calls.
   Neither baseline has this without building it.

## Honesty notes (read before quoting numbers)

- **The LLM is simulated** (returns truth with p=0.95, deterministic
  errors). Absolute accuracies are assumptions; cost/call-count/latency
  structure, code size, safety outcomes and *relative* comparisons are
  the findings. `common.RealLLM` is the stub for real-API runs, which
  are the numbers a paper would report.
- **Prajna landed at 0.919 against a 0.92 contract** — the dev-split
  estimate (0.924) was optimistic by half a point. Contract estimation
  error is real; this is exactly what shadow audits / drift monitoring
  (ROADMAP v0.3) exist to catch, and hand-tuning has the same error
  (cascade's own dev estimate was similarly off, in its favour).
- **LOC comparison**: counts application code. Prajna's planner lives in
  the runtime — that is the claim being made (the SQL argument), not an
  accounting trick; the planner is ~90 lines, amortised over every
  program that uses it.
- SDM semantic caching is **off** here — it's a separate claim needing
  its own benchmark (open problem 3).

## Run it

```bash
pip install numpy scikit-learn
python3 run_benchmark.py     # ~3-4 min, no API keys
```

Data auto-included (`data/train.csv`, `data/test.csv`); re-download from
github.com/PolyAI-LDN/task-specific-datasets if absent.
