# Prajna: an AI-native programming language

**What if neural inference were a language primitive — but a typed,
costed, verifiable one that the compiler tries to eliminate?**

This repo is a runnable semantics prototype (an embedded Python DSL) of
an AI-native language design. It is deliberately small. The point is not
the implementation; the point is the six mechanisms, which together make
programs that use LLMs **fast** (the compiler replaces the LLM with
cheaper implementations wherever a contract allows), **accurate**
(neural outputs cannot enter deterministic code without passing a
verification gate), and **debuggable** (every run is bit-reproducible).

New: a three-way benchmark against a hand-tuned FrugalGPT-style cascade, with an inference planner and a SQL-style explain(), lives in bench/

## Run it

```bash
pip install -r requirements.txt   # numpy, scikit-learn
python3 demo.py
```

No API keys, no network. The "frontier LLM" is an injectable offline
oracle so the demo is deterministic; swap in a real API client for
production.

## The problem

Today we bolt LLMs onto languages via API calls. The results are slow
(every semantic operation pays frontier-model latency), unreliable
(hallucinations flow unchecked into deterministic code), and
undebuggable (no two runs are alike). Languages still embody 1970s
assumptions: computation is exact, values are certain, functions are
written rather than learned.

## The design: six mechanisms

**1. Belief types.** Soft values are `Belief(value, confidence, source,
cost)`. Confidence and provenance are carried by the type system, not by
fields you remember to check.

**2. Grounding.** `ground(belief, by=validator, threshold=…, fallback=…)`
is the *only* way a soft value becomes a crisp one. It returns a plain
value or fails deterministically. Hallucination becomes a visible,
catchable error class — the way Rust made null a type error.

**3. Contract-driven model lowering.** A semantic function is declared
by examples plus a contract:

```python
classify = sem("classify_intent", examples=[...], llm=oracle,
               accuracy=0.85, min_confidence=0.5)
```

"Compilation" walks a ladder — rule synthesis → distilled statistical
head → LLM — and links the *cheapest* implementation whose held-out
accuracy meets the contract. The programmer states intent; the compiler
chooses the model.

**4. Deterministic replay.** Every soft call is memoised against
(function, implementation-version, input). Identical runs are
bit-identical and free. The trace store doubles as the distillation
dataset.

**5. SDM semantic cache.** A sparse distributed memory sits in front of
the model. Inputs are encoded as rank-order N-of-M codes; writes
superimpose onto counters at activated random hard locations; reads take
a confidence-weighted vote. *Paraphrases* of previously answered inputs
are served in microseconds, on CPU, with no model call — unlike exact
memoisation, and plausibly cheaper than embedding-cosine caches, unbenchmarked, see open problem 3.

**6. Hot-path distillation.** `fn.relower()` folds high-confidence
traces back into the example set and re-runs the lowering ladder. A
function born as an LLM call dies as a tiny learned head.

## What the demo shows (actual output, abridged)

```
[lower] classify_intent: trying rules -> LOO acc 0.89 (contract 0.85)
[lower] classify_intent: LINKED rules (cost 0.01/call)

'hi, you charged my card two times!!' -> Belief('billing', conf=0.95, source=rules, cost=0.01)
GroundingError caught deterministically: cannot ground Belief('giraffe', ...)
Belief('billing', conf=0.95, source=rules+replay, cost=0)
'my card was charged two times hello' -> Belief('billing', conf=1.00, source=sdm_cache, cost=0.001)

[lower] detect_sarcasm: no cheap level met contract; LINKED llm (cost 10.0/call)
  ... 16 traces collected ...
[relower] detect_sarcasm: trying distilled_head -> LOO acc 0.90
[lower] detect_sarcasm: LINKED distilled_head (cost 0.1/call)
'oh marvellous, it died again' -> Belief('sarcastic', conf=0.78, source=distilled_head, cost=0.1)
llm calls total: 16 (unchanged => distillation held)
```

One function never needed the LLM at all; the other needed it for
exactly 16 calls before being compiled down to a 100x-cheaper head.

## Open problems (take one and build it)

See ROADMAP.md for a concrete build order

1. **A real type system.** Here, grounding is a runtime check. The
   research version is static: soft-ness as a *typed effect* (like IO in
   Haskell), with `ground` as its elimination form, checked at compile
   time. Nobody has published this.
2. **Generative sem-functions.** The lowering ladder here is
   classification-shaped. Generation needs constrained decoding
   (LMQL/Guidance-style) at the LLM level and seq2seq distillation
   below it, with contracts stated as property tests rather than
   accuracy.
3. **SDM cache at scale.** Benchmark rank-order SDM caching against
   embedding-cosine caches (GPTCache) on paraphrase datasets (PAWS,
   QQP): hit rate vs. false-hit rate vs. CPU latency, plus write/decay
   policies and caching of output *vectors* for generative calls. This
   is a self-contained systems paper.
4. **Cost-aware whole-program optimisation.** Lowering here is
   per-function. A real compiler should optimise the ensemble: share
   heads across functions, batch escalations, trade contract slack
   between hot and cold paths under a global cost budget.
5. **Formal semantics.** Confidence propagation through composition of
   sem-functions is undefined here (each call is independent). What is
   the right algebra — probabilistic, Dempster-Shafer, interval?
6. **Contract validity.** Leave-one-out on curated examples overstates
   real accuracy; distillation traces are model-labelled, so the head
   learns the LLM's errors. Real contracts need held-out human-audited
   sets and drift monitoring — which the trace store makes cheap.

## Prior art

DSPy (declarative LLM programs + optimizers ≈ mechanism 3 without the
type discipline) · LMQL, Guidance (constrained decoding) · Instructor,
Marvin (schema grounding) · Pyro, Church (probabilistic semantics) ·
GPTCache (semantic caching, embedding-based) · Kanerva's SDM and
rank-order N-of-M codes (Furber et al., IEEE TNN 2007). The combination
— graded types + contract lowering + deterministic replay +
associative-memory caching — has, to our knowledge, not been unified in
one language design.

## Writing

https://medium.com/@joyboseroy/programming-languages-assume-computers-are-deterministic-ai-doesnt-661b2029b04c

https://joyboseroy.medium.com/where-an-ai-native-programming-model-beats-python-and-where-it-should-not-try-b5c2f06b0205

## Files

- `prajna.py` — the DSL: Belief, ground, RankOrderEncoder, SDMCache,
  the lowering ladder, TraceStore, SemFunction.
- `demo.py` — all six mechanisms end to end.

MIT licensed. Issues and forks welcome — especially against the open
problems above.
