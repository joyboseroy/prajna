# Roadmap

Prajna is a semantics prototype. This roadmap orders what comes next by
effort-to-payoff, so anyone (including future me) can pick an item and
build it. Issues welcome against any of these.

## v0.2 — Make it real (low-hanging fruit, hours each)

- [ ] **Real LLM backend.** An `LLMOracle` subclass that calls an actual
  API (Anthropic / OpenAI / Groq) behind an env-var key, with the
  offline oracle kept as the default so the demo stays deterministic.
  ~40 lines; turns the prototype into something usable.
- [ ] **Embedding-based distilled head.** Add a lowering level between
  the hashing head and the LLM: frozen sentence-transformer
  (all-MiniLM-L6-v2) features + logistic head. Makes hot-path
  distillation credible on real paraphrase variation.
- [ ] **Persist compiled implementations.** Cache the linked
  implementation (joblib) keyed by the example-set hash, so
  "compilation" survives process restarts instead of retraining on
  every import. This is what makes `sem()` feel like a compiler.
- [ ] **Second demo: extraction + grounding.** A sem-function that
  extracts dates/amounts from text and grounds into a dataclass via
  regex/schema validators. Shows `ground()` doing more than checking
  label membership.
- [ ] **Program-level cost ledger.** `prajna.report()`: per-function
  table of calls by source (rules / head / cache / replay / llm),
  total cost, cache hit rate. The demo's closing ledger, as an API.
- [ ] **Tests + CI.** pytest for encoder similarity properties, SDM
  store/recall, grounding paths, lowering determinism; the stock
  GitHub Actions Python workflow; badge in README.
- [ ] **Packaging.** `pyproject.toml`, `pip install` from source; PyPI
  only after checking the name (prajna may be taken — `prajna-lang`).
- [ ] **Citability.** CITATION.cff + Zenodo DOI via the GitHub
  integration.

## v0.3 — Semantics that don't exist yet elsewhere

- [ ] **Confidence algebra.** Define composition: what is the confidence
  of `f(g(x))`? Start with product and min as selectable policies;
  document why neither is right (open problem 5) and what evidence
  would settle it.
- [ ] **Generative sem-functions.** JSON-schema-constrained output at
  the LLM level with a validate-and-retry loop; contracts as property
  tests (e.g. "output parses AND field X in range") instead of
  accuracy; seq2seq distillation deferred.
- [ ] **Belief-aware control flow.** Sugar for confidence-dispatched
  branching (`match belief: case v, c if c > 0.9: ...`), so the
  escalate-vs-accept pattern in the demo becomes a language construct.
- [ ] **Shadow audits / drift monitoring.** Randomly escalate a small
  fraction of cheap-level calls to the LLM and compare, giving a live
  estimate of whether the linked implementation still meets its
  contract (open problem 6). Re-lower or re-escalate automatically on
  sustained violation.

## v0.4 — The two papers

- [ ] **SDM cache benchmark.** Rank-order SDM vs embedding-cosine
  (GPTCache-style) semantic caching on QQP/PAWS: hit rate vs
  false-hit rate curves, CPU latency, memory footprint, one-shot
  write cost. Self-contained systems paper; the benchmark script
  lives in this repo either way.
- [ ] **Static soft-ness checking.** Prototype "uncertainty as a typed
  effect" using Python type annotations + a mypy plugin: `Belief[T]`
  cannot flow where `T` is expected without `ground()`. A real
  static-semantics paper would follow from this, but the plugin alone
  is a useful artifact.

## v0.5+ — Aspirational

- [ ] Whole-program cost optimisation: share heads across functions,
  batch escalations, trade contract slack under a global budget.
- [ ] Output-vector SDM caching for generative calls, with write/decay
  policies.
- [ ] An actual surface syntax (a small parser over the DSL), only if
  the semantics stabilise. Syntax last.

## Non-goals

- Competing with DSPy/LangChain on breadth. Prajna is a language-design
  argument, not a framework.
- Wrapping every model provider. One real backend is enough to make the
  point.
- Performance engineering of the SDM before the benchmark says it is
  worth it.
