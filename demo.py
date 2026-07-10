"""Demo: the Prajna prototype end to end.

Scenario: customer-message intent routing (the canonical example all
three design docs used). Shows:

  1. contract-driven lowering at 'compile' time (links a cheap level)
  2. grounding (soft -> crisp gate) with a deterministic fallback
  3. deterministic replay
  4. SDM semantic cache answering paraphrases without any model call
  5. a hard case escalating to the LLM, then hot-path distillation
     re-lowering the function so the LLM is no longer needed
"""

from prajna import (Belief, GroundingError, LLMOracle, TraceStore,
                    ground, sem)

# ---------------------------------------------------------------- setup
# The "frontier model": injectable oracle so the demo is offline +
# deterministic. Pretend each call costs 10 units of latency/money.
def big_model(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["refund", "money back", "charg", "cancel",
                            "unsubscribe", "reimburse"]):
        return "billing"
    if any(w in t for w in ["crash", "error", "bug", "broken", "fails",
                            "freez", "login", "password"]):
        return "tech_support"
    return "general"

llm = LLMOracle(big_model)
traces = TraceStore()

print("=" * 72)
print("STEP 1 - declare a semantic function by examples + contract")
print("=" * 72)

classify = sem(
    "classify_intent",
    examples=[
        ("I want a refund for my last order", "billing"),
        ("please refund me and cancel my subscription", "billing"),
        ("requesting a refund, I was charged twice", "billing"),
        ("you charged me twice this month", "billing"),
        ("my card was charged incorrectly", "billing"),
        ("cancel my subscription please", "billing"),
        ("the app crashes on startup", "tech_support"),
        ("app crashes whenever I open settings", "tech_support"),
        ("getting an error when I log in", "tech_support"),
        ("login error after the update", "tech_support"),
        ("the export feature is broken", "tech_support"),
        ("search is broken and shows an error", "tech_support"),
        ("what are your opening hours", "general"),
        ("what hours are you open on weekends", "general"),
        ("do you ship to India", "general"),
        ("how long does shipping take", "general"),
        ("quick question about your company", "general"),
        ("question about your office location", "general"),
    ],
    llm=llm,
    accuracy=0.85,          # the contract
    min_confidence=0.50,    # runtime escalation threshold
    trace_store=traces,
)

print()
print("=" * 72)
print("STEP 2 - run it: soft values, then ground() into crisp routing")
print("=" * 72)

ROUTES = {"billing", "tech_support", "general"}

def route(msg: str) -> str:
    b = classify(msg)
    print(f"  {msg!r:55s} -> {b}")
    # soft -> crisp gate: schema validator + confidence threshold,
    # deterministic fallback instead of a hallucinated route
    return ground(b, by=lambda v: v in ROUTES, threshold=0.5,
                  fallback=lambda: "human_review")

for m in ["hi, you charged my card two times!!",
          "app keeps freezing and crashing after the update",
          "hello! quick question about shipping",
          ]:
    print(f"  routed to: {route(m)}\n")

print("grounding failure path (validator rejects an out-of-schema value):")
try:
    ground(Belief("giraffe", 0.99, "demo"), by=lambda v: v in ROUTES)
except GroundingError as e:
    print(f"  GroundingError caught deterministically: {e}\n")

print("=" * 72)
print("STEP 3 - deterministic replay: same input, zero cost, same bits")
print("=" * 72)
b = classify("hi, you charged my card two times!!")
print(f"  {b}   <- source shows +replay, cost 0\n")

print("=" * 72)
print("STEP 4 - SDM semantic cache: paraphrase answered without a model")
print("=" * 72)
print("  (rank-order N-of-M codes over hashed tokens; near-match")
print("   activation of hard locations; confidence-weighted vote)")
for m in ["you charged my card twice!",            # near-dup of a seen input
          "my card was charged two times hello",   # paraphrase
          "the app keeps freezing and crashing"]:  # near-dup of a seen input
    b = classify(m)
    print(f"  {m!r:45s} -> {b}")
print(f"  cache stats: {classify.cache.hits} hits / "
      f"{classify.cache.misses} misses\n")

print("=" * 72)
print("STEP 5 - a genuinely hard function lowers all the way to the LLM,")
print("          then distils back down after collecting traces")
print("=" * 72)

# Deliberately inadequate examples: rules/head can't meet the contract,
# so the ladder links the frontier model. SDM cache is off here so every
# call produces a trace (we want the distillation dataset to grow fast).
hard = sem(
    "detect_sarcasm",
    examples=[
        ("oh great, another outage", "sarcastic"),
        ("great, thanks for the fast reply", "sincere"),
        ("wow, amazing, it broke again", "sarcastic"),
        ("wow, amazing work team", "sincere"),
    ],
    llm=LLMOracle(lambda t: "sarcastic" if ("again" in t or "another" in t
                                            or "sure it will" in t
                                            or t.startswith("oh "))
                  else "sincere", confidence=0.96),
    accuracy=0.85,
    trace_store=traces,
    use_sdm_cache=False,
)

print()
print("  running 16 inputs through the linked LLM (cost 10 each):")
inputs = [
    "oh perfect, it failed again", "thanks, that fixed it",
    "another brilliant update that broke everything",
    "really appreciate the quick help",
    "sure it will work this time, sure",
    "this new feature is genuinely great, thanks",
    "oh great, crashed again", "thank you so much, works now",
    "oh wonderful, another error", "the fix works, thanks a lot",
    "oh fantastic, another day another outage",
    "appreciate the support, all good now",
    "oh lovely, it broke again after the patch",
    "thanks team, the update works",
    "oh brilliant, another regression again",
    "much appreciated, everything works",
]
for t in inputs:
    hard(t)
print(f"  llm calls so far: {hard.llm.calls}")

print("\n  hot-path distillation: fold traces back in, re-run the ladder:")
changed = hard.relower()
print(f"  implementation swapped: {changed} "
      f"(now linked: {hard.impl.name}, cost {hard.impl.cost}/call)")

print("\n  new inputs after re-lowering (note source and cost):")
for t in ["oh marvellous, it died again", "thanks a lot, works perfectly"]:
    print(f"  {t!r:45s} -> {hard(t)}")
print(f"  llm calls total: {hard.llm.calls} (unchanged => distillation held)")

print()
print("=" * 72)
print("Total abstract cost ledger")
print("=" * 72)
print(f"  frontier-model calls across the whole program: "
      f"{llm.calls + hard.llm.calls}")
print("  everything else ran on rules, a distilled head, the SDM cache,")
print("  or deterministic replay.")
