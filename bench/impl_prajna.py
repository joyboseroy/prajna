"""Implementation 3: PRAJNA. Application code only — planning, tuning,
escalation, replay and grounding live in the language runtime."""

from prajna import GroundingError, TraceStore, ground
from prajna_ext import PlannedSemFunction
from common import (ACTION_THRESHOLD, CONSEQUENTIAL, HEAD_LATENCY_MS,
                    LLM_LATENCY_MS, RULE_LATENCY_MS)

LAT = {"rules": RULE_LATENCY_MS, "distilled_head": HEAD_LATENCY_MS,
       "tfidf_head": HEAD_LATENCY_MS, "llm": LLM_LATENCY_MS}


def latency(source):
    base = LAT.get(source.split("->")[0].replace("+replay", ""), 0.0)
    return base + (LLM_LATENCY_MS if "escalated" in source else 0.0)


def run(test, llm, fit, dev, target_acc=0.92):
    classify = PlannedSemFunction("intent", fit, dev, target_acc, llm,
                                  TraceStore())
    preds, lat, cost, actions, deferred = [], [], 0.0, [], 0
    for text, true in test:
        b = classify(text, true)
        preds.append(b.value)
        cost += b.cost
        lat.append(latency(b.source))
        if b.value in CONSEQUENTIAL:
            # the ONLY soft->effect gate: validated, thresholded, with a
            # deterministic fallback. Unguarded flow is unrepresentable.
            decision = ground(b, by=lambda v: v in CONSEQUENTIAL,
                              threshold=ACTION_THRESHOLD,
                              fallback=lambda: "human_review")
            if decision == "human_review":
                deferred += 1
            else:
                actions.append(("block_card", text, true))
    return dict(preds=preds, latencies=lat,
                cost=cost + classify.plan_compile_cost,
                actions=actions, deferred=deferred, unguarded_flows=0,
                fn=classify)
