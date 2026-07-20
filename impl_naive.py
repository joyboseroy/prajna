"""Implementation 1: NAIVE. Every message goes to the LLM; the action
fires directly off the model's string output. This is what most first
LLM integrations actually look like."""

from common import (ACTION_THRESHOLD, CONSEQUENTIAL, LLM_COST,
                    LLM_LATENCY_MS)


def run(test, llm):
    preds, lat, cost, actions = [], [], 0.0, []
    for text, true in test:
        label = llm.classify(text, true)
        preds.append(label)
        lat.append(LLM_LATENCY_MS)
        cost += LLM_COST
        # UNGUARDED belief->effect flow: no confidence available at all,
        # the raw model string decides an external effect.
        if label in CONSEQUENTIAL:
            actions.append(("block_card", text, true))
    return dict(preds=preds, latencies=lat, cost=cost, actions=actions,
                deferred=0, unguarded_flows=1)
