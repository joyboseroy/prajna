from common import SimulatedLLM, load_data, ACTION_THRESHOLD
from prajna import TraceStore
from prajna_ext import PlannedSemFunction

fit, dev, test, labels = load_data()
fn = PlannedSemFunction("classify_intent", fit, dev, 0.92,
                        SimulatedLLM(labels), TraceStore(), verbose=False)
print(fn.explain(effects={"BLOCK_CARD": {
    "threshold": ACTION_THRESHOLD,
    "validator": "label in CONSEQUENTIAL",
    "fallback": "HUMAN_REVIEW"}}))
