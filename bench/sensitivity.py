import numpy as np
from common import SimulatedLLM, load_data
from prajna import TraceStore
from prajna_ext import PlannedSemFunction

fit, dev, test, labels = load_data()
for target in (0.85, 0.92, 0.95):
    llm = SimulatedLLM(labels)
    fn = PlannedSemFunction("intent", fit, dev, target, llm,
                            TraceStore(), verbose=False)
    preds = [fn(t, l).value for t, l in test]
    acc = float(np.mean([p == l for p, (_, l) in zip(preds, test)]))
    p = fn.plan
    print(f"contract={target:.2f} level={p['level']} tau={p['tau']:.2f} "
          f"test_acc={acc:.3f} llm_frac={llm.calls/len(test):.2f} "
          f"exp_cost={p['exp_cost']:.2f}")
