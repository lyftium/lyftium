# Pierre

Paris · FR / EN · [Kaggle](https://www.kaggle.com/lyftium) · [notes](https://lyftium.github.io)

I like problems where the wrong metric looks convincing. Most of what I publish here is about checking a claim before you ship it — then wrapping the useful bits so someone else can run them without me in the room.

---

**If you only open two repos:** [off-policy-bandit-evaluation](https://github.com/lyftium/off-policy-bandit-evaluation) (did the estimator actually recover the truth?) and [drawing-review-service](https://github.com/lyftium/drawing-review-service) (can another engineer call it, score it, and prepare a cluster?).

---

### Off-policy evaluation

[off-policy-bandit-evaluation](https://github.com/lyftium/off-policy-bandit-evaluation) · [DEMO](https://github.com/lyftium/off-policy-bandit-evaluation/blob/main/docs/DEMO.md)

I wanted to know whether you can rank a new recommender from someone else’s logs. On 1.37 M Open Bandit rounds, IPW / SNIPW / DR get within **5.3–6.1 %** of the production CTR and their CIs cover it; Direct Method’s CI is tiny and wrong. A simple popularity policy looks ~**2.1×** on a time split — interesting enough for an A/B, not something I’d ship from the estimate alone. Numbers live in [`results.json`](https://github.com/lyftium/off-policy-bandit-evaluation/blob/main/results.json).

### Drawing review service

[drawing-review-service](https://github.com/lyftium/drawing-review-service) · [DEMO](https://github.com/lyftium/drawing-review-service/blob/main/docs/DEMO.md)

A public rebuild of a drawing-review pipeline as a FastAPI service: golden eval gates, non-root image, Kind Ready with a rollback transcript, GHCR digest on file. Stubs on the laptop; OpenShift / KServe YAML is there and schema-checked, but I have not applied it on a live GPU cluster.

### LoRA, with the boring step first

[lora-finetune-recipe](https://github.com/lyftium/lora-finetune-recipe)

Same held-out set for base and adapter, lineage next to the report. The thing that surprised me: few-shot prompting alone jumps exact_match **0.0 → 0.8** on that hash — so the flashy adapter table (vs an unprompted base) is not the fair story. I leave both in the README on purpose.

### Serving and the front door

[vllm-serving-benchmark-t4](https://github.com/lyftium/vllm-serving-benchmark-t4) · [agent-gateway-stack](https://github.com/lyftium/agent-gateway-stack)

One Kaggle campaign on 2× T4: continuous batching moved generated tok/s by about **35×** on the small model (`results.json`, single run). The gateway next door owns keys, quotas, and approve-once tool calls; Kind proves it against a mock, not a GPU farm.

---

I trained as an aerospace engineer and in data science (Liora × Mines Paris – PSL). Public work here is fashion OBD logs, synthetic drawings, and laptop / Kind evidence — not customer production metrics. If a README sounds careful about limits, that is intentional.
