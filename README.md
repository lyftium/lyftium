# Pierre

Data scientist with an integration bent — Paris, FR / EN.

I score models and decisions before arguing they should ship, then package the pieces so another engineer can call them, measure them, and prepare them for a cluster.

[Kaggle](https://www.kaggle.com/lyftium) · [Pages](https://lyftium.github.io) · [GitHub](https://github.com/lyftium)

---

## Where to look

| If you care about… | Open this | You should see |
|:--|:--|:--|
| Decision evaluation | [off-policy-bandit-evaluation](https://github.com/lyftium/off-policy-bandit-evaluation) · [DEMO](https://github.com/lyftium/off-policy-bandit-evaluation/blob/main/docs/DEMO.md) | 1.37 M real logs, IPW / SNIPW / DR with bootstrap CIs, a time split — estimated lift is not an A/B |
| Packaging a model path | [drawing-review-service](https://github.com/lyftium/drawing-review-service) · [DEMO](https://github.com/lyftium/drawing-review-service/blob/main/docs/DEMO.md) | FastAPI, golden gates, Kind Ready, a GHCR digest; OpenShift / KServe YAML is prepared, not live on GPU |
| Serving under load | [vLLM on T4](https://github.com/lyftium/vllm-serving-benchmark-t4) → [agent-gateway-stack](https://github.com/lyftium/agent-gateway-stack) → DRS `deploy/kserve/` | Measured levers (one campaign), a Kind gateway against a mock, manifests rather than a live InferenceService |

Also worth five minutes: [lora-finetune-recipe](https://github.com/lyftium/lora-finetune-recipe) — same-split baseline before the adapter. Each flagship has a `docs/DEMO.md`.

---

## Projects

### 1. Off-policy evaluation — [off-policy-bandit-evaluation](https://github.com/lyftium/off-policy-bandit-evaluation)

Can you rank a new recommendation policy from someone else’s logs before an A/B test?

On the Open Bandit Dataset, yes: within **5.3–6.1 %** relative error on 1.37 M rounds. IPW / SNIPW / DR intervals cover the production CTR; Direct Method’s CI is tiny and misses it. A per-slot popularity policy looks about **2.1×** the logger on a chronological hold-out — a candidate for A/B, not a shipped lift.

*Python · GBDT · bootstrap CIs · [`results.json`](https://github.com/lyftium/off-policy-bandit-evaluation/blob/main/results.json)*

### 2. Drawing review as a service — [drawing-review-service](https://github.com/lyftium/drawing-review-service)

Detect → OCR → VLM → rules → human review, rebuilt as a public demonstrator on synthetic drawings. Contracts and golden eval on the laptop; Kind apply Ready with a rollback transcript; GHCR digest recorded. OpenShift AI on GPU is not claimed live.

*FastAPI · Docker · Kubernetes / KServe manifests · Prometheus*

### 3. LoRA with the baseline first — [lora-finetune-recipe](https://github.com/lyftium/lora-finetune-recipe)

One CLI: evaluate the base, train a LoRA adapter, evaluate on the **same** held-out set, write a report with git / config / data hashes. Prompted modes on that hash: exact_match **0.0 → 0.8** fewshot. Tiny DDP shows two-process mechanics, not multi-GPU quality.

*PyTorch · Hugging Face · PEFT*

### 4. Serving levers on T4 — [vllm-serving-benchmark-t4](https://github.com/lyftium/vllm-serving-benchmark-t4)

One Kaggle campaign on 2× Tesla T4: continuous batching, online concurrency, AWQ vs fp16, tensor parallel, prefix caching. Digits live only in `results.json` / CSV under a fixed [PROTOCOL](https://github.com/lyftium/vllm-serving-benchmark-t4/blob/main/PROTOCOL.md). Single-run exploratory — not a multi-seed SLO.

*vLLM · Python · Kaggle T4*

### 5. Gateway and bounded agent — [agent-gateway-stack](https://github.com/lyftium/agent-gateway-stack)

OpenAI-compatible gateway (key-hash auth, per-tenant quotas, failover, no prompt logging) and a tool-calling agent with an allowlist, typed schemas, and approve-once for side effects. Kind Ready against mock-llm; ISA and approve-once transcripts on disk. Live GPU OpenShift is not claimed.

*FastAPI · Kubernetes / OpenShift overlays · Prometheus*

---

## Limits worth reading once

| Easy to infer | What is actually shown |
|:--|:--|
| “OpenShift / KServe deployed” | Manifests validate; no live GPU cluster apply |
| “Multi-GPU training” | Two-process DDP mechanics on a tiny model |
| “Serving SLOs” | One Kaggle T4 campaign, one run per config |
| “Causal / A/B lift shipped” | OPE **estimated** lift + CIs on fashion OBD logs |

---

Pin tip for the GitHub UI: data-science glance → **OPE · LoRA · DRS · vLLM · AGS**. Integrator glance → **DRS · vLLM · AGS · LoRA · OPE**.

<sub>Aerospace engineering · Data science (Liora × Mines Paris – PSL)</sub>
