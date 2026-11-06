<div align="center">

# Pierre

**Data scientist · AI integration**

I evaluate models and decisions before shipping them, then package the pieces so another engineer can call, score, and prepare them.

[Kaggle](https://www.kaggle.com/lyftium) · [Pages](https://lyftium.github.io) · Paris · FR / EN

</div>

---

### Open these first

| Role | Start here | What you will actually find |
|:--|:--|:--|
| Data science | [off-policy-bandit-evaluation](https://github.com/lyftium/off-policy-bandit-evaluation) · [DEMO](https://github.com/lyftium/off-policy-bandit-evaluation/blob/main/docs/DEMO.md) | 1.37 M real bandit logs · IPW / SNIPW / DR · bootstrap CIs · time split · estimated lift ≠ A/B |
| Integration | [drawing-review-service](https://github.com/lyftium/drawing-review-service) · [DEMO](https://github.com/lyftium/drawing-review-service/blob/main/docs/DEMO.md) | FastAPI + golden gates · Kind Ready · GHCR digest · OpenShift / KServe YAML prepared, **not** live GPU |
| Serving path | [vLLM T4](https://github.com/lyftium/vllm-serving-benchmark-t4) → [agent-gateway-stack](https://github.com/lyftium/agent-gateway-stack) → DRS `deploy/kserve/` | Measured levers (single-run) · Kind gateway + mock · manifests, not a live InferenceService |

Also: [lora-finetune-recipe](https://github.com/lyftium/lora-finetune-recipe) (same-split baseline before the adapter). Every flagship has a `docs/DEMO.md`.

---

## Projects

### [off-policy-bandit-evaluation](https://github.com/lyftium/off-policy-bandit-evaluation) — rank a policy from someone else’s logs

On the Open Bandit Dataset, estimators recover the production CTR within **5.3–6.1 %** relative error from 1.37 M uniform-random rounds (IPW / SNIPW / DR CIs cover truth; DM’s does not). A per-slot popularity policy looks ~**2.1×** the logger on a chronological hold-out — a candidate for A/B, not a shipped lift.

Stack: Python · scikit-learn / GBDT · bootstrap CIs · `results.json`

### [drawing-review-service](https://github.com/lyftium/drawing-review-service) — drawing review as a microservice

Detection → OCR → VLM → rules → human review, rebuilt as a public demonstrator (synthetic drawings only). Contracts, golden eval, non-root image, Kind apply Ready + rollback transcript, GHCR digest. OpenShift AI / GPU apply is **not** documented live.

Stack: FastAPI · Docker · Kubernetes / KServe manifests · Prometheus

### [lora-finetune-recipe](https://github.com/lyftium/lora-finetune-recipe) — baseline first, then the adapter

One CLI: evaluate the base → train LoRA → evaluate on the **same** held-out set → write a report with git / config / data hashes. Prompted modes on the same `test_data_hash`: exact_match unprompted **0.0** → fewshot **0.8**. Tiny DDP proves two-process mechanics, not multi-GPU quality.

Stack: PyTorch · Hugging Face · PEFT

### [vllm-serving-benchmark-t4](https://github.com/lyftium/vllm-serving-benchmark-t4) — which serving levers move numbers

One Kaggle campaign on 2× Tesla T4: continuous batching, online concurrency, AWQ vs fp16, tensor parallel, prefix caching. Numbers live only in `results.json` / CSV under a fixed [`PROTOCOL.md`](https://github.com/lyftium/vllm-serving-benchmark-t4/blob/main/PROTOCOL.md). Single-run exploratory — not a multi-seed SLO.

Stack: vLLM · Python · Kaggle T4

### [agent-gateway-stack](https://github.com/lyftium/agent-gateway-stack) — shared front door + bounded agent

OpenAI-compatible gateway (key-hash auth, per-tenant quotas, failover, no prompt logging) and a tool-calling agent with allowlist, typed schemas, and approve-once for side effects. Kind Ready against mock-llm; ISA + approve-once agent-run transcript on disk. Live GPU OpenShift **not** claimed.

Stack: FastAPI · Kubernetes / OpenShift overlays · Prometheus

---

## What this is not

| Easy to infer | What is actually shown |
|:--|:--|
| “OpenShift / KServe deployed” | Manifests validate; **no** live GPU cluster apply |
| “Multi-GPU training” | Two-process DDP mechanics on a tiny model |
| “Serving SLOs” | One Kaggle T4 campaign, one run per config |
| “Causal / A/B lift shipped” | OPE **estimated** lift + CIs on fashion OBD logs |

---

**Pins (GitHub UI):** data-science glance → OPE · LoRA · DRS · vLLM · AGS. Integrator glance → DRS · vLLM · AGS · LoRA · OPE.

<sub>Aerospace engineering · Data science (Liora × Mines Paris – PSL) · FR: évaluation hors-politique, baselines, packaging API→image→Kind — OpenShift préparé, pas live.</sub>
