# Pierre

Data scientist · AI integration — Paris, FR / EN.

I evaluate models and decisions before arguing they should ship, then package them so another engineer can call, score, and prepare them for a cluster.

[Kaggle](https://www.kaggle.com/lyftium) · [Pages](https://lyftium.github.io) · [GitHub](https://github.com/lyftium)

---

## Start here

| Lens | Repo | Proof you can open in 30 s |
|:--|:--|:--|
| Data science | [off-policy-bandit-evaluation](https://github.com/lyftium/off-policy-bandit-evaluation) · [DEMO](https://github.com/lyftium/off-policy-bandit-evaluation/blob/main/docs/DEMO.md) | 1.37 M real logs · IPW / SNIPW / DR · bootstrap CIs · time split · estimated lift ≠ A/B |
| Integration | [drawing-review-service](https://github.com/lyftium/drawing-review-service) · [DEMO](https://github.com/lyftium/drawing-review-service/blob/main/docs/DEMO.md) | FastAPI + golden gates · Kind Ready · GHCR digest · OpenShift / KServe YAML prepared, not live GPU |
| Serving path | [vLLM T4](https://github.com/lyftium/vllm-serving-benchmark-t4) → [gateway](https://github.com/lyftium/agent-gateway-stack) → DRS `deploy/kserve/` | Measured levers (one campaign) · Kind gateway + mock · manifests, not a live InferenceService |

Also: [lora-finetune-recipe](https://github.com/lyftium/lora-finetune-recipe) — same-split baseline before the adapter. Every flagship has `docs/DEMO.md`.

---

## Projects

### 1. Off-policy evaluation — [off-policy-bandit-evaluation](https://github.com/lyftium/off-policy-bandit-evaluation)

Can you rank a new recommendation policy from someone else’s logs before an A/B test?

On the Open Bandit Dataset: within **5.3–6.1 %** relative error on 1.37 M rounds. IPW / SNIPW / DR cover the production CTR; Direct Method’s CI misses it. A per-slot popularity policy looks about **2.1×** the logger on a chronological hold-out — an A/B candidate, not a shipped lift.

*Python · GBDT · bootstrap CIs · [`results.json`](https://github.com/lyftium/off-policy-bandit-evaluation/blob/main/results.json)*

### 2. Drawing review as a service — [drawing-review-service](https://github.com/lyftium/drawing-review-service)

Detect → OCR → VLM → rules → human review, rebuilt as a public demonstrator on synthetic drawings. Contracts and golden eval locally; Kind Ready with a rollback transcript; GHCR digest on record. OpenShift AI on GPU is not claimed live.

*FastAPI · Docker · Kubernetes / KServe manifests · Prometheus*

### 3. LoRA with the baseline first — [lora-finetune-recipe](https://github.com/lyftium/lora-finetune-recipe)

Evaluate the base, train a LoRA adapter, evaluate on the **same** held-out set, write a report with git / config / data hashes. Prompted modes on that hash: exact_match **0.0 → 0.8** fewshot. Tiny DDP shows two-process mechanics, not multi-GPU quality.

*PyTorch · Hugging Face · PEFT*

### 4. Serving levers on T4 — [vllm-serving-benchmark-t4](https://github.com/lyftium/vllm-serving-benchmark-t4)

One Kaggle campaign on 2× Tesla T4: batching, concurrency, AWQ vs fp16, tensor parallel, prefix caching. Digits only in `results.json` / CSV under a fixed [PROTOCOL](https://github.com/lyftium/vllm-serving-benchmark-t4/blob/main/PROTOCOL.md). Single-run exploratory — not a multi-seed SLO.

*vLLM · Python · Kaggle T4*

### 5. Gateway and bounded agent — [agent-gateway-stack](https://github.com/lyftium/agent-gateway-stack)

OpenAI-compatible gateway (auth, quotas, failover, no prompt logging) and a tool-calling agent with allowlist, typed schemas, and approve-once for side effects. Kind Ready against mock-llm; ISA and approve-once transcripts on disk. Live GPU OpenShift is not claimed.

*FastAPI · Kubernetes / OpenShift overlays · Prometheus*

---

## Limits

| Easy to infer | What is actually shown |
|:--|:--|
| OpenShift / KServe “deployed” | Manifests validate; no live GPU cluster apply |
| Multi-GPU training | Two-process DDP mechanics on a tiny model |
| Serving SLOs | One Kaggle T4 campaign, one run per config |
| Causal / A/B lift shipped | OPE **estimated** lift + CIs on fashion OBD logs |

---

**Pin tip (GitHub UI, before the interview):** data science → **OPE · LoRA · DRS · vLLM · AGS**. Integrator → **DRS · vLLM · AGS · LoRA · OPE**.

<sub>Aerospace engineering · Data science (Liora × Mines Paris – PSL)</sub>
