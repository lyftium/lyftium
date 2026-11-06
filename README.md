# Pierre

Data scientist / AI integration — Paris, FR / EN.

I evaluate decisions and models first, then package them so another engineer can call, score, and prepare them for a cluster.

[Kaggle](https://www.kaggle.com/lyftium) · [Pages](https://lyftium.github.io) · [GitHub](https://github.com/lyftium)

---

## Look here first

- [Off-policy bandit evaluation](https://github.com/lyftium/off-policy-bandit-evaluation) — 1.37 M OBD logs; IPW / SNIPW / DR + bootstrap CIs vs known CTR ([`results.json`](https://github.com/lyftium/off-policy-bandit-evaluation/blob/main/results.json)) · [DEMO](https://github.com/lyftium/off-policy-bandit-evaluation/blob/main/docs/DEMO.md)
- [Drawing review service](https://github.com/lyftium/drawing-review-service) — FastAPI + golden gates + Kind Ready + GHCR digest (OpenShift / KServe YAML prepared, not live GPU) · [DEMO](https://github.com/lyftium/drawing-review-service/blob/main/docs/DEMO.md)
- [vLLM on T4](https://github.com/lyftium/vllm-serving-benchmark-t4) / [LoRA recipe](https://github.com/lyftium/lora-finetune-recipe) — measured serving levers; baseline-before-adapter training

Also: [agent-gateway-stack](https://github.com/lyftium/agent-gateway-stack) (auth, quotas, approve-once). Each flagship has `docs/DEMO.md`.

---

## Projects

### 1. Off-policy evaluation — [off-policy-bandit-evaluation](https://github.com/lyftium/off-policy-bandit-evaluation)

From 1.37 M uniform-random logs, IPW / SNIPW / DR recover production BTS CTR within **5.3–6.1 %** relative error; Direct Method’s CI misses the truth. Chronological hold-out: per-slot popularity ≈ **2.1×** the logger — an A/B candidate, not a shipped lift.

### 2. Drawing review service — [drawing-review-service](https://github.com/lyftium/drawing-review-service)

FastAPI microservice with golden eval gates, Docker, Kind Ready, and a recorded GHCR digest. CPU stubs and synthetic drawings by default; OpenShift / KServe paths are in-repo, not live on GPU.

### 3. LoRA recipe — [lora-finetune-recipe](https://github.com/lyftium/lora-finetune-recipe)

Same-split baseline before the adapter. Prompted modes on a fixed `test_data_hash`: exact_match **0.0 → 0.8** fewshot. Tiny DDP is two-process mechanics, not multi-GPU quality.

### 4. vLLM on T4 — [vllm-serving-benchmark-t4](https://github.com/lyftium/vllm-serving-benchmark-t4)

One Kaggle campaign: continuous batching raised generated tok/s **~35×** (`max_num_seqs` 1 → 128 on 0.5B fp16). Digits only in `results.json` under a fixed PROTOCOL. Single-run, not a multi-seed SLO.

### 5. Gateway & agent — [agent-gateway-stack](https://github.com/lyftium/agent-gateway-stack)

Key-hash auth, per-tenant quotas, failover, and an approve-once tool agent. Kind Ready against mock-llm; live GPU OpenShift not claimed.

---

## Limits

| Easy to infer | What is actually shown |
|:--|:--|
| OpenShift / KServe “deployed” | Manifests validate; no live GPU cluster apply |
| Multi-GPU training | Two-process DDP mechanics on a tiny model |
| Serving SLOs | One Kaggle T4 campaign, one run per config |
| Causal / A/B lift shipped | OPE **estimated** lift + CIs on fashion OBD logs |

<sub>Aerospace engineering background · Data science (Liora × Mines Paris – PSL)</sub>
