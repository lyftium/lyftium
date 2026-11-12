# Pierre — Data Scientist & AI Engineer

Paris, France · Français / English · open to CDI / freelance collaboration in FR

I evaluate whether a model or decision policy beats its baseline, then package the surrounding service so another engineer can run and test it. Looking for **data science**, **applied ML**, and **AI integration** roles (services, evaluation, LLM serving).

[Portfolio site](https://lyftium.github.io) · [Kaggle](https://www.kaggle.com/lyftium)

**Formation :** programme Data Scientist — Liora, avec Mines Paris – PSL

## 30-second map

| Focus | Repo | One-click |
|---|---|---|
| Package a review pipeline as a service | [drawing-review-service](https://github.com/lyftium/drawing-review-service) | [Demo](https://github.com/lyftium/drawing-review-service/blob/main/docs/DEMO.md) · [Evidence](https://github.com/lyftium/drawing-review-service/tree/main/docs/evidence) |
| Measure LLM serving trade-offs | [vllm-serving-benchmark-t4](https://github.com/lyftium/vllm-serving-benchmark-t4) | [Demo](https://github.com/lyftium/vllm-serving-benchmark-t4/blob/main/docs/DEMO.md) · [Results](https://github.com/lyftium/vllm-serving-benchmark-t4/blob/main/results.json) |
| Evaluate policies before an A/B test | [off-policy-bandit-evaluation](https://github.com/lyftium/off-policy-bandit-evaluation) | [Demo](https://github.com/lyftium/off-policy-bandit-evaluation/blob/main/docs/DEMO.md) · [Results](https://github.com/lyftium/off-policy-bandit-evaluation/blob/main/results.json) |
| Prompted baseline before LoRA | [lora-finetune-recipe](https://github.com/lyftium/lora-finetune-recipe) | [Demo](https://github.com/lyftium/lora-finetune-recipe/blob/main/docs/DEMO.md) |
| Gateway + approval-gated tools | [agent-gateway-stack](https://github.com/lyftium/agent-gateway-stack) | [Demo](https://github.com/lyftium/agent-gateway-stack/blob/main/docs/DEMO.md) |

## Selected work

### Packaging a drawing-review pipeline as a service

[drawing-review-service](https://github.com/lyftium/drawing-review-service)

How do you make a drawing-analysis workflow testable when its model backends and GPU environment are not available to every engineer?

I built a FastAPI demonstrator with typed responses, interchangeable inference backends, golden-set evaluation, and business-defect gates. The container and Kubernetes deployment can be exercised independently of model weights: CPU stubs keep the API and evaluation path runnable, while recorded Kind checks cover a local deployment and rollback. Separating extraction metrics from review findings makes a noisy backend visible even when some extraction scores still look good.

**Limit:** evaluation uses 12 synthetic drawings and stub backends, not real-model quality or a GPU-backed deployment.

[Run the demo](https://github.com/lyftium/drawing-review-service/blob/main/docs/DEMO.md) · [Evidence](https://github.com/lyftium/drawing-review-service/tree/main/docs/evidence)

### Measuring LLM serving trade-offs

[vllm-serving-benchmark-t4](https://github.com/lyftium/vllm-serving-benchmark-t4)

When does a serving optimization stop helping as load increases? I compared batching, AWQ, tensor parallelism, and prefix caching with vLLM 0.6.6.post1 on two Tesla T4 GPUs, keeping the measurements and conditions alongside the findings.

For Qwen2.5-1.5B-Instruct, AWQ throughput was 1.61× fp16 at one sequence but 0.50× at 128 sequences. The useful lesson was not “quantize for speed”: the right choice depended on load, and the small quality probe moved too. I now treat model, workload, and concurrency as part of any serving result.

**Limit:** one run per configuration; immutable model revisions are not recorded.

[Results and inspection walkthrough](https://github.com/lyftium/vllm-serving-benchmark-t4/blob/main/docs/DEMO.md)

### Evaluating policies before an A/B test

[off-policy-bandit-evaluation](https://github.com/lyftium/off-policy-bandit-evaluation)

I used 1.37 million logged interactions from the Open Bandit Dataset to compare direct-method, IPW, self-normalized IPW, and doubly robust estimators, with bootstrap intervals and a chronological policy holdout.

A narrow direct-method interval missed the independently measured reference CTR, while the weighted estimators covered it. A later experiment identified a popularity policy worth testing rather than declaring a win from offline logs. The chronological split addresses time leakage in policy evaluation; the reward model’s cross-fitting is a separate mechanism.

**Limit:** these are offline estimates on public fashion-click data, not observed A/B lift.

[Results and reproduction](https://github.com/lyftium/off-policy-bandit-evaluation/blob/main/docs/DEMO.md)

## Supporting projects

- **[LoRA fine-tuning recipe](https://github.com/lyftium/lora-finetune-recipe):** prompted baselines, adapter training, and run lineage. Few-shot prompting reached 0.80 exact match; the recorded 0.97 adapter result used a different prompt mode, so it is not a controlled gain over that baseline. [Demo](https://github.com/lyftium/lora-finetune-recipe/blob/main/docs/DEMO.md)
- **[AI gateway and tool agent](https://github.com/lyftium/agent-gateway-stack):** key-hash authentication, per-replica quotas, failover, and an approval-gated tool-call demonstrator. Overhead was measured against a mock backend; approval records an intent without separate operator authentication. [Demo](https://github.com/lyftium/agent-gateway-stack/blob/main/docs/DEMO.md)
- **[Road-damage detection](https://github.com/lyftium/road-damage-yolo-evaluation):** YOLO11n on 402 public held-out images, mAP@50 0.526. The same split was also used to choose confidence thresholds, so this is an optimistic validation result from one time-capped run, not an independent test.

## Tools demonstrated in these projects

Python, scikit-learn, PyTorch, PEFT, FastAPI, Docker, Kubernetes (Kind), and vLLM.
