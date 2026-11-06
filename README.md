# Pierre — Data Scientist & AI Engineer

Paris, France · French / English

I work on two connected questions: does a model or decision policy improve on its baseline, and can another engineer run the service around it? My projects combine experimental evaluation with APIs, containers, and measured inference trade-offs. I am looking for data science, applied ML, and AI integration roles.

[Project notes](https://lyftium.github.io) · [Kaggle](https://www.kaggle.com/lyftium)

**Education:** Data Scientist program — Liora, with Mines Paris – PSL

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

- **[LoRA fine-tuning recipe](https://github.com/lyftium/lora-finetune-recipe):** prompted baselines, adapter training, and run lineage. Few-shot prompting reached 0.80 exact match; the recorded 0.97 adapter result used a different prompt mode, so it is not a controlled gain over that baseline.
- **[AI gateway and tool agent](https://github.com/lyftium/agent-gateway-stack):** key-hash authentication, per-replica quotas, failover, and an approval-gated tool-call demonstrator. Overhead was measured against a mock backend; approval records an intent without separate operator authentication.
- **GOLLAMC:** final project for the Liora Data Scientist program with Mines Paris – PSL. The repository is private; a demo is available on request.

## Benchmarks

| Area | What I measured | Evidence |
|---|---|---|
| LLM serving | 1.61× AWQ/fp16 throughput for 1.5B at one sequence, but 0.50× at 128 | [vLLM benchmark](https://github.com/lyftium/vllm-serving-benchmark-t4/blob/main/BENCHMARK.md) |
| Off-policy evaluation | DM missed the independently logged reference CTR; IPW/SNIPW/DR intervals covered it | [1.37M-round study](https://github.com/lyftium/off-policy-bandit-evaluation/blob/main/BENCHMARK.md) |
| Fine-tuning | Few-shot base 0.80; adapter 0.97 with a different prompt, so not a controlled +0.17 gain | [LoRA reports](https://github.com/lyftium/lora-finetune-recipe/blob/main/BENCHMARK.md) |
| Object detection | 0.526 mAP@50 on 402 held-out public road images | [YOLO evaluation](https://github.com/lyftium/road-damage-yolo-evaluation/blob/main/BENCHMARK.md) |
| Service overhead | +1.65 ms p50 non-stream and +1.54 ms p50 stream at concurrency 1, against an instant mock | [Gateway benchmark](https://github.com/lyftium/agent-gateway-stack/blob/main/BENCHMARK.md) |
| Evaluation harness | Noisy synthetic stubs failed review gates despite 0.9419 detection recall | [Drawing-service evaluation](https://github.com/lyftium/drawing-review-service/blob/main/BENCHMARK.md) |

## Tools demonstrated in these projects

Python, scikit-learn, PyTorch, PEFT, FastAPI, Docker, Kubernetes (Kind), and vLLM.
