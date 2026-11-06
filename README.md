# Pierre — Data Scientist & AI Engineer

Aerospace engineering background. I evaluate models and decision policies, then build the services needed to use them.

Paris · French / English.

Looking for data science and applied ML / AI engineering roles.

[Kaggle](https://www.kaggle.com/lyftium) · [Notes](https://lyftium.github.io)

<!-- Add when you have them: [CV](...) · [LinkedIn](...) · [Email](mailto:...) -->
<!-- Add the exact program name and whether it is completed or in progress. -->

## Selected projects

### Evaluating recommendation policies from logged data

[off-policy-bandit-evaluation](https://github.com/lyftium/off-policy-bandit-evaluation)

Can we score a new recommendation policy from someone else’s logs before an A/B test?

I implemented the estimators (direct method, IPW, self-normalised IPW, doubly robust), the chronological split, and the bootstrap intervals. The Open Bandit Dataset supplies the logs; the `obp` library is only used for the production policy’s action distribution.

On 1.37 million uniform-random rounds, the direct method’s confidence interval was tiny and still missed the production CTR. The weighted estimators’ intervals covered it. A simple popularity policy looked stronger on the hold-out — that is a reason to run an experiment, not a result I would ship from the estimate.

Offline estimate only; no live A/B. Fashion e-commerce logs, not a customer system.

[Walkthrough](https://github.com/lyftium/off-policy-bandit-evaluation/blob/main/docs/DEMO.md)

### Turning a drawing-review prototype into a service

[drawing-review-service](https://github.com/lyftium/drawing-review-service)

A local drawing-review script is hard to share or score. I built a FastAPI service around that workflow: request and response contracts, golden-set checks, and a container that runs as a non-root user.

CPU stubs mean another engineer can run the tests without downloading model weights. I tested the Kubernetes deployment on a local Kind cluster, including a rollback; I have not recorded a GPU-backed deployment.

[Walkthrough](https://github.com/lyftium/drawing-review-service/blob/main/docs/DEMO.md)

### Measuring LLM serving trade-offs on T4 GPUs

[vllm-serving-benchmark-t4](https://github.com/lyftium/vllm-serving-benchmark-t4)

I wanted to see which serving settings actually move throughput and latency together, so I ran one vLLM campaign on two Tesla T4 GPUs and kept the protocol next to the numbers.

For the 1.5B model I tested, AWQ was faster than fp16 at one sequence and slower at 128 sequences. Tensor parallelism also failed to help the small model under load — the extra communication cost was larger than the split. These are single-run measurements on that hardware, not a capacity plan.

[Walkthrough](https://github.com/lyftium/vllm-serving-benchmark-t4/blob/main/docs/DEMO.md)

## Other work

- [LoRA fine-tuning](https://github.com/lyftium/lora-finetune-recipe) — same held-out set for the base model and the adapter. Few-shot prompting alone reached exact match 0.80, so I do not treat the unprompted base as the baseline for the adapter.
- [AI gateway](https://github.com/lyftium/agent-gateway-stack) — API keys, quotas, failover, and an approval step before a tool call with side effects. Overhead was measured against a mock backend, not a live inference workload.
- [Kaggle notebooks](https://www.kaggle.com/lyftium)

## Tools used in these projects

Python, gradient-boosted trees, PyTorch, FastAPI, Docker, Kubernetes (Kind), vLLM, PEFT / Hugging Face.
