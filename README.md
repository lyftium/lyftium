# Pierre — Data Scientist & AI Engineer

Aerospace engineering background. I evaluate models and decision policies, then build the services needed to use them.

Paris · French / English.

Looking for data science and applied ML / AI engineering roles.

[Kaggle](https://www.kaggle.com/lyftium) · [Notes](https://lyftium.github.io)

## Selected projects

### Turning a drawing-review prototype into a service

[drawing-review-service](https://github.com/lyftium/drawing-review-service)

A drawing-review script that only runs on one machine is hard to share or score. I built the FastAPI contracts, the golden-set checks, and a container that runs as a non-root user. CPU stubs stand in for the models, so the tests do not download weights.

I applied that image on a local Kind cluster and recorded a rollback. Manifests aimed at a GPU cluster are schema-checked only.

[Walkthrough](https://github.com/lyftium/drawing-review-service/blob/main/docs/DEMO.md)

### Measuring LLM serving trade-offs on T4 GPUs

[vllm-serving-benchmark-t4](https://github.com/lyftium/vllm-serving-benchmark-t4)

I ran one vLLM campaign on two Tesla T4 GPUs and kept the protocol next to the numbers.

For the 1.5B model, AWQ was about 1.6× fp16 at one sequence and about 0.5× at 128 sequences. Tensor parallelism was slower than one GPU once the batch was large. One run per cell, on that hardware.

[Walkthrough](https://github.com/lyftium/vllm-serving-benchmark-t4/blob/main/docs/DEMO.md)

### Evaluating recommendation policies from logged data

[off-policy-bandit-evaluation](https://github.com/lyftium/off-policy-bandit-evaluation)

Can we score a new recommendation policy from someone else’s logs before an A/B test?

I implemented direct method, IPW, self-normalised IPW, and doubly robust estimators, plus the chronological split and the bootstrap intervals. The Open Bandit Dataset supplies the logs. `obp` is only used for the production policy’s action distribution.

On 1.37 million uniform-random rounds, the direct method’s interval was tiny and missed the production click-through rate. The weighted estimators’ intervals covered it. A popularity policy looked stronger on the hold-out; that is a reason to run an experiment, not a shipped result. No live A/B.

[Walkthrough](https://github.com/lyftium/off-policy-bandit-evaluation/blob/main/docs/DEMO.md)

## Other work

- [LoRA fine-tuning](https://github.com/lyftium/lora-finetune-recipe) — few-shot prompting reached exact match 0.80 on 100 held-out examples. The adapter’s 0.97 is against an unprompted base, not against that prompt.
- [AI gateway](https://github.com/lyftium/agent-gateway-stack) — key auth, per-replica quotas, failover, and a demo approval that records a queue intent. The latency figure is against a mock model.
- [Kaggle notebooks](https://www.kaggle.com/lyftium)

## Tools used in these projects

Python, gradient-boosted trees, PyTorch, FastAPI, Docker, Kubernetes (Kind), vLLM, PEFT.
