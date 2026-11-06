# Pierre — Data Scientist · Applied ML & AI Engineering

Aerospace engineering background; data science training with Liora / Mines Paris – PSL.  
Paris · French / English.

I work on both model evaluation and the software needed to use models: scoring a policy from logged data, checking whether fine-tuning helps beyond prompting, or turning a document pipeline into an API another engineer can run.

Looking for data science and applied ML / AI engineering roles.

[Kaggle](https://www.kaggle.com/lyftium) · [Notes](https://lyftium.github.io) · [GitHub](https://github.com/lyftium)

<!-- Add when ready: [CV](...) · [LinkedIn](...) · [Email](mailto:...) -->

## Selected projects

### Evaluating recommendation policies from logged data

[off-policy-bandit-evaluation](https://github.com/lyftium/off-policy-bandit-evaluation)

Can we evaluate a recommendation policy before running an A/B test?

I compared direct, importance-weighted, and doubly robust estimators on **1.37 million** rounds from the Open Bandit Dataset (uniform-random logs → estimate the production Bernoulli-TS policy’s CTR), with bootstrap confidence intervals and a chronological split. Digits live in [`results.json`](https://github.com/lyftium/off-policy-bandit-evaluation/blob/main/results.json).

The useful finding: Direct Method gave a **very narrow** interval that still **missed** the reference CTR, while IPW / SNIPW / DR intervals covered it (about **5.3–6.1 %** relative error). A simple popularity policy looked stronger on the hold-out — that is a reason to run an experiment, not a result I would ship from the estimate alone.

[Walkthrough](https://github.com/lyftium/off-policy-bandit-evaluation/blob/main/docs/DEMO.md)

### Turning a drawing-review prototype into a service

[drawing-review-service](https://github.com/lyftium/drawing-review-service)

I built a FastAPI demonstrator for an engineering-drawing review workflow: pluggable inference backends, structured responses, evaluation checks, and a container image.

CPU stubs let another engineer run the tests without downloading model weights. The service has been exercised on a local Kind cluster (including a rollback transcript); the OpenShift AI / KServe manifests are in the repo and schema-checked, but I have **not** applied them on a live GPU cluster.

[Walkthrough](https://github.com/lyftium/drawing-review-service/blob/main/docs/DEMO.md)

### Measuring LLM serving trade-offs on T4 GPUs

[vllm-serving-benchmark-t4](https://github.com/lyftium/vllm-serving-benchmark-t4)

I benchmarked vLLM on two Tesla T4 GPUs — batching, quantization, tensor parallelism, and prefix caching — and kept the raw campaign in [`results.json`](https://github.com/lyftium/vllm-serving-benchmark-t4/blob/main/results.json).

One result that changed how I talk about quantization: for the tested **1.5B** model, AWQ was about **1.6×** faster than fp16 at one sequence, but about **0.5×** at 128 sequences. Continuous batching on the small model moved generated tok/s by roughly **35×** in that same campaign. These are exploratory **single-run** measurements, not production capacity estimates.

[Walkthrough](https://github.com/lyftium/vllm-serving-benchmark-t4/blob/main/docs/DEMO.md)

## Other work

- [LoRA fine-tuning](https://github.com/lyftium/lora-finetune-recipe) — same held-out set for base and adapter; few-shot prompting alone reached exact_match **0.8**, so the unprompted base is a weak baseline for judging the adapter.
- [AI gateway](https://github.com/lyftium/agent-gateway-stack) — key auth, quotas, failover, and approve-once tool calls (Kind + mock LLM; not a live GPU cluster).
- [Kaggle notebooks](https://www.kaggle.com/lyftium)

## Tools used in these projects

Python, scikit-learn / GBDT, PyTorch, FastAPI, Docker, Kubernetes (Kind), vLLM, PEFT / Hugging Face.
