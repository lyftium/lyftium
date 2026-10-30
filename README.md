### Hi, I'm lyftium 👋

**Data Scientist · Paris, France** · open to data scientist roles · [optinampout.com](https://www.optinampout.com)

I build ML models, test them against honest baselines, and ship the ones that hold up as real services (FastAPI, Docker, Kubernetes, vLLM).

Before data science I was an aerospace engineer on A220 cabin interiors: design, technical drawings, validation and certification. That's where I learned to care about evidence before sign-off, and it's how I approach models today.

## Selected work

- **[off-policy-bandit-evaluation](https://github.com/lyftium/off-policy-bandit-evaluation)**: estimating a recommender's click-through rate without running an A/B test. On 13.7M real logged rounds, the importance-weighted estimators landed within 6% of the true CTR.
- **GOLLAMC** *(private, demo on request)*: AI review of engineering drawings. YOLO symbol detection, OCR and a LoRA-tuned LLaVA, orchestrated by a LangGraph agent, fully local. 74% mAP@50 and 69.2% F1 on 38 held-out drawings.
- **[drawing-review-service](https://github.com/lyftium/drawing-review-service)**: taking a prototype like that to production. FastAPI, Docker, Kubernetes manifests for GPU nodes, vLLM serving, Prometheus metrics, and a regression set that gates each release.
- **[lora-finetune-recipe](https://github.com/lyftium/lora-finetune-recipe)**: LoRA fine-tuning with a baseline-vs-adapter evaluation. Exact match went from 0% to 97% on a synthetic task.

## Toolbox

- **Stats & experimentation:** A/B testing, causal inference, off-policy evaluation, bootstrap
- **ML & LLMs:** scikit-learn, gradient boosting, PyTorch, Hugging Face, computer vision, LoRA/QLoRA, vLLM, LangGraph
- **Data:** SQL, pandas, PySpark, Kafka, ClickHouse
- **MLOps:** Docker, Kubernetes, FastAPI, MLflow, GitHub Actions

## Education

Data Scientist program, Liora with Mines Paris – PSL

---

Hiring for a data role in Paris? I'd be glad to talk. Reach me through [optinampout.com](https://www.optinampout.com).
