<h1 align="center">LYFTIUM</h1>
<h3 align="center">Data Scientist · ML Engineer · Aerospace AI (Safran)</h3>

<p align="center">
  <b>Experimentation & causal inference · Recommendation · Computer vision & VLMs · Production ML</b>
</p>

<p align="center">
  <a href="https://www.optinampout.com"><img src="https://img.shields.io/badge/Website-optinampout.com-0A66C2?style=for-the-badge" alt="Website: optinampout.com" /></a>
  <img src="https://img.shields.io/badge/Open_to-Data_Scientist_roles-2ea44f?style=for-the-badge" alt="Open to Data Scientist roles" />
</p>

<p align="center"><sub><i>Data scientist — de l'expérimentation à la mise en production.</i></sub></p>

---

## What I bring

- **Decisions you can trust before the A/B test.** On 13.7M real bandit logs, my IPW, SNIPW and doubly robust estimators landed within **5.3–5.8%** of the production policy's true CTR, and every 95% bootstrap confidence interval covered the truth. → [off-policy-bandit-evaluation](https://github.com/lyftium/off-policy-bandit-evaluation)
- **Computer vision on real engineering data, measured honestly.** For aerospace drawing review in a Safran context, my YOLO detector reached **74.0% mAP@50, 69.2% F1** on a 38-drawing OEM holdout. It's below the 0.82 target, so the release gate keeps it out of production while the agent platform ships. → GOLLAMC
- **Models that run as services.** I package models behind APIs and agents: a FastAPI LLaVA service for aerospace drawings (LlavaAero), 78 MCP tools across 3 servers (GOLLAMC), Docker, CI and MLflow-tracked retraining.

**Currently:** completing GOLLAMC, my final project for the Liora Data Scientist program (Mines Paris – PSL), and looking for my next Data Scientist role.

---

## Data science toolkit

| | |
|---|---|
| **Statistics & experimentation** | A/B test design, causal inference, off-policy evaluation (IPW, SNIPW, doubly robust, direct method), replay evaluation, bootstrap confidence intervals, rare-event (CTR) analysis, Monte Carlo simulation |
| **Modeling** | scikit-learn, gradient boosting, linear and logistic models, contextual bandits (LinUCB, Thompson sampling, ε-greedy), anomaly detection, leakage-free temporal splits |
| **CV, NLP & VLMs** | PyTorch, Hugging Face, YOLO object detection, Donut OCR, LLaVA fine-tuning with LoRA / QLoRA, local LLMs (Ollama, vLLM), LLM agents (LangGraph, MCP) |
| **Data** | SQL, pandas, NumPy, PySpark, Kafka streaming, ClickHouse, Elasticsearch, data-quality checks, labeling pipelines |
| **Production** | Docker, Kubernetes, FastAPI, MLflow, GitHub Actions, Linux, Prometheus / Grafana, dataset cards and evaluation contracts |

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/SQL-336791?style=flat-square&logo=postgresql&logoColor=white" alt="SQL" />
  <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="pandas" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Hugging_Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="Hugging Face" />
  <img src="https://img.shields.io/badge/PySpark-E25A1C?style=flat-square&logo=apachespark&logoColor=white" alt="PySpark" />
  <img src="https://img.shields.io/badge/MLflow-0194E2?style=flat-square&logo=mlflow&logoColor=white" alt="MLflow" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/Kubernetes-326CE5?style=flat-square&logo=kubernetes&logoColor=white" alt="Kubernetes" />
</p>

---

## Projects

### [Off-policy evaluation of recommendation policies on 13.7M real bandit logs](https://github.com/lyftium/off-policy-bandit-evaluation)
- **Problem:** can we know how a new recommender will perform *before* running an A/B test?
- **Approach:** on the public Open Bandit Dataset (ZOZOTOWN: 1.37M uniform-random and 12.4M production rounds), I compared the IPW, SNIPW, doubly robust and direct-method estimators against the production policy's known CTR, with bootstrap CIs and held-out days.
- **Result:** IPW, SNIPW and DR were within **5.3–5.8% relative error**, and every 95% CI covered the truth. The model-only direct method was precise but biased, and its CI missed. A per-slot popularity policy beat the logging policy on held-out days, **0.74% vs 0.36% CTR (~2.1×)**. The reward model's AUC was only 0.63, which explains why simple policies won.
- **Stack:** Python package, pandas, NumPy, scikit-learn, pytest, reproducible pipeline, MIT license.

### GOLLAMC: computer vision and LLM agents for aerospace drawing review *(private · demo on request)*
- **Problem:** manual GD&T / ASME Y14.5 review of dense aerospace drawings is slow and error-prone (Safran context).
- **Approach:** a 17-class YOLO symbol detector, Donut OCR, a LLaVA-1.5-7B fine-tuned with LoRA / QLoRA and rule-based ASME validators. They're combined in a LangGraph agent and exposed as **78 tools across 3 MCP servers**, all running locally. The data side has versioned label-quality datasets (v6 → v27), dataset cards, evaluation contracts and deduplicated holdout evaluation.
- **Result:** **74.0% mAP@50, 66.2% class-rollup and 69.2% F1** on a 38-drawing OEM holdout (v25), against a 0.82 mAP target. The platform ships, and the model release stays gated until it clears the target.
- **Stack:** PyTorch, Hugging Face (Transformers, PEFT, bitsandbytes), YOLO, Ollama, LangGraph, MCP, Streamlit, Docker, pytest, GitHub Actions, Kaggle GPUs.

### More
- **LlavaAero** *(private)*: a FastAPI service that runs a LLaVA vision-language model over aerospace drawings to extract design characteristics, with Docker builds, CI/CD and Prometheus / Grafana monitoring. GOLLAMC's predecessor.
- **Tolerance analyzer** *(private)*: statistical tolerance stack-up analysis in Jupyter, comparing worst-case, RSS and Monte Carlo methods.
- **Real-time blockchain data platform** *(private)*: Kafka → ClickHouse streaming, transaction labeling, anomaly detection, data-quality monitoring and bandit-based decision models tracked in MLflow.

---

<p align="center"><b>Hiring a data scientist?</b> Reach me through <a href="https://www.optinampout.com">optinampout.com</a>.</p>
