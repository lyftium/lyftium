<h1 align="center">Hi, I'm lyftium 👋</h1>
<p align="center"><b>Data Scientist</b> based in Paris, France</p>

<p align="center">
  <a href="https://www.optinampout.com"><img src="https://img.shields.io/badge/Website-optinampout.com-0A66C2?style=for-the-badge" alt="Website: optinampout.com" /></a>
  <img src="https://img.shields.io/badge/Open_to-Data_Scientist_roles-2ea44f?style=for-the-badge" alt="Open to Data Scientist roles" />
</p>

I like problems where the honest answer matters more than the impressive one. Most of my work sits between statistics and engineering: figuring out whether a model or a policy will actually work before it reaches users, and then shipping the ones that do as real services.

I've applied that to recommendation systems, computer vision for aerospace engineering drawings, and streaming data in fintech. Along the way I've learned to care as much about evaluation, data quality and reproducibility as about the model itself.

## A few things I've built

**[Off-policy evaluation of recommendation policies](https://github.com/lyftium/off-policy-bandit-evaluation)**

Can you tell how a new recommender will perform without running an A/B test? I tested that on the public Open Bandit Dataset from ZOZOTOWN, which has 13.7M real logged rounds. I compared IPW, self-normalized IPW, doubly robust and direct-method estimators against the production policy's known click-through rate.

The three importance-weighted estimators landed within 5.3–5.8% of the true CTR, and every 95% bootstrap interval covered it. The direct method looked precise but was biased, and its interval missed. A simple per-slot popularity policy beat the logging policy on held-out days (0.74% vs 0.36% CTR). The reward model only reached an AUC of 0.63, which explains why the simple policy won. Everything is a small, tested Python package you can rerun end to end.

**GOLLAMC: AI-assisted review of aerospace drawings** *(private, demo on request)*

Checking GD&T and ASME Y14.5 annotations on dense aerospace drawings by hand is slow and easy to get wrong. GOLLAMC combines a 17-class YOLO symbol detector, Donut OCR, and LLaVA-1.5-7B fine-tuned with LoRA and QLoRA, plus rule-based ASME checks. A LangGraph agent orchestrates them, exposed as 78 tools across three MCP servers, and everything runs locally.

On a 38-drawing OEM holdout the detector reaches 74.0% mAP@50 and 69.2% F1. That's short of the 0.82 mAP target, so the model stays behind a release gate until it gets there. Much of the work went into the data: versioned, deduplicated label sets with dataset cards and evaluation contracts, so every number is reproducible.

**Smaller projects** *(private)*

- **LlavaAero:** a FastAPI service that runs a LLaVA vision-language model over engineering drawings, with Docker, CI/CD and Prometheus/Grafana monitoring.
- **Tolerance analyzer:** statistical tolerance stack-ups comparing worst-case, RSS and Monte Carlo methods.
- **Real-time blockchain data platform:** Kafka to ClickHouse streaming, transaction labeling, anomaly detection, data-quality monitoring, and bandit-based decision models tracked in MLflow.

## What I work with

I'm strongest in **statistics and experimentation**: A/B test design, causal inference, off-policy evaluation, bootstrap intervals and Monte Carlo simulation. For **modeling** I use scikit-learn, gradient boosting, linear models and contextual bandits (LinUCB, Thompson sampling), and I'm careful about leakage-free temporal splits. On the **deep learning** side I work with PyTorch and Hugging Face: object detection, OCR, vision-language models with LoRA/QLoRA fine-tuning, local LLMs with Ollama and vLLM, and LLM agents with LangGraph and MCP.

For **data** I'm comfortable with SQL, pandas, PySpark, Kafka, ClickHouse and Elasticsearch. To **ship** things I use Docker, Kubernetes, FastAPI, MLflow, GitHub Actions and Linux.

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

## Let's talk

I'm looking for a data scientist role in Paris. If you're hiring, or just want to talk about evaluation, recommendation or computer vision, you can reach me through [optinampout.com](https://www.optinampout.com).
