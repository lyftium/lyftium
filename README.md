<h1 align="center">Hi, I'm LYFTIUM 👋</h1>
<h3 align="center">AI Platform & MLOps Engineer · Data Scientist · Aerospace AI (Safran)</h3>

<p align="center">
  <a href="https://github.com/lyftium"><img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=20&pause=1000&color=2F81F7&center=true&vCenter=true&width=680&lines=Industrializing+AI+components+as+microservices;LLM+%26+VLM+serving%2C+fine-tuning+(LoRA%2FQLoRA);Agentic+AI+on+MCP+for+aerospace+engineering;Causal+inference+%26+off-policy+evaluation" alt="Typing SVG" /></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Open_to-AI_Platform_%7C_MLOps_%7C_Data_Science_roles-2ea44f?style=for-the-badge" />
  <a href="https://www.optinampout.com"><img src="https://img.shields.io/badge/Website-optinampout.com-0A66C2?style=for-the-badge" /></a>
</p>

<p align="center"><i>🇫🇷 J'industrialise des composants IA (LLM, VLM, agents) : du fine-tuning au déploiement conteneurisé, avec une gouvernance MLOps traçable.</i></p>

---

### 🧠 About me
I take AI components from notebook to production: **refactor and package them as containerized microservices**, **serve and fine-tune LLMs and vision-language models**, wire them into **agentic workflows**, and keep every model release **traceable, evaluated and human-gated**. I'm also a data scientist at heart: **experiment design, causal inference and recommendation**.

- 🛩️ Built **GOLLAMC**, a local agentic platform for aerospace drawing review, in a Safran context
- 🧩 Shipping AI as services: Docker images, MCP tool servers, FastAPI/Streamlit front-ends, Kubernetes deployments
- 🎯 Fine-tuning VLMs (LLaVA-1.5-7B with LoRA/QLoRA) and detectors (YOLO), with dataset cards and evaluation contracts
- ⚡ Running real-time data platforms: Kafka → ClickHouse → labeled training data → MLflow-tracked models
- 🔬 Estimating how a new policy will perform *before* the A/B test, with honest uncertainty

---

### 🛩️ Featured project: GOLLAMC — agentic AI for aerospace drawing review *(private · demo on request)*
**G**oose + **O**llama + **LL**aVA + **A**erospace **M**CP **C**AD. A locally hosted platform that ingests engineering PDFs and drawings, checks them against ASME Y14.5 / GD&T rules, drafts FAI and ballooned DXF, generates 3D STEP/STL, and keeps an evidence ledger, with **no cloud dependency on the core path**. Final project (*PFE*) of my Data Scientist training (Liora · Mines Paris – PSL), framed with Safran.

| Layer | What's in it |
|---|---|
| 🤖 **Agents as microservices** | **3 MCP servers exposing 78 tools** (vision/ASME, CadQuery 3D, AutoCAD 2D), orchestrated by Goose and a **LangGraph** pipeline |
| 🧠 **Models** | Local LLMs/VLMs via **Ollama** (LLaVA, MiniCPM-V, Qwen 2.5), **YOLO11** symbol detector, **Donut** OCR, rule-based ASME validators |
| 🎯 **Training** | **LoRA / QLoRA fine-tuning of LLaVA-1.5-7B** (PEFT, bitsandbytes 4-bit, DeepSpeed launch config), versioned YOLO fine-tunes (v6 → v27) on Kaggle GPUs |
| 📦 **Packaging** | Dockerfile + devcontainer, Makefile targets, pytest suite, GitHub Actions (lint, tests, agent-manifest validation) |
| 🛡️ **Governance** | Dataset cards and versioned lineage, evaluation contracts with holdout provenance, **fail-closed release gates**: the agent platform ships, but **model release stays blocked** until human-gold and holdout floors are met |

> Honest numbers, straight from the repo: the YOLO detector reached **74.0% mAP@50** on a 38-drawing OEM holdout (v25) against a **0.82 target**, so it is *not* promoted to release.

---

### ⭐ Data science showcase: [Off-Policy Evaluation of Recommendation Policies on 13.7M Real Bandit Logs](https://github.com/lyftium/off-policy-bandit-evaluation)
*Can we know how a new recommender performs **before** running an A/B test?* I used the public Open Bandit Dataset (ZOZOTOWN): 1.37M uniform-random rounds and 12.4M production rounds.

| Result | Number |
|---|---|
| Error of IPW, SNIPW and doubly robust estimates of the production policy's true CTR | **5.3–5.8% relative**, and all 95% bootstrap CIs covered the truth |
| Model-only Direct Method | Precise but biased: its CI missed the truth |
| Per-slot popularity policy vs. the logging policy, on held-out days | **0.74% vs 0.36% CTR (~2.1×)** |
| Reward model AUC | 0.63, which is why simpler policies won here |

The repo includes a Python package, a pytest suite, a reproducible pipeline and an MIT license.

---

### ☸️ AI infrastructure & LLM serving

| Area | What I use |
|---|---|
| 🐳 **Containers & orchestration** | Docker image builds, docker-compose, devcontainers, Kubernetes deployments and rollouts, Linux, shell scripting, systemd |
| 🚀 **LLM / VLM inference** | vLLM, Ollama, Hugging Face Transformers, 4-bit quantization (bitsandbytes), local-first serving behind API services |
| 🎯 **Training & fine-tuning** | PyTorch, LoRA / QLoRA with PEFT, DeepSpeed, GPU-constrained training (batch / accumulation / resolution tuning), YOLO fine-tuning |
| 🤖 **Agentic AI** | Model Context Protocol (MCP) tool servers, LangGraph, Goose, multi-agent workflows, each tool server deployable as its own service |
| 🔁 **MLOps** | MLflow tracking and registry, scheduled retraining, GitHub Actions CI, Prometheus / Grafana monitoring, data-quality checks |
| 🛡️ **Model governance** | Dataset and model documentation, data lineage and versioning, evaluation contracts on held-out data, human-in-the-loop approval, fail-closed release gates, audit trails |

### 🛠️ Data science skills

| Area | What I use |
|---|---|
| 📊 **Experimentation & causal inference** | A/B test design, off-policy evaluation (IPW, SNIPW, doubly robust, direct method), replay evaluation, bootstrap CIs, leakage-free temporal splits |
| 🤖 **Machine learning** | Gradient boosting, linear and logistic models, contextual bandits (LinUCB, Thompson sampling, ε-greedy), anomaly detection, labeling pipelines |
| 👁️ **Computer vision & VLMs** | LLaVA-style VLMs for engineering drawings, object detection (YOLO), document OCR (Donut), GD&T / ASME Y14.5 symbol taxonomy |
| 📈 **Statistics** | Rare-event (CTR) analysis, uncertainty quantification, statistical tolerance stack-up analysis, EDA, data quality |
| 🔄 **Data engineering** | SQL, ClickHouse, Kafka / Redpanda streaming, PySpark, Elasticsearch, pandas, NumPy, real-time ingestion and backfills |
| 🏭 **Domains** | Aerospace engineering (Safran), recommendation systems, fintech and on-chain markets |

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Hugging_Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/vLLM-30A2FF?style=flat-square" />
  <img src="https://img.shields.io/badge/Ollama-000000?style=flat-square&logo=ollama&logoColor=white" />
  <img src="https://img.shields.io/badge/LangGraph-1C3C3C?style=flat-square&logo=langchain&logoColor=white" />
  <img src="https://img.shields.io/badge/MCP-6E56CF?style=flat-square" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/Kubernetes-326CE5?style=flat-square&logo=kubernetes&logoColor=white" />
  <img src="https://img.shields.io/badge/MLflow-0194E2?style=flat-square&logo=mlflow&logoColor=white" />
  <img src="https://img.shields.io/badge/GitHub_Actions-2088FF?style=flat-square&logo=githubactions&logoColor=white" />
  <img src="https://img.shields.io/badge/Prometheus-E6522C?style=flat-square&logo=prometheus&logoColor=white" />
  <img src="https://img.shields.io/badge/Linux-FCC624?style=flat-square&logo=linux&logoColor=black" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/SQL-336791?style=flat-square&logo=postgresql&logoColor=white" />
  <img src="https://img.shields.io/badge/PySpark-E25A1C?style=flat-square&logo=apachespark&logoColor=white" />
  <img src="https://img.shields.io/badge/Kafka-231F20?style=flat-square&logo=apachekafka&logoColor=white" />
  <img src="https://img.shields.io/badge/ClickHouse-FFCC01?style=flat-square&logo=clickhouse&logoColor=black" />
  <img src="https://img.shields.io/badge/Elasticsearch-005571?style=flat-square&logo=elasticsearch&logoColor=white" />
</p>

---

### 📁 More work
- **LlavaAero** *(private)*: FastAPI service that runs a LLaVA vision-language model over aerospace drawings to extract design characteristics, with Docker builds, CI/CD workflows and Prometheus/Grafana monitoring config. GOLLAMC's predecessor.
- **Real-time market-data research platform** *(private)*: Kafka → ClickHouse streaming pipelines, transaction labeling and anomaly-detection models, data-quality monitoring, and bandit-based decision models tracked in MLflow.
- **Tolerance analyzer** *(private)*: statistical tolerance stack-up analysis in Jupyter.
- **pyspark-mcp / smarteam-mcp-server** *(private)*: MCP servers that give LLM agents PySpark data tooling and ENOVIA SmarTeam PLM access.
- [youtube-mcp-server](https://github.com/LYFTIUM-INC/youtube-mcp-server): MCP server for YouTube analytics and content analysis (Python).
- [10x-agentic-setup](https://github.com/LYFTIUM-INC/10x-agentic-setup): Python environment for LLM-agent development.

---

### 📊 GitHub stats
<p align="center">
  <img height="165" src="https://github-readme-stats.vercel.app/api?username=lyftium&show_icons=true&include_all_commits=true&count_private=true&theme=github_dark&hide_border=true" />
  <img height="165" src="https://github-readme-stats.vercel.app/api/top-langs/?username=lyftium&layout=compact&theme=github_dark&hide_border=true" />
</p>

<p align="center"><i>Open to AI platform, MLOps and data science roles · Ouvert aux postes d'ingénieur IA / MLOps en France. Reach me through <a href="https://www.optinampout.com">optinampout.com</a> or GitHub.</i></p>
