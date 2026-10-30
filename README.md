<h1 align="center">Hi, I'm LYFTIUM 👋</h1>
<h3 align="center">Data Scientist · ML @ Safran · San Francisco</h3>

<p align="center">
  <a href="https://github.com/lyftium/off-policy-bandit-evaluation"><img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=20&pause=1000&color=2F81F7&center=true&vCenter=true&width=640&lines=Causal+inference+%26+off-policy+evaluation;Recommendation+systems+%26+contextual+bandits;Vision-language+models+%26+LLM+agents;Streaming+data+pipelines+%E2%86%92+production+ML" alt="Typing SVG" /></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Open_to-Data_Scientist_roles-2ea44f?style=for-the-badge" />
  <a href="https://www.optinampout.com"><img src="https://img.shields.io/badge/Website-optinampout.com-0A66C2?style=for-the-badge" /></a>
</p>

---

### 🧠 About me
I turn messy, high-volume data into decisions you can trust. I work across the full stack of data science: **experiment design and causal inference**, **machine learning and recommendation**, **vision-language models**, and the **streaming pipelines and MLOps** that put models into production.

- 🔬 Estimating how a new policy will perform *before* the A/B test, with honest uncertainty
- 🛩️ Building ML and vision-language tools for aerospace engineering at Safran
- ⚡ Running real-time data platforms: Kafka → ClickHouse → labeled training data → MLflow-tracked models
- ☸️ Shipping models with Docker, Kubernetes and CI/CD

---

### ⭐ Featured project: [Off-Policy Evaluation of Recommendation Policies on 13.7M Real Bandit Logs](https://github.com/lyftium/off-policy-bandit-evaluation)
*Can we know how a new recommender performs **before** running an A/B test?* I used the public Open Bandit Dataset (ZOZOTOWN): 1.37M uniform-random rounds and 12.4M production rounds.

| Result | Number |
|---|---|
| Error of IPW, SNIPW and doubly robust estimates of the production policy's true CTR | **5.3–5.8% relative**, and all 95% bootstrap CIs covered the truth |
| Model-only Direct Method | Precise but biased: its CI missed the truth |
| Per-slot popularity policy vs. the logging policy, on held-out days | **0.74% vs 0.36% CTR (~2.1×)** |
| Reward model AUC | 0.63, which is why simpler policies won here |

The repo includes a Python package, a pytest suite, GitHub Actions CI, a reproducible pipeline and an MIT license.

---

### 🛠️ Skills

| Area | What I use |
|---|---|
| 📊 **Experimentation & causal inference** | A/B test design, off-policy evaluation (IPW, SNIPW, doubly robust, direct method), replay evaluation, bootstrap CIs, leakage-free temporal splits |
| 🤖 **Machine learning** | Gradient boosting, linear and logistic models, contextual bandits (LinUCB, Thompson sampling, ε-greedy), anomaly detection, labeling pipelines |
| 👁️ **Vision-language models & GenAI** | VLMs for engineering drawings (LLaVA-style), local LLMs, LLM agents, Model Context Protocol (MCP) servers, multi-agent workflows |
| 📈 **Statistics & analysis** | Rare-event (CTR) analysis, uncertainty quantification, statistical tolerance stack-up analysis, EDA, data quality |
| 🔄 **Data engineering** | SQL, ClickHouse, Kafka / Redpanda streaming, PySpark, Elasticsearch, pandas, NumPy, real-time ingestion and backfills |
| 🚀 **MLOps & cloud** | MLflow tracking and registry, scheduled retraining, Docker, Kubernetes, GitHub Actions CI/CD, Linux and systemd, monitoring and alerting |
| ⛓️ **Blockchain data** | Ethereum node data, on-chain transaction analytics, real-time event detection and labeling |
| 🏭 **Domains** | Aerospace engineering (Safran), recommendation systems, fintech and on-chain markets |

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/SQL-336791?style=flat-square&logo=postgresql&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Hugging_Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white" />
  <img src="https://img.shields.io/badge/PySpark-E25A1C?style=flat-square&logo=apachespark&logoColor=white" />
  <img src="https://img.shields.io/badge/Kafka-231F20?style=flat-square&logo=apachekafka&logoColor=white" />
  <img src="https://img.shields.io/badge/ClickHouse-FFCC01?style=flat-square&logo=clickhouse&logoColor=black" />
  <img src="https://img.shields.io/badge/Elasticsearch-005571?style=flat-square&logo=elasticsearch&logoColor=white" />
  <img src="https://img.shields.io/badge/MLflow-0194E2?style=flat-square&logo=mlflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/Kubernetes-326CE5?style=flat-square&logo=kubernetes&logoColor=white" />
  <img src="https://img.shields.io/badge/GitHub_Actions-2088FF?style=flat-square&logo=githubactions&logoColor=white" />
  <img src="https://img.shields.io/badge/Linux-FCC624?style=flat-square&logo=linux&logoColor=black" />
  <img src="https://img.shields.io/badge/Rust-000000?style=flat-square&logo=rust&logoColor=white" />
  <img src="https://img.shields.io/badge/TypeScript-3178C6?style=flat-square&logo=typescript&logoColor=white" />
</p>

---

### 📁 More work
- **Real-time market-data research platform** *(private)*: Kafka → ClickHouse streaming pipelines, transaction labeling and anomaly-detection models, data-quality monitoring, and bandit-based decision models tracked in MLflow.
- **LlavaAero / GOLLAMC** *(private)*: vision-language models and local-LLM agents for aerospace 2D and CAD drafting.
- **Tolerance analyzer** *(private)*: statistical tolerance stack-up analysis in Jupyter.
- **pyspark-mcp / elasticsearch** *(private)*: PySpark and Elasticsearch data tooling for LLM agents.
- [youtube-mcp-server](https://github.com/LYFTIUM-INC/youtube-mcp-server): YouTube analytics and content-analysis service (Python).
- [10x-agentic-setup](https://github.com/LYFTIUM-INC/10x-agentic-setup): Python environment for LLM-agent development.

---

### 📊 GitHub stats
<p align="center">
  <img height="165" src="https://github-readme-stats.vercel.app/api?username=lyftium&show_icons=true&include_all_commits=true&count_private=true&theme=github_dark&hide_border=true" />
  <img height="165" src="https://github-readme-stats.vercel.app/api/top-langs/?username=lyftium&layout=compact&theme=github_dark&hide_border=true" />
</p>

<p align="center"><i>Open to Data Scientist roles. Reach me through <a href="https://www.optinampout.com">optinampout.com</a> or GitHub.</i></p>
