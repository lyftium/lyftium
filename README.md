# lyftium

ML / software in Paris. I mostly work on taking models out of notebooks and into services that other people can actually run: APIs, containers, GPU serving, and evaluation that is not optional.

French and English.

**contact@lyftium.com** · [optinampout.com](https://www.optinampout.com)

## Projects

### [drawing-review-service](https://github.com/lyftium/drawing-review-service)

Turned a drawing-review prototype into a small production-shaped service: detect, OCR, vision-language model on vLLM, rule checks, then human review. FastAPI, Docker, Kubernetes (plus an OpenShift overlay), Prometheus, release gates, run lineage.

Public repo uses stub backends so it runs without GPUs. The real stack is wired the same way.

### [lora-finetune-recipe](https://github.com/lyftium/lora-finetune-recipe)

A short LoRA / QLoRA loop I can re-run: baseline, train adapter, score on the same held-out set, write a report with git sha and config/data hashes. CPU smoke path, plus multi-process DDP via torchrun.

### Other work (private, happy to walk through)

- Local CAD / drawing agents for engineering workflows
- PLM-side tooling
- Occasional data and infra work

## Tools I reach for

Python, PyTorch, Hugging Face, FastAPI, Docker, Kubernetes, Linux, vLLM, Prometheus, GitHub Actions.

## Education

Data Scientist program, Liora with Mines Paris – PSL
