# Candidature - Ingenieur integrateur de composants IA (Chateaufort)

## Lettre courte

Madame, Monsieur,

Je candidate au poste d ingenieur integrateur de composants IA. Mon fil rouge est celui de la mission: prendre des composants IA (detection, VLM, LoRA, agents), les industrialiser en micro-services, et les deployer sur un cluster de calcul avec qualite, securite et tracabilite.

Preuves publiques:

1. Industrialisation micro-services + cluster - https://github.com/lyftium/drawing-review-service
   FastAPI, Docker, Kubernetes GPU, overlay OpenShift (Route TLS + patch SCC restricted), manifests vLLM, gates d evaluation, lineage, Prometheus.
2. Fine-tuning LLM + entrainement distribue - https://github.com/lyftium/lora-finetune-recipe
   PEFT LoRA/QLoRA, baseline vs adapter, DDP via torchrun (preuve multi-process gloo; chemin CUDA nccl). YAML PyTorchJob pour Training Operator / OpenShift AI.
3. Metiers engineering / simulation (plus) - revue de plans techniques, travaux prives CAD/VLM aerospatiale (demo sur demande), integration PLM.
4. MLOps / tracabilite reglementaire - mapping ingenierie lineage -> oversight humain / logs / gates (docs/ai-act-mapping.md), sans sur-revendiquer une conformite juridique.

Disponible pour un echange technique en Ile-de-France.

Cordialement,
lyftium - contact@lyftium.com - https://github.com/lyftium

## Grille entretien (JD -> preuve)

| Attendu | Reponse 30s | Lien |
|---|---|---|
| Industrialiser composants en micro-services | Refactor, Docker, deploy K8s/OpenShift, gates avant promo | drawing-review-service |
| Engineering et simulation | Pipeline plan -> detect/OCR/VLM -> regles -> revue humain | idem |
| Enrichir le cluster | vLLM + gateway design + PyTorchJob DDP | deploy/openshift |
| Conseil environnements de calcul | Notes OpenShift SCC/Route, gateway, DDP vs FSDP | docs/openshift.md |
| Python / PyTorch / HF | LoRA recipe mesuree | lora-finetune-recipe |
| Kubernetes et OpenShift | Base K8s + overlay Route/SCC | deploy/openshift |
| vLLM / AI Gateway | Manifests vLLM + design gateway | docs/gateway.md |
| Train multi-GPU + fine-tune | DDP torchrun + LoRA; PyTorchJob | results/tiny-ddp |
| Agents en micro-services | Stages pluggables derriere une API | drawing-review |
| MLOps / exigences reglementaires | Lineage + gates + HITL + checklist | docs/ai-act-mapping.md |

## 30 premiers jours

1. Cartographier cluster (GPU, gateway, Training Operator, registries).
2. Reprendre 1-2 composants societe et les empaqueter au standard plateforme.
3. Brancher metriques + lineage au socle commun.

## Limites honestes (ne pas inventer)

- Submit live PyTorchJob sur cluster OpenShift AI du Groupe (YAML pret).
- Tuning serving multi-GPU a partir de SLO TTFT/p95 en prod.
- Gateway deployee (design pret).
