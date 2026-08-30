# SRAS: Sparse Reward-Aware Selector for Edge-Native RAG Pipelines

A production-grade, selector-only RL framework for document selection in retrieval-augmented generation (RAG), designed to run efficiently on resource-constrained edge hardware including the Raspberry Pi 4.

---

## What This System Does

SRAS trains a lightweight cross-attention selector using Proximal Policy Optimization (PPO) with sparse reward signals. Given a question and a pool of candidate documents, the selector learns to pick the top-k documents most likely to help a generator produce a correct answer, without retraining the generator. The generator (T5-base) is auxiliary and used only for reward computation during training and for evaluation; it is **not** the core contribution.

---

## Directory Structure

```
SRAS - ICEdge/
├── sras/                        # Core package
│   ├── config/                  # YAML config loading and typed dataclasses
│   ├── models/                  # CrossAttentionSelector definition
│   ├── rl/                      # PPOAgent and RewardComputer
│   ├── training/                # Supervised warmup and PPO trainer
│   ├── evaluation/              # Evaluator, metrics, benchmarking, robustness, failure analysis
│   ├── baselines/               # BM25, dense, hybrid, learned ranker selectors
│   ├── compression/             # Quantization, pruning, distillation, evaluator
│   ├── deployment/              # DeploymentProfiler, EdgeRunner
│   ├── data/                    # CorpusStore, datasets, embeddings, HF dataset loaders
│   ├── generator/               # T5-base GeneratorInterface
│   ├── analysis/                # PlotGenerator, SystemStoryLogger
│   └── auxiliary/               # E2EComparison (bounded selector contribution study)
├── configs/                     # YAML config files
├── train.py                     # Main training entry point
├── evaluate.py                  # Full evaluation with baselines and failure analysis
├── benchmark.py                 # Edge latency sweep with deployment profiling
├── setup_data.py                # Data pipeline (embed → QA → rewards → SQuAD)
├── run_ablations.py             # Core + expanded ablation study runner
├── run_compression.py           # Compression experiments (quantize/prune/distill)
├── run_robustness.py            # Robustness sweep (noise/redundant/adversarial/domain)
├── run_failure_analysis.py      # Failure breakdown (selector vs generator mistakes)
├── run_deployment.py            # Deployment profiling with psutil + thermal monitoring
└── requirements.txt
```

---

## Installation

### Standard machine (Linux/macOS/Windows WSL)

```bash
pip install -r requirements.txt
```

### Raspberry Pi 4 (see dedicated section below)

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

For Pi 4 specifically: use Python 3.9–3.11, install `torch` for ARM (`aarch64`). PyTorch does not publish official Pi wheels. Use the community builds at `https://torch.kmtea.eu/whl/stable.html` or build from source.

---

## Step 1: Prepare Your Data

Place your raw corpus in `data/flat_corpus.json`. The corpus should be a list of documents, each with at minimum a `"text"` field and optionally `"category"`, `"id"` etc.

Then run the full data setup pipeline:

```bash
python setup_data.py
```

This will:
1. Flatten and validate the corpus into `data/corpus_metadata.json`
2. Compute sentence-transformer embeddings → `data/doc_embeddings.pt`
3. Generate QA pairs from the corpus → `data/generated_qa_pairs.json`
4. Compute a reward matrix (relaxed F1 + BERTScore) → `data/reward_matrix.json`
5. Download and cache a SQuAD dev subset → `data/squad_dev_subset.json`

This step can take 30–60 minutes depending on corpus size and hardware. It only needs to be run once.

---

## Step 2: Train the Selector

### Full pipeline (supervised warmup → PPO)

```bash
python train.py --config configs/base.yaml --mode full
```

### Supervised warmup only

```bash
python train.py --config configs/base.yaml --mode supervised
```

### PPO only (assumes supervised checkpoint exists)

```bash
python train.py --config configs/base.yaml --mode ppo
```

### Key training flags

| Flag | Description |
|---|---|
| `--config` | Path to YAML config (default: `configs/base.yaml`) |
| `--mode` | `supervised`, `ppo`, or `full` |
| `--device` | `cpu`, `cuda`, or `auto` |
| `--no-supervised-warmup` | Disable warmup (ablation) |
| `--no-reward-shaping` | Disable BERTScore in reward (ablation) |
| `--no-curriculum` | Disable curriculum learning (ablation) |

Checkpoints are saved to `models/`. The best model (highest average PPO reward) is saved as `models/sras_selector_ppo_base.pt`.

---

## Step 3: Evaluate

```bash
python evaluate.py --config configs/base.yaml
```

This runs all variants in `benchmark.model_registry` against all datasets in `evaluation.datasets` (default: `internal` + `squad`). It also runs BM25, dense, and hybrid baselines for comparison, and produces per-question-type F1 breakdowns.

### Key evaluation flags

| Flag | Description |
|---|---|
| `--datasets internal squad` | Datasets to evaluate on |
| `--pool-size 30` | Candidate pool size per query |
| `--top-k 3` | Number of documents to select |
| `--noise 0.2` | Inject 20% noisy documents into candidate pool |
| `--redundant 0.1` | Inject 10% redundant documents |
| `--adversarial 0.1` | Inject 10% adversarial documents |
| `--no-baselines` | Skip BM25/dense/hybrid baselines |
| `--no-plots` | Skip PDF plot generation |

Results are saved to `results/` and `figures/sras_model_eval/`.

---

## Step 4: Run Ablations

### Core ablations (warmup, reward shaping, curriculum)

```bash
python run_ablations.py --config configs/base.yaml
```

### Expanded ablations (top-k, pool size, reward weights, warmup length, curriculum schedule)

```bash
python run_ablations.py --config configs/base.yaml --expanded
```

### Run specific variants only

```bash
python run_ablations.py --variants sras_ppo_base sras_ppo_nosw ablation_topk_1
```

Available variants for `--expanded`:

| Variant | What it tests |
|---|---|
| `sras_ppo_base` | Full model (baseline) |
| `sras_ppo_nosw` | No supervised warmup |
| `sras_ppo_nors` | No reward shaping (F1 only) |
| `sras_ppo_nocl` | No curriculum learning |
| `ablation_topk_1` | top-k = 1 |
| `ablation_topk_5` | top-k = 5 |
| `ablation_pool_10` | Pool size = 10 |
| `ablation_pool_50` | Pool size = 50 |
| `ablation_reward_0507` | F1 weight=0.5, BERTScore weight=0.7 |
| `ablation_reward_0802` | F1 weight=0.8, BERTScore weight=0.2 |
| `ablation_warmup_50` | Supervised warmup for 50 epochs |
| `ablation_warmup_100` | Supervised warmup for 100 epochs |
| `ablation_curriculum_linear` | Linear curriculum schedule |
| `ablation_curriculum_fixed` | Fixed pool size throughout training |

---

## Step 5: Robustness Evaluation

```bash
python run_robustness.py --config configs/base.yaml --pool-size 30 --top-k 3
```

Sweeps noise rates `[0.0, 0.1, 0.2, 0.3, 0.5]`, redundant rates `[0.0, 0.1, 0.2, 0.3]`, and adversarial rates `[0.0, 0.1, 0.2]`, each averaged over 3 trials. Also tests domain shift across corpus categories (`Law And Society`, `Technology and Cybersecurity`, `Philosophy And Ethics`). Results go to `results/robustness/robustness_results.json`. Plots: `figures/sras_model_eval/robustness_sweep.pdf` and `domain_shift.pdf`.

To customize what is swept, edit the `evaluation.robustness` block in `configs/base.yaml`.

---

## Step 6: Failure Analysis

```bash
python run_failure_analysis.py --config configs/base.yaml --pool-size 30 --top-k 3
```

Separates failures into:
- **Selector failures**: selected documents did not contain the answer
- **Generator failures**: answer was reachable in selected docs but generator missed it

Produces per-question-type breakdown (what/who/when/where/how/why/which) with failure rates and representative cases. Results go to `results/failure_analysis/failure_analysis.json`. Plot: `figures/sras_model_eval/failure_breakdown.pdf`.

---

## Step 7: Compression Experiments

### Quantize (INT8)

```bash
python run_compression.py --config configs/base.yaml --quantize
```

### Prune (global magnitude pruning, 30% sparsity)

```bash
python run_compression.py --config configs/base.yaml --prune --prune-amount 0.3
```

### Knowledge distillation (student with smaller hidden dim)

```bash
python run_compression.py --config configs/base.yaml --distill
```

### All three together

```bash
python run_compression.py --config configs/base.yaml --quantize --prune --distill
```

Results go to `models/compressed/` and `compression_results.json`. Plot: `compression_comparison.pdf`.

---

## Step 8: Edge Latency Benchmark

```bash
python benchmark.py --config configs/base.yaml
```

Sweeps pool sizes `[10, 30, 50, 100]` and reports p50/p95/p99 latency, memory delta, parameter count, and model size for all registered model variants. Also runs `DeploymentProfiler` for RAM (via `psutil`), throughput at multiple batch sizes, CPU energy proxy, and thermal monitoring. Results: `figures/sras_model_eval/edge_benchmark.json` and `deployment_profile.json`.

---

## Step 9: Full Deployment Profiling

```bash
python run_deployment.py --config configs/base.yaml
```

### Print hardware info only

```bash
python run_deployment.py --hardware-info
```

This detects whether it is running on a Raspberry Pi (via `/proc/cpuinfo` and architecture), reports RAM, CPU cores, and thermal readings. Results go to `results/deployment/`.

---

## Raspberry Pi 4: Complete Setup Guide

### Hardware requirements

- Raspberry Pi 4 Model B, 4GB or 8GB RAM recommended
- 32GB+ microSD (Class 10 / A2) or USB SSD for better I/O
- Active cooling recommended (the thermal profiler will report throttling)

### OS and Python

```bash
# Install Raspberry Pi OS 64-bit (Bookworm recommended)
# Then:
sudo apt update && sudo apt install -y python3-pip python3-venv git
python3 -m venv sras_env
source sras_env/bin/activate
```

### Install PyTorch for ARM64

PyTorch does not publish official Pi 4 wheels. Use the community ARM builds:

```bash
pip install torch --index-url https://torch.kmtea.eu/whl/stable.html
# Or build from source (takes 2–4 hours):
# https://github.com/pytorch/pytorch#from-source
```

### Install remaining dependencies

```bash
pip install transformers sentence-transformers bert-score numpy tqdm pyyaml matplotlib pandas rank-bm25 psutil datasets
```

### Transfer the codebase

```bash
# From your development machine:
rsync -avz "SRAS - ICEdge/" pi@<PI_IP>:~/sras/
```

### Run the selector on Pi (inference only)

The Pi should **not** be used for training; that should be done on a GPU machine and the checkpoint transferred. On the Pi, you run evaluation and profiling:

```bash
# On Pi:
cd ~/sras
python run_deployment.py --config configs/base.yaml --force-cpu
```

### Pi-specific config override

Add this block to your `configs/base.yaml` when running on Pi:

```yaml
benchmark:
  deployment:
    is_raspberry_pi: true
    n_iterations: 50        # reduce for faster runs
    measure_thermal: true   # reads /sys/class/thermal/thermal_zone0/temp
```

### Read thermal data from vcgencmd (Pi-specific)

The `DeploymentProfiler` automatically tries `vcgencmd measure_temp` when `is_raspberry_pi: true`. Make sure it is available:

```bash
sudo apt install -y libraspberrypi-bin
vcgencmd measure_temp  # should return temp=XX.X'C
```

### Run the full edge benchmark on Pi

```bash
python benchmark.py --config configs/base.yaml --device cpu --iterations 50
python run_deployment.py --config configs/base.yaml --hardware-info
python run_failure_analysis.py --config configs/base.yaml
```

### What the Pi profiler reports

- **RAM**: peak RSS memory in MB via `psutil`
- **Latency**: p50/p95/p99 in milliseconds over N iterations
- **Throughput**: queries/second at batch sizes 1, 4, 8, 16
- **Energy proxy**: CPU time × CPU count × estimated watts
- **Thermal**: peak and mean temperature via `/sys/class/thermal` and `vcgencmd`
- **Hardware info**: whether it detected a Pi, RAM total, CPU count, model string

All results are saved to `results/deployment/` in JSON.

---

## Multilingual Evaluation (TriviaQA / NaturalQuestions / MLQA)

Enable benchmark datasets in `configs/base.yaml`:

```yaml
evaluation:
  benchmark_datasets:
    use_triviaqa: true
    triviaqa_subset_n: 500
    use_natural_questions: true
    nq_subset_n: 500
    use_mlqa: true
    mlqa_languages: [en, de, es, ar, hi, zh, vi]
    mlqa_subset_n: 200
    cache_dir: data/hf_cache
```

Then run:

```bash
python evaluate.py --config configs/base.yaml --datasets internal squad triviaqa nq mlqa_en mlqa_de
```

The HuggingFace `datasets` library will download and cache the data on first run. Subsequent runs use the local cache at `data/hf_cache/`.

---

## End-to-End Selector Contribution Study

To verify SRAS's contribution is in the selector (not the generator):

```python
from sras.auxiliary.e2e_comparison import E2EComparison

cmp = E2EComparison(output_dir="results/e2e_comparison")
results = cmp.compare(
    sras_model=model,
    questions=questions,
    golds=golds,
    corpus_docs=corpus_docs,
    doc_embs=doc_embs,
    q_embs=q_embs,
    generator=generator,
    baselines={"bm25": bm25_selector, "dense": dense_selector},
)
```

All methods share the same generator; only the selector changes. Results are saved to `results/e2e_comparison/e2e_comparison.json`.

---

## System Story Logging (for paper claims)

```python
from sras.analysis.system_story import SystemStoryLogger

story = SystemStoryLogger(output_dir="results/system_story")

story.log_edge_friendliness(
    model_params=model.count_parameters(),
    model_size_mb=2.1,
    p50_latency_ms=12.4,
    p99_latency_ms=18.7,
    ram_mb=94.0,
    device="cpu",
    is_raspberry_pi=True,
)

story.log_selector_contribution(
    sras_f1=0.612,
    bm25_f1=0.481,
    dense_f1=0.534,
    hybrid_f1=0.558,
    random_f1=0.221,
    oracle_f1=0.784,
)

story.save()  # → results/system_story/system_story.json
```

---

## Config System

All scripts accept `--config path/to/config.yaml`. The YAML is deep-merged with an optional overrides dict, so ablation configs only need to specify the fields they change. Example:

```yaml
# configs/ablation_topk_1.yaml
name: ablation_topk_1
training:
  top_k: 1
evaluation:
  top_k: 1
```

The full schema is defined in `sras/config/schema.py` as typed Python dataclasses.

---

## Outputs Reference

| Output | Location |
|---|---|
| Trained checkpoints | `models/` |
| PPO training logs | `logs/ppo_training_log_*.json` |
| Evaluation results | `results/*.json` |
| Ablation summary | `figures/sras_model_eval/ablation_eval_summary.json` |
| Robustness results | `results/robustness/robustness_results.json` |
| Failure analysis | `results/failure_analysis/failure_analysis.json` |
| Compression results | `models/compressed/compression_results.json` |
| Deployment profile | `results/deployment/deployment_profile.json` |
| All plots (PDF) | `figures/sras_model_eval/*.pdf` |
| System story | `results/system_story/system_story.json` |
| HF dataset cache | `data/hf_cache/` |
