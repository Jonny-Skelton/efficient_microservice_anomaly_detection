# MESTGAD project

- Conda env: `mestgad-mamba` (PyTorch 2.0.1+cu118, mamba-ssm 1.2.0)
- Activation pattern: `module load anaconda && source activate mestgad-mamba && module load cuda/11.8 && export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH`
- Currently fighting mamba-ssm import chain dragging in transformers (see init.py patch)
- AIOps dataset unavailable; MSDS-only analysis
- Don't run heavy compute on login node — submit via sbatch
- result.log + results/ have ablation data; join_results.py builds the joined CSV

# MESTGAD Project — Handoff Context

## Project summary

Graduate course project extending **MSTGAD** (Multi-modal Spatial-Temporal Graph
Anomaly Detection for microservice systems, ASE 2023) into **MESTGAD**. Two
architectural contributions:

1. **Temporal Attention Module (TAM) → MambaTemporalModule.** Replaces the
   quadratic-in-W multi-head attention with per-modality Mamba selective state
   space blocks plus a linear cross-modal mixing layer. Goal: linear-time O(W)
   temporal encoding so longer context windows become tractable.
2. **Association discrepancy scoring anchored on the SAM** (Spatial Attention
   Module). Supplements the reconstruction-error-based anomaly score with a
   KL-divergence term between a learned prior and observed SAM attention
   distributions. Rationale: catches distributional shifts in inter-service
   relationships (cascading failures) that pure reconstruction error misses.

Loss: `L = (1/epoch)*L1 + (1-1/epoch)*L2 + lambda_ad*L_ad`, where L1/L2 are
MSTGAD's original semi-supervised reconstruction and classification losses.

Implementation lives in two files: `MESTGAD.py` (top-level model) and
`MESTGAD_util.py` (encoder/decoder/temporal/attention modules). These drop
into the upstream MSTGAD repo as replacements for `src/model.py` and
`src/model_util.py`.

## Infrastructure

- **Cluster**: HCC Swan (SLURM). Target GPU: **L40S 48GB** (13 nodes × 4 GPUs).
  All jobs pin `--constraint='gpu_l40s&gpu_48gb'`.
- **Working directory**: `/work/helab/jonnyskel/efficient_microservice_anomaly_detection/`
  (cloned from https://github.com/alipay/microservice_system_twin_graph_based_anomaly_detection).
- **Conda env**: `mestgad` (Python 3.8, PyTorch 1.12.0+cu113, torch-geometric 2.2.0,
  torch-scatter 2.1.0, torch-sparse 0.6.16). Env activation in sbatch scripts requires:
  ```bash
  module load anaconda
  eval "$(conda shell.bash hook)"
  conda activate mestgad
  module load cuda
  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
  ```

## Datasets

- **MSDS** (concurrent_data variant from Zenodo 3549604): ~5 service instances,
  downloaded and preprocessed successfully via `util/pre_MSDS.py`. All MSDS
  training runs work.
- **AIOps-Challenge**: paper uses it but dataset is not openly distributed
  (CCF AIOps Challenge 2022 requires competition registration). **Not available
  for this project.** Every AIOps task in the sweeps fails in ~10 seconds with
  FileNotFoundError. Writeup will be MSDS-only with explicit acknowledgment.
  This matches the paper itself, whose ablation study (Table IV) is also
  MSDS-only.

## Experimental design

Four SLURM array jobs, all targeting L40S with 1 GPU per task:

| Script | Array shape | Purpose |
|---|---|---|
| `01_detection_quality.sbatch` | 4 configs × 2 datasets × 5 seeds = 40 | Main ablation table |
| `02_complexity_profile.sbatch` | single job, exclusive node | TAM vs Mamba complexity (headline figure) |
| `03_window_scaling.sbatch` | 2 models × 5 windows × 2 datasets × 3 seeds = 60 | F1 vs W scaling |
| `04_lambda_sensitivity.sbatch` | 6 λ × 2 datasets × 3 seeds = 36 | AD loss weight sensitivity |

Four ablation configs in script 01 correspond to the 4 corners of
(use_mamba, use_ad):
- `mstgad`: TAM + reconstruction only (baseline)
- `mstgad_mamba`: Mamba + reconstruction only
- `mstgad_ad`: TAM + reconstruction + AD
- `mestgad`: Mamba + AD (full system)

## Current status of jobs

**Completed runs (what's analyzable):**

- **Detection (01)**: MSDS half completed successfully (20 runs: 4 configs × 5 seeds).
  AIOps half all fast-failed. Results in `result.log` (appended by every training
  run) and `results/detect/<config>_msds_seed<n>/`. Baseline task 0 (mstgad,
  MSDS, seed 0) hit **F1 = 0.977**, closely matching the paper's reported 0.957
  on MSDS — confirms pipeline correctness.

- **Profile (02)**: **Broken, needs rerun.** First run had wrong module signatures
  and all 16 cells errored (signature mismatch — used wrong class names/kwargs).
  Fixed `profile_temporal.py` produced in last turn. Old broken output should be
  moved aside before rerunning.

- **Window sweep (03)**: Partial. MSDS tasks completed where they fit in memory.
  Observed pattern: mstgad OOMs at W≥80 on MSDS (fails in ~30s before training
  starts). mestgad completed for W∈{10,20,40} MSDS. So we have:
  - mstgad: W∈{10,20,40} × MSDS × 3 seeds (9 runs)
  - mestgad: W∈{10,20,40} × MSDS × 3 seeds (9 runs), and possibly W=80,160
    depending on which of tasks 30–32 / 31 covered which cells

  The mstgad W≥80 OOM is itself a meaningful finding ("baseline TAM cannot fit
  W=80 even on 48GB, MESTGAD can") and should be reported as such.

- **Lambda sweep (04)**: 18 MSDS runs completed across all 6 λ values × 3 seeds.
  Clean sensitivity curve available. AIOps half all fast-failed.

## Files that matter for analysis

| File | Contents |
|---|---|
| `result.log` | One-line-per-evaluation, appended by every training run. Contains PR/RC/AUC/AP/F1 for both "best-loss" and "best-F1" checkpoints. **Must be joined back to (config, dataset, seed, window, λ) via the run's `results/<sweep>/<subdir>/params.json`** since `result.log` itself only has a hash ID. |
| `results/detect/<config>_msds_seed<n>/params.json` | Hyperparameters + hash ID for each detection run |
| `results/wsweep/<model>_msds_W<w>_seed<n>/params.json` | Same for window sweep |
| `results/lambda/mestgad_msds_lam<λ>_seed<n>/params.json` | Same for lambda sweep |
| `results/profile/temporal_complexity.json` | Structured JSON with module/window/time/memory. This is the clean file — not appended, overwritten per run. |
| `logs/<sweep>/*.err` and `*.out` | SLURM stdout/stderr. Useful for confirming OOM vs other failure modes. |

## result.log format

Appended format (from upstream `main.py`):
```
 <config_name>-<hash_id> --weight_decay:0.0005   --learning_change:100 
loss   pr:0.962  rc:0.985  auc:0.9999 ap:0.988 f1: 0.973 pred_right: 16368 ...
f1     pr:0.962  rc:0.992  auc:0.9999 ap:0.988 f1: 0.977 pred_right: 16367 ...
```

Each training run appends 3 lines: a header with `<config_name>-<hash_id>`, then
two eval lines — one for the checkpoint with best validation loss (`loss`), one
for the best F1 checkpoint (`f1`). For final analysis, use the `f1` line (better
selection criterion for this task).

The `<hash_id>` links back to `results/<sweep>/<subdir>/params.json`, where
you'll find the full config including `random_seed`, `lambda_ad`, `window`,
`config`, `dataset`.

## What the new chat should help with

1. **Parse `result.log` + `params.json` files** into a single analyzable DataFrame
   keyed by (config/model, dataset, seed, window, lambda_ad) with all metrics.
2. **Analyze the complexity profile** (once the rerun completes) — log-log plot
   of time and memory vs W, identify crossover point, confirm expected slopes
   (TAM ≈ quadratic, Mamba ≈ linear).
3. **Build the ablation table**: mean ± std F1 across 5 seeds for each of the
   4 configs on MSDS.
4. **Build the F1-vs-W plot** including the mstgad OOM annotation.
5. **Build the λ sensitivity plot** and identify optimal λ.
6. **Writeup**: frame the MSDS-only results honestly, acknowledge AIOps
   unavailability, note that paper's own ablation is MSDS-only so this is
   defensible.

## Key technical decisions baked into the code

- **Mamba blocks per modality, then linear cross-modal mixing** (not attention
  fusion). Replaces TAM's shared C-matrix averaging.
- **AD anchored to SAM, not TAM**. More semantically meaningful for
  microservice anomaly detection (inter-service disruption is spatial).
- **Separate Mamba hyperparameters** `d_state=16, d_conv=4, expand=2` (Mamba
  paper defaults). Not swept.
- **Original MSTGAD splits**: 70/30 train/test in released code (paper claims
  60/10/30). Kept the released code's split for apples-to-apples baseline
  comparison.

## Working style preference

User writes all implementation code. Claude provides skeletons, conceptual
framing, shape annotations, TODO comments, and references — but does NOT write
the implementation itself. This is a deliberate learning strategy for the
project. Exception: peripheral infrastructure (sbatch scripts, analysis
scripts, profiling harness) where the user is happy to have working code.

User prefers single self-contained files over modular structures for intuitive
understanding, and enjoys being challenged/corrected on technical reasoning.

## Immediate next step when new chat starts

The complexity profile rerun (`02_complexity_profile.sbatch` with fixed
`profile_temporal.py`) should be launched first since it's fast (~1 min) and
produces the headline figure. Once all four jobs' outputs are in hand, the
analysis work begins.