# MiniMind-Math-Lab Context Summary

This file is the handoff context for ending the current Codex thread and
continuing work in a new project/thread.

Current date: 2026-06-17

## Project Positioning Decision

We decided to reposition the old MiniMind-MathTutor work as:

MiniMind-Math-Lab: small-parameter language model math ability boundary
experiment.

The project is no longer positioned as a high-accuracy GSM8K math assistant.
The core goal is to study what a MiniMind 64M model can and cannot do after
pretrain, SFT, math SFT, possible LoRA, and black-box teacher data.

Important conclusion:

- MiniMind 64M is valuable for learning and reproducing the LLM workflow.
- It is suitable for baseline pretraining, SFT/LoRA process reproduction,
  data construction, debug prediction, and difficulty-wise evaluation.
- It is not suitable as a reliable complex multi-step math reasoning model.
- For an actually useful math tutor, start a new Qwen-MathTutor project and
  migrate the data, teacher generation, and evaluation framework to a stronger
  base model.

## Workspace Locations

Local Windows workspace:

```text
D:\pythonfile\minimind_project\minimind
```

Remote Linux/4090 workspace used earlier:

```text
/home/ljk/minimind
host: jsl-4090
conda env: mm4090
GPU: 2 x RTX 4090
```

## Key Constraints To Preserve

- Do not continue the old seven-stage plan.
- Do not modify MiniMind core model structure.
- Do not launch blind full training.
- Do not submit checkpoints, outputs, large JSONL data, or `.pth` weights.
- Keep generated artifacts under `outputs/`, `out/`, or `checkpoints/`.
- Keep the repo simple: one main config, one main doc set, a few scripts.
- Treat MiniMind 64M as an ability-boundary experiment, not as the final
  product model.

## Major Decisions Made

1. Stop chasing high GSM8K accuracy with MiniMind 64M.
2. Keep the 500 Qwen-GSM8K SFT experiment as evidence for the capacity
   bottleneck.
3. Use compact final-answer SFT targets for small-model overfit sanity checks.
4. Use difficulty-wise evaluation as the main lens:
   - `arithmetic`
   - `template_word_problem`
   - `gsm8k_easy`
   - `gsm8k_medium`
   - `hard_reasoning`
5. Add a fixed boundary evaluation set so every checkpoint is compared against
   the same small benchmark.
6. Move the future high-quality assistant direction to a new Qwen-MathTutor
   project.

## Important Commit Already Made

The project repositioning was committed:

```text
73a3fb7 reposition: finalize MiniMind Math Lab as small-model ability boundary project
```

That commit updated the README/docs/config/eval/data pieces to reflect
MiniMind-Math-Lab.

## Files Changed In The Current Uncommitted Step

After the repositioning commit, we continued by adding the boundary-evaluation
workflow. These changes are currently uncommitted unless committed later:

```text
README_MathTutor.md
configs/math_tutor.yaml
docs/EXPERIMENT_LOG.md
docs/PROJECT.md
docs/RUNBOOK.md
src/math_tutor/data.py
src/math_tutor/eval.py
CONTEXT.md
```

There are also unrelated dirty/untracked files in the worktree from previous
work. Do not stage them unless explicitly intended:

```text
CODE_OF_CONDUCT.md
README_en.md
archive/math_tutor_old/docs/current_status_and_2gpu_plan.md
archive/math_tutor_old/docs/stage1_official_sft_mini.md
dataset/dataset.md
sample_docs/math_notes_sample.md
stage3_compact.patch
```

## What Was Added For Boundary Evaluation

### src/math_tutor/data.py

Added a deterministic boundary benchmark builder:

- `build_boundary_eval_records(items_per_bucket=10)`
- `write_boundary_eval_file(...)`
- CLI flags:
  - `--build-boundary-eval`
  - `--items-per-bucket`

The generated benchmark has 50 examples:

- 10 arithmetic questions
- 10 template word problems
- 10 GSM8K easy style problems
- 10 GSM8K medium style problems
- 10 hard reasoning style problems

The output is intended to be written to:

```text
outputs/eval_boundary_core.jsonl
```

Do not commit that generated file.

### src/math_tutor/eval.py

Added boundary summary reporting:

- `accuracy_by_difficulty`
- `boundary_summary`
- `stable_boundary`
- `first_unstable_bucket`
- per-bucket status:
  - `pass`
  - `partial`
  - `fail`
  - `missing`

Default thresholds:

```text
pass: accuracy >= 0.8
partial: 0.5 <= accuracy < 0.8
fail: accuracy < 0.5
```

Also added CLI support:

```text
--checkpoint
```

This lets the user evaluate any checkpoint without editing
`configs/math_tutor.yaml`.

### configs/math_tutor.yaml

Added:

```yaml
evaluation:
  boundary_eval_file: outputs/eval_boundary_core.jsonl
  boundary_report_path: outputs/math_boundary_report.json
  boundary:
    items_per_bucket: 10
    pass_threshold: 0.8
    partial_threshold: 0.5
```

### docs and README

Updated:

- `README_MathTutor.md`
- `docs/PROJECT.md`
- `docs/RUNBOOK.md`
- `docs/EXPERIMENT_LOG.md`

The docs now explain that the boundary set is a quick diagnostic tool, not a
replacement for GSM8K or a training set.

## Commands Already Verified Locally

Python compile check passed:

```bash
python -m py_compile src\math_tutor\data.py src\math_tutor\eval.py scripts\build_math_data.py scripts\eval_math.py
```

Boundary set generation passed:

```bash
$env:PYTHONPATH='src'
python scripts\build_math_data.py --config configs\math_tutor.yaml --build-boundary-eval --output outputs\eval_boundary_core.jsonl
```

It produced:

```text
total: 50
arithmetic: 10
template_word_problem: 10
gsm8k_easy: 10
gsm8k_medium: 10
hard_reasoning: 10
```

Fallback evaluation also passed:

```bash
$env:PYTHONPATH='src'
python scripts\eval_math.py --config configs\math_tutor.yaml --mode math_sft --input outputs\eval_boundary_core.jsonl --checkpoint out\not_exists_for_boundary_check.pth --sample --output outputs\math_boundary_report.json
```

Important: fallback score is 1.0 only because it evaluates stored assistant
answers from the JSONL. It verifies the toolchain, not the actual model.

## How To Find The Real MiniMind Boundary

Run this on the Linux/4090 environment where the real `.pth` checkpoint exists.
Do not use `--sample`.

```bash
cd ~/minimind
conda activate mm4090

PYTHONPATH=src python scripts/build_math_data.py \
  --config configs/math_tutor.yaml \
  --build-boundary-eval \
  --output outputs/eval_boundary_core.jsonl

PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python scripts/eval_math.py \
  --config configs/math_tutor.yaml \
  --mode math_sft \
  --input outputs/eval_boundary_core.jsonl \
  --checkpoint out/full_sft_math_compact_500_strong_768.pth \
  --output outputs/math_boundary_report.json \
  --debug
```

Read:

```text
outputs/math_boundary_report.json
```

The key field is:

```text
report.boundary_summary
```

Interpretation:

- `stable_boundary = arithmetic`: the model only reliably handles direct
  arithmetic.
- `stable_boundary = template_word_problem`: it handles fixed simple word
  problems but not GSM8K-like composition.
- `stable_boundary = gsm8k_easy`: it can handle one- or two-step problems but
  not longer GSM8K.
- `stable_boundary = gsm8k_medium`: surprisingly strong for 64M; verify with
  a larger held-out set before claiming.
- `stable_boundary = hard_reasoning`: likely overfitting or benchmark leakage;
  inspect debug predictions before believing it.

## Current Experimental Observations

Known from prior runs:

- 500 compact strong overfit can reach near-perfect performance on the same
  training subset.
- Generalization to unseen complex GSM8K remains poor.
- A 5000-sample compact model showed very low overfit/generalization in one
  test, which suggests dataset/checkpoint/config mismatch or small-model
  capacity limits.
- Long Qwen-style chain-of-thought targets are too difficult for MiniMind 64M.
- Short final-answer targets improve format and memorization, but do not create
  reliable reasoning ability.

## Black-Box Distillation Note

Qwen and MiniMind tokenizers are different, so token-level white-box KL
distillation from Qwen to MiniMind is not valid in this project.

For MiniMind, "black-box distillation" mostly means:

- generate teacher answers;
- convert them to MiniMind conversation SFT data;
- optionally create candidate/ranking/preference data;
- train MiniMind only on text-level outputs, not teacher logits.

This is closely related to SFT. The main difference is whether the teacher is
used only to produce one target answer, or also to produce/score multiple
candidates for preference-style training.

## Recommended Next Project

Start a separate project:

```text
Qwen-MathTutor
```

Move the useful pieces there:

- teacher data generation prompts;
- compact final-answer formatting;
- debug prediction inspection;
- difficulty-wise evaluation;
- boundary-summary reporting;
- runbook discipline around outputs/checkpoints.

Use a stronger base model for actual math assistant behavior. MiniMind-Math-Lab
should remain a compact research/resume artifact demonstrating the full
training/evaluation workflow and the capacity boundary of a 64M model.

## What To Do Before Leaving This Repo

If you want to keep the boundary-evaluation changes:

```bash
git add README_MathTutor.md configs/math_tutor.yaml src/math_tutor/data.py src/math_tutor/eval.py docs/PROJECT.md docs/RUNBOOK.md docs/EXPERIMENT_LOG.md CONTEXT.md
git commit -m "eval: add fixed boundary benchmark for MiniMind Math Lab"
```

Do not add:

```text
outputs/
out/
checkpoints/
*.pth
large JSONL files
unrelated dirty files
```

