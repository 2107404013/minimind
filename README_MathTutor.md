# MiniMind-MathTutor

This is a simplified learning structure for building a math QA assistant on top
of MiniMind. It preserves the original MiniMind model and trainer code, and adds
a small project layer for math data, teacher-data preparation, training command
wrappers, RAG, evaluation, and WebUI work.

## Start here

- Project overview: `docs/PROJECT.md`
- Commands and stage order: `docs/RUNBOOK.md`
- Experiment notes: `docs/EXPERIMENT_LOG.md`
- Unified config: `configs/math_tutor.yaml`

## Current baseline

The current usable base is the locally trained mini pretrain checkpoint:

```text
out/pretrain_768.pth
```

Stage 1 official SFT mini was configured and debug-tested on a single RTX 5060.
Full official SFT should continue to use MiniMind's official
`trainer/train_full_sft.py`.

## Safe defaults

The new training helper scripts print commands by default. They do not start
training unless `--run` is explicitly provided.

Large local artifacts are ignored by Git:

- `checkpoints/`
- `models/`
- `data/raw/`
- `data/processed/`
- `outputs/`
- `wandb/`
- model weight and log files

## Recommended order

1. Run or resume official SFT mini baseline.
2. Convert math raw data to MiniMind SFT conversations JSONL.
3. Prepare teacher prompts and clean teacher outputs.
4. Train math SFT.
5. Train math LoRA.
6. Add candidate-answer KL distillation.
7. Build RAG index.
8. Run unified evaluation.
9. Start WebUI work.
