# MiniMind-MathTutor Experiment Log

## 2026-06-07: Project status and two-GPU plan

Commit: `3ff368d docs: add current status and dual 4090 training plan`

Summary:

- Confirmed MiniMind project structure.
- Found local mini pretrain checkpoints.
- Confirmed official scripts for pretrain, SFT, LoRA, distillation, inference,
  serving, and WebUI.
- Confirmed official training scripts support `torchrun` and DDP in principle.
- Noted Windows native NCCL risk for local multi-GPU runs.
- Recommended keeping mini checkpoint for pipeline validation before full
  pretrain work.

## 2026-06-07: Stage 1 official SFT mini configuration

Commit: `0b47957 stage1: configure official MiniMind SFT mini baseline`

Summary:

- Read MiniMind official `trainer/train_full_sft.py`.
- Added Stage 1 SFT configs for local single RTX 5060 and future dual RTX 4090.
- Confirmed official script uses argparse rather than YAML.
- Ran a 16-sample debug SFT pass on the local RTX 5060.

Debug output excerpt:

```text
Model Params: 63.91M
Trainable Params: 63.912M
Epoch:[1/1](1/16), loss: 3.1348
Epoch:[1/1](16/16), loss: 3.6446
```

Debug artifacts kept locally but not tracked:

- `out/debug_stage1_sft_768.pth`
- `checkpoints/debug_stage1_sft_768.pth`
- `checkpoints/debug_stage1_sft_768_resume.pth`

## 2026-06-07: Simplified project structure

Commit: pending in this stage.

Summary:

- Archived scattered previous MathTutor docs and configs.
- Added one unified config at `configs/math_tutor.yaml`.
- Added small helper package under `src/math_tutor/`.
- Added simple scripts for data conversion, teacher prompts, official trainer
  command building, candidate KL prep, RAG index building, evaluation, and WebUI
  scaffold.
- Added small sample data and sample docs for local tests.
