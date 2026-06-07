# MiniMind-MathTutor Project

## Goal

MiniMind-MathTutor is a learning-first math QA assistant built on the original
MiniMind repository. The project keeps MiniMind core model code unchanged and
adds a small, readable layer for data conversion, teacher-data preparation,
math SFT, LoRA, candidate-answer distillation, RAG, evaluation, and WebUI work.

## Current completed work

- MiniMind was cloned locally.
- `pretrain_t2t_mini.jsonl` pretraining was completed locally.
- Mini pretrain checkpoints exist in `out/` and `checkpoints/`.
- Stage 1 official SFT mini baseline was configured.
- A 16-sample SFT debug run completed on the local single RTX 5060.
- The previous scattered MathTutor docs and configs were archived under
  `archive/math_tutor_old/`.

## Important local artifacts

- Mini pretrain weight used by official SFT: `out/pretrain_768.pth`
- Mini pretrain resume checkpoint: `checkpoints/pretrain_768_resume.pth`
- Stage 1 debug SFT weight: `out/debug_stage1_sft_768.pth`
- Stage 1 debug resume checkpoint: `checkpoints/debug_stage1_sft_768_resume.pth`
- Official SFT data: `dataset/sft_t2t_mini.jsonl`
- Official pretrain mini data: `dataset/pretrain_t2t_mini.jsonl`

Large artifacts must remain untracked.

## Simplified structure

```text
configs/math_tutor.yaml
scripts/build_math_data.py
scripts/generate_teacher.py
scripts/train_math_sft.py
scripts/train_math_lora.py
scripts/train_math_kl.py
scripts/build_rag_index.py
scripts/eval_math.py
scripts/app_math_tutor.py
src/math_tutor/
sample_data/
sample_docs/
docs/PROJECT.md
docs/RUNBOOK.md
docs/EXPERIMENT_LOG.md
README_MathTutor.md
archive/math_tutor_old/
```

## What stays original

The following MiniMind areas are preserved as upstream project code:

- `model/`
- `trainer/`
- original scripts in `scripts/`
- `dataset/lm_dataset.py`
- `eval_llm.py`
- `README.md`, `README_en.md`, `LICENSE`

## Risk notes

- The local Windows machine should use single-process training on the RTX 5060.
  The official DDP path uses NCCL and is better suited to Linux or WSL2.
- The lab dual RTX 4090 machine can use `torchrun` after CUDA, PyTorch, and NCCL
  are verified.
- Candidate-answer KL from Qwen2.5-Math to MiniMind is not the same as the
  official MiniMind token-level distillation script, because the tokenizers and
  vocabularies differ.
- Do not commit checkpoints, full datasets, model weights, logs, or outputs.
