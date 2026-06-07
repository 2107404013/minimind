# AGENTS.md

## Project Goal

This repository is based on MiniMind. The current subproject is MiniMind-MathTutor: a compact math question-answering assistant built on MiniMind.

The project should support:

- official MiniMind SFT baseline
- math SFT
- LoRA fine-tuning
- Qwen2.5-Math-7B-Instruct teacher generation
- candidate-level KL distillation
- RAG for math notes and formulas
- unified evaluation
- simple WebUI demo

## Structure Rules

Keep the project compact and easy to study.

Do not create many new folders.
Do not create a new README for every stage.
Do not create repeated config files.
Do not create many sample files.

Use this structure:

- `configs/math_tutor.yaml` for all MathTutor configs
- `src/math_tutor/` for core logic
- `scripts/` for command-line entry scripts
- `docs/PROJECT.md` for project design
- `docs/RUNBOOK.md` for run commands
- `docs/EXPERIMENT_LOG.md` for experiment records
- `README_MathTutor.md` for final project README
- `sample_data/` for minimal sample data
- `sample_docs/` for minimal sample documents

## File Modification Rules

Before modifying files, first list the files you plan to change.

Prefer modifying existing files instead of creating new ones.

Each stage should usually modify only 3 to 6 files.

If a new file is needed, explain why the feature cannot fit into an existing file.

Do not modify MiniMind core model code unless it is necessary for compatibility.

Do not rewrite the original MiniMind training framework. Reuse existing MiniMind pretrain, SFT, LoRA, and inference logic whenever possible.

## Data and Model Rules

Never commit large files.

Do not commit:

- `checkpoints/`
- `models/`
- `data/raw/`
- `data/processed/`
- `outputs/`
- `wandb/`
- `*.pt`
- `*.pth`
- `*.safetensors`
- `*.bin`
- `*.log`

Large data, model weights, generated teacher outputs, RAG indexes, and evaluation outputs should stay outside Git tracking.

## Development Workflow

Each stage should be completed with a clear git commit.

Use commit messages like:

- `stage1: add official SFT mini baseline workflow`
- `stage2: add compact math data and Qwen teacher workflow`
- `stage3: add compact math SFT workflow`
- `stage4: add compact math LoRA workflow`
- `stage5: add compact candidate-level KL distillation`
- `stage6: add compact RAG and unified evaluation`
- `stage7: finalize compact MiniMind MathTutor demo and docs`

Before committing, always run:

```bash
git status
```

Only commit source code, configs, docs, and small sample files.

## Testing Rules

Do not start full training unless explicitly requested.

For each stage, first run a dry run or sample run.

Prefer commands like:

```bash
python scripts/build_math_data.py --config configs/math_tutor.yaml --sample
python scripts/train_math_sft.py --config configs/math_tutor.yaml --mode math_sft --dry_run
python scripts/eval_math.py --config configs/math_tutor.yaml --sample
```

## GPU Rules

The lab machine has two RTX 4090 GPUs.

Use GPU 0 mainly for MiniMind training and evaluation.
Use GPU 1 mainly for Qwen2.5-Math-7B-Instruct teacher generation.

Use `CUDA_VISIBLE_DEVICES` explicitly in commands when possible.

Examples:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_math_sft.py --config configs/math_tutor.yaml --mode math_sft
CUDA_VISIBLE_DEVICES=1 python scripts/generate_teacher.py --config configs/math_tutor.yaml --limit 10
```

If `torchrun` is supported by the reused MiniMind script, provide a two-GPU command.
If not, explain why single-GPU execution is used.

## Documentation Rules

Do not create scattered stage documents.

Update only:

- `docs/PROJECT.md`
- `docs/RUNBOOK.md`
- `docs/EXPERIMENT_LOG.md`
- `README_MathTutor.md`

`PROJECT.md` explains design.
`RUNBOOK.md` records commands.
`EXPERIMENT_LOG.md` records what was run and what result was observed.
`README_MathTutor.md` is for final GitHub display and resume packaging.

## MathTutor Technical Rules

The math data format should be compatible with MiniMind SFT conversations format:

```json
{
  "conversations": [
    {
      "role": "user",
      "content": "Please solve the math problem and give clear solution steps.\n..."
    },
    {
      "role": "assistant",
      "content": "..."
    }
  ],
  "source": "...",
  "level": "...",
  "type": "...",
  "final_answer": "..."
}
```

For distillation, do not use token-level KL between Qwen and MiniMind because their tokenizers may differ.

Use candidate-level KL distillation:

```text
Loss = SFT_Loss + lambda_kl * KL(P_teacher_candidates || P_student_candidates)
```

RAG should be used for math notes, formulas, definitions, theorems, and similar examples. It should not replace actual calculation ability.

## Final Goal

The final project should be easy to read, easy to run, and easy to explain in interviews.

The priority is:

1. clear structure
2. reproducible workflow
3. compact files
4. meaningful experiments
5. simple demo
