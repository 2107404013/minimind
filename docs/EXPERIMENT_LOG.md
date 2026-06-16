# MiniMind-MathTutor 实验记录

## 2026-06-07：当前状态与双卡计划

提交：`3ff368d docs: add current status and dual 4090 training plan`

主要内容：

- 确认了 MiniMind 项目结构。
- 找到了本地 mini pretrain checkpoints。
- 确认了官方 pretrain、SFT、LoRA、distillation、inference、serving 和 WebUI 相关脚本。
- 确认官方训练脚本原则上支持 `torchrun` 和 DDP。
- 记录了 Windows 原生环境下 NCCL 的多卡风险。
- 建议先使用 mini checkpoint 跑通全流程，再考虑 full pretrain。

## 2026-06-07：阶段 1 官方 SFT mini 配置

提交：`0b47957 stage1: configure official MiniMind SFT mini baseline`

主要内容：

- 阅读了 MiniMind 官方 `trainer/train_full_sft.py`。
- 为本地单卡 RTX 5060 和后续双 RTX 4090 分别记录了 SFT 配置。
- 确认官方脚本使用 argparse，不直接读取 YAML。
- 在本地 RTX 5060 上跑通过 16 条样本的 SFT debug run。

debug 输出摘录：

```text
Model Params: 63.91M
Trainable Params: 63.912M
Epoch:[1/1](1/16), loss: 3.1348
Epoch:[1/1](16/16), loss: 3.6446
```

debug 产物保留在本地，但不纳入 Git：

- `out/debug_stage1_sft_768.pth`
- `checkpoints/debug_stage1_sft_768.pth`
- `checkpoints/debug_stage1_sft_768_resume.pth`

## 2026-06-07：简化项目结构

提交：`46078de refactor: simplify MiniMind MathTutor project structure`

主要内容：

- 将原来分散的 MathTutor 文档和配置移动到 `archive/math_tutor_old/`。
- 新增统一配置 `configs/math_tutor.yaml`。
- 新增轻量 helper package：`src/math_tutor/`。
- 新增数据转换、教师请求、官方训练命令封装、候选 KL 数据准备、RAG、评测和 WebUI 脚手架脚本。
- 新增少量 sample 数据和 sample 文档，便于后续小规模测试。
# 2026-06-07: Stage 1 official SFT mini baseline workflow

Commit: pending, `stage1: add official SFT mini baseline workflow`

Goal:

- Build a general MiniMind chat baseline from the completed mini pretrain
  checkpoint.
- Use MiniMind official `trainer/train_full_sft.py`.
- Keep this separate from math SFT; this stage is for normal conversation
  ability.

Data:

- Train file: `dataset/sft_t2t_mini.jsonl`
- Valid file: not used by the official MiniMind SFT script.

Checkpoint:

- Base checkpoint: `out/pretrain_768.pth`
- Official trainer argument: `--from_weight pretrain`
- Expected output: `out/full_sft_768.pth`
- Expected resume checkpoint: `checkpoints/full_sft_768_resume.pth`

Dry run result:

- Completed command dry run with:
  `python scripts/train_math_sft.py --config configs/math_tutor.yaml --mode official_sft --dry_run`
- Output printed the delegated official command:
  `train_full_sft.py --data_path ../dataset/sft_t2t_mini.jsonl --save_dir ../out --save_weight full_sft --from_weight pretrain ...`
- No full training was started and no checkpoint was written.

# 2026-06-07: Stage 1 official SFT mini baseline completed

Goal:

- Finish the general MiniMind SFT baseline from the completed mini pretrain
  checkpoint.
- This is the normal dialogue baseline, not math SFT.

Remote training:

- Machine: lab Linux server `jsl-4090`, dual RTX 4090.
- Conda env: `mm4090`.
- Data: `dataset/sft_t2t_mini.jsonl`.
- Base checkpoint: `out/pretrain_768.pth`.
- Command path: MiniMind official `trainer/train_full_sft.py` via `torchrun`.
- Epochs: 2.
- Batch size per process: 8.
- Gradient accumulation steps: 4.
- Max sequence length: 768.
- Learning rate: `1e-5`.

Outputs copied back locally:

- SFT checkpoint: `out/full_sft_768.pth`.
- Resume checkpoint: `checkpoints/full_sft_768_resume.pth`.
- Training log: `outputs/logs/stage1_official_sft_20260607_173708.log`.

Log check:

- Final observed line reached `Epoch:[2/2](56608/56608)`.
- No matches found for `error`, `traceback`, `nan`, `out of memory`, `oom`, or `RuntimeError`.

Dialogue validation:

- Local device: NVIDIA GeForce RTX 5060 Laptop GPU.
- Validation loaded `out/full_sft_768.pth` with the official MiniMind model
  loading logic.
- Prompt: `你好，请简单介绍一下你自己。`
- Result: model produced a coherent Chinese self-introduction as MiniMind.
- Note: the sample response included an extra `</think>` token before repeating
  the answer, so later inference prompts/templates may need cleanup.

# 2026-06-07: Stage 2 math data and Qwen teacher workflow

Commit: pending, `stage2: add compact math data and Qwen teacher workflow`

Goal:

- Build compact math SFT data in MiniMind conversations format.
- Support raw `.json` and `.jsonl` math records.
- Detect `question/problem/input/query` fields for problems.
- Detect `answer/solution/output/response` fields for answers.
- Filter empty records, duplicates, and overlong records.
- Provide deterministic train/valid/test splitting.
- Add a Qwen2.5-Math-7B-Instruct teacher generation path with resume,
  `limit`, optional 4bit loading, and failed sample logging.

Data format:

- User message: `请解答下面的数学题，并给出清晰的解题步骤：`
- Assistant message: step-by-step solution, ending with `答案是：...` when a
  final answer field is available.
- Metadata: `source`, `level`, `type`, and `final_answer`.

Implementation result:

- `src/math_tutor/data.py` now contains reusable JSON/JSONL reading,
  conversion, deduplication, length filtering, and split helpers.
- `scripts/build_math_data.py` exposes the CLI entry point.
- `src/math_tutor/teacher.py` loads Qwen from a local path or cached Hugging
  Face name, supports optional 4bit loading, resumes existing output files, and
  appends failed per-sample generations to `outputs/failed_teacher.jsonl`.
- `scripts/generate_teacher.py` exposes the CLI entry point.
- `configs/math_tutor.yaml` records the compact data and teacher settings.
- `sample_data/math_raw_sample.jsonl` and `sample_data/math_sft_sample.jsonl`
  remain the only sample data files for this stage.

Validation:

- Ran `python scripts/build_math_data.py --config configs/math_tutor.yaml --sample`.
- Result: `read=3`, `converted=3`, no empty, duplicate, too-short, or too-long
  rows.
- Ran `python -m py_compile src/math_tutor/data.py src/math_tutor/teacher.py scripts/build_math_data.py scripts/generate_teacher.py`.
- Result: syntax check passed.
- Ran the teacher command with `HF_HUB_OFFLINE=1` and
  `TRANSFORMERS_OFFLINE=1` to ensure no model download.
- Result: the current shell `python` environment does not have `torch`, so Qwen
  loading stopped with `transformers and torch are required for Qwen teacher
  generation.` No model weights were downloaded and no generation was started.

# 2026-06-07: Stage 3 math SFT workflow

Commit: pending, `stage3: add compact math SFT workflow`

Goal:

- Train MiniMind-MathTutor with math SFT data.
- Start from either `official_sft_mini` (`out/full_sft_768.pth`) or a later
  math continue-pretrain checkpoint.
- Reuse MiniMind official `trainer/train_full_sft.py`.
- Do not introduce a new training framework.
- Do not start full training in this stage validation.

Configured default:

- Base checkpoint: `out/full_sft_768.pth`.
- Official trainer `--from_weight`: `full_sft`.
- Train file: `data/processed/math_sft.jsonl`.
- Valid file: `data/processed/math_sft_valid.jsonl` for record keeping.
- Output checkpoint: `out/full_sft_math_768.pth`.
- Epochs: 1.
- Batch size per process: 8.
- Gradient accumulation steps: 4.
- Learning rate: `1e-5`.
- Max sequence length: 768.

Notes:

- `valid_file`, `warmup_ratio`, and `eval_steps` are present in
  `configs/math_tutor.yaml` for reproducibility, but the official MiniMind SFT
  script does not consume them.
- The training wrapper dry-runs by default unless `--run` is passed.
- The minimal evaluator reads MiniMind conversations JSONL, generates with a
  MiniMind checkpoint when available, extracts the text after `答案是`, reports
  answer contains, average output length, and average latency.

Validation:

- Dry run command:
  `CUDA_VISIBLE_DEVICES=0 python scripts/train_math_sft.py --config configs/math_tutor.yaml --mode math_sft --dry_run`
  printed the delegated official `train_full_sft.py` command and did not start
  training. Local note: `data/processed/math_sft.jsonl` is not present on this
  machine yet, so the wrapper reported it as a dry-run warning.
- Sample eval command:
  `python scripts/eval_math.py --config configs/math_tutor.yaml --mode math_sft --sample`
  completed on 3 sample rows. Because `out/full_sft_math_768.pth` does not
  exist before math SFT training, sample mode evaluated stored assistant
  answers. Result: `answer_contains=1.0`, average output length `49.0`.
- Full math SFT has not been started.

# 2026-06-08: Official SFT schema compatibility patch

Goal:

- Remove the manual `*_sft_compat.jsonl` conversion step used during 20/100 row
  math SFT smoke runs.
- Keep MiniMind core trainer and dataset code unchanged.

Implementation:

- Added a MathTutor helper that converts conversation rows to the narrow schema
  expected by MiniMind official `SFTDataset`.
- `scripts/build_math_data.py` supports `--official-sft-compatible`.
- `scripts/generate_teacher.py` supports `--official-sft-compatible`, so Qwen
  teacher output can be used directly as `trainer/train_full_sft.py --data_path`.

Recommended next command shape:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/generate_teacher.py \
  --config configs/math_tutor.yaml \
  --input outputs/math_sft_100.jsonl \
  --output outputs/teacher_100_train.jsonl \
  --limit 100 \
  --official-sft-compatible
```

# 2026-06-15: Stage 3 math SFT workflow refresh

Commit: pending, `stage3: add compact math SFT workflow`

Goal:

- Keep the math SFT path compact and based on MiniMind official
  `trainer/train_full_sft.py`.
- Support `official_sft` and `math_sft` command modes from
  `scripts/train_math_sft.py`.
- Validate only with dry run and sample evaluation, not full training.

Experiment settings:

- Base checkpoint: `training.math_sft.base_checkpoint`.
- Default base: `out/full_sft_768.pth`.
- Official trainer weight name: derived from `base_checkpoint` when
  `training.math_sft.from_weight` is null.
- Train file: `data/processed/math_sft.jsonl`.
- Valid file: `data/processed/math_sft_valid.jsonl` for record keeping.
- Output directory: `out`.
- Output checkpoint: `out/full_sft_math_768.pth`.
- Learning rate: `1e-5`.
- Batch size: `8`.
- Gradient accumulation steps: `4`.
- Max sequence length: `768`.
- Epochs: `1`.
- Warmup ratio: `0.03` recorded only.
- Save steps: `1000`.
- Eval steps: `1000` recorded only.
- Resume: `false`.

Validation commands:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_math_sft.py \
  --config configs/math_tutor.yaml \
  --mode math_sft \
  --dry_run
```

```bash
python scripts/eval_math.py \
  --config configs/math_tutor.yaml \
  --mode math_sft \
  --sample
```

Validation results:

- Dry run: passed. The wrapper printed the delegated
  `trainer/train_full_sft.py` command and did not start training. Local note:
  `data/processed/math_sft.jsonl` is not present on this machine yet.
- Sample evaluation: passed on 3 sample rows with sample-answer fallback because
  `out/full_sft_math_768.pth` does not exist yet. Result:
  `answer_contains=1.0`, `avg_output_length=49.0`.
- Full training: not started in this validation stage.

# 2026-06-15: Stage 3 low SFT accuracy diagnosis

Problem:

- A 5000-row Qwen-generated GSM8K math SFT run completed and produced
  `out/full_sft_math_official_5000_768.pth`.
- Full evaluation on `outputs/math_sft_gsm8k_test.jsonl` reported
  `answer_contains` around `0.04`.
- Qualitative samples showed long English reasoning, unstable final-answer
  markers, and weak arithmetic.
- Qwen2.5-Math and MiniMind do not share tokenizers, so token-level white-box
  KL is not appropriate for Qwen-to-MiniMind distillation.

Possible causes to diagnose:

- Wrong base checkpoint loaded during SFT.
- Wrong checkpoint loaded during evaluation.
- Training loss did not decrease or the captured log is insufficient.
- No validation loss is recorded because the official MiniMind SFT trainer does
  not consume `valid_file`.
- Training data is not in MiniMind `conversations` format.
- Loss mask has no supervised assistant tokens.
- `max_seq_len=768` truncates long Qwen answers before the final answer.
- MiniMind tokenizer expands digits, math symbols, English text, or Chinese
  answer markers more than expected.
- `final_answer` is missing or not normalized.
- Eval answer extraction misses `答案是`, English `answer is`, `####`, boxed,
  fraction, decimal, negative, or percent formats.

Implemented diagnostics:

- `python scripts/train_math_sft.py --config configs/math_tutor.yaml --mode math_sft --diagnose`
  prints checkpoint, data, tokenizer, truncation, and loss-mask diagnostics.
- `python scripts/eval_math.py --config configs/math_tutor.yaml --mode math_sft --debug --sample`
  writes the first configured debug predictions to
  `outputs/debug_predictions.jsonl`.
- `CUDA_VISIBLE_DEVICES=0 python scripts/train_math_sft.py --config configs/math_tutor.yaml --mode math_sft --overfit_debug`
  prepares a 100-row overfit SFT run using MiniMind official
  `trainer/train_full_sft.py`.

Diagnosis result placeholders:

- Base checkpoint check: `out/full_sft_official_768.pth` exists on remote.
- Eval checkpoint check: evaluation loaded
  `out/full_sft_math_official_5000_768.pth` for the 5000-row SFT checkpoint.
- Train loss trend: 5000-row run completed; 100-row overfit debug printed
  epoch losses `0.2369 -> 0.2887 -> 0.4109`, so the small overfit run did not
  show a healthy decreasing trend.
- Valid loss: not recorded by official SFT trainer.
- Conversations schema: valid MiniMind conversations were found.
- Loss mask assistant-token count: positive supervised assistant labels were
  found in the official `SFTDataset`.
- Truncation rate: about `0.0214` at `max_seq_len=768`; truncation is present
  but not the dominant failure mode.
- Tokenizer probe: available through `--diagnose`.
- Debug prediction accuracy on first 20 rows: pending.
- 100-row overfit result on the raw teacher format failed to memorize:
  `exact_match=0.0`, `relaxed_match=0.0`, `answer_contains=0.01`, and
  `invalid_output_rate=0.88`.

Current diagnosis:

- The Stage 3 training/eval plumbing is loading distinct checkpoints, and the
  loss mask has supervised assistant tokens.
- The main immediate issue is the teacher data contract: rows often lack a
  top-level `final_answer`, and assistant answers do not consistently end with
  the final marker expected by evaluation.
- Next Stage 3 fix is to normalize SFT rows so every assistant answer ends with
  `答案是：<final_answer>`, then rerun the 100-row overfit test before any full
  5000-row retraining.

Next improvement direction:

- Do not start blind full retraining.
- Do not implement token-level KL between Qwen and MiniMind.
- Use black-box candidate-level distillation data: Qwen answer, MiniMind
  current answer, gold final-answer response, and perturbed wrong answer.
- Convert candidates into `chosen` / `rejected` preference pairs for later DPO
  or ranking-loss work.
