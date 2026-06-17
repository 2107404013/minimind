# MiniMind-Math-Lab 实验记录

## 2026-06-17：固定边界集复评结果

Checkpoint：

```text
out/full_sft_math_compact_500_strong_768.pth
```

评测设置：

- 原始英文边界集，`temperature=0.85`，`do_sample=true`；
- 原始英文边界集，低温采样，`temperature=0.2`，`top_p=0.9`；
- 中文 prompt 边界集，greedy decoding。

复评结果：

- `stable_boundary = none`；
- `first_unstable_bucket = arithmetic`；
- arithmetic 档没有达到 `accuracy >= 0.8` 的 pass 阈值；
- debug predictions 显示模型可以生成数学相关文本，偶尔能给出正确答案，但在未见边界题上计算和步骤都不稳定。

结论：

`full_sft_math_compact_500_strong_768.pth` 可以在 500 条 compact 训练子集上强记忆，但不能泛化到固定未见边界题。采样温度、英文/中文 prompt 和答案格式复核后，结论仍然不变：当前 MiniMind 64M 数学 SFT checkpoint 不具备稳定的未见题数学推理能力。

项目解释：

- 这不是训练链路失败；此前 compact 100/500 过拟合实验已经证明 SFT、checkpoint 加载、loss mask 和 evaluator 可以工作；
- 这说明 64M 小模型在当前数据和训练设置下主要学习了格式与局部记忆，而不是可靠的数学泛化；
- MiniMind-Math-Lab 后续应作为小模型能力边界实验收尾，不再继续盲目扩大数据量或训练轮数；
- 真正可用的数学助手方向应迁移到新的 Qwen-MathTutor 项目。

## 2026-06-17：新增固定边界评测集

目标：把后续实验从“试一次训练看感觉”改成“每个 checkpoint 跑同一份小边界集”。

新增内容：

- `scripts/build_math_data.py --build-boundary-eval` 可生成 `outputs/eval_boundary_core.jsonl`；
- 每档 10 道题，覆盖 `arithmetic`、`template_word_problem`、`gsm8k_easy`、`gsm8k_medium`、`hard_reasoning`；
- `src/math_tutor/eval.py` 报告新增 `boundary_summary`；
- 默认 `accuracy >= 0.8` 记为 pass，`0.5 <= accuracy < 0.8` 记为 partial，低于 `0.5` 记为 fail；
- `stable_boundary` 表示当前模型稳定通过的最高难度档，`first_unstable_bucket` 表示首次不稳定的难度档。

使用原则：

- 不把该边界集当成正式 GSM8K 分数；
- 不用它做训练集；
- 只用于快速比较 MiniMind 64M 不同训练版本的能力边界。

## 2026-06-17：项目重新定位为 MiniMind-Math-Lab

项目名称：

```text
MiniMind-Math-Lab：小参数语言模型数学能力边界实验
```

定位调整：

- MiniMind 64M 的主要价值是学习和复现 LLM 全流程；
- 当前项目不再追求 GSM8K 高正确率；
- MiniMind 64M 适合做从零预训练 baseline、SFT / LoRA 流程复现、小模型数学能力边界分析和数据难度分层评测；
- MiniMind 64M 不适合直接作为复杂多步数学推理模型；
- 后续会新建 Qwen-MathTutor 项目，将数据构建、教师生成和评测框架迁移到更强底座模型。

## 2026-06-17：MiniMind 64M + 500 Qwen-GSM8K SFT

实验名称：

```text
MiniMind 64M + 500 Qwen-GSM8K SFT
```

实验设置：

- base model：MiniMind 64M；
- teacher：Qwen2.5-Math-7B-Instruct；
- SFT 数据：500 条 Qwen 生成的 GSM8K 解答；
- 数据格式：MiniMind conversations；
- 训练目的：检查 64M 小模型在短数学 SFT 数据上的记忆能力、格式学习能力和未见题泛化能力。

当前观察：

- 500 条 compact/short target 训练可以在训练子集上达到很高记忆效果；
- 对未见过的复杂 GSM8K 样本，正确率仍然较低；
- 说明训练链路、checkpoint 加载和 answer extraction 基本可用，但 64M 模型在复杂数学推理上的泛化有限。

可能原因：

- 模型参数量过小；
- 500 条数据不足以覆盖 GSM8K 的复杂题型分布；
- 长 CoT 对 64M 模型过难；
- MiniMind 64M 的数学计算和多步推理能力不足；
- final answer 提取、答案格式和单位归一化仍可能影响评测。

后续结论：

- 将 MiniMind 作为能力边界实验，而不是高准确率数学问答助手；
- 不继续强行追求高 GSM8K 分数；
- 保留分难度评测，重点观察 arithmetic、template word problem、GSM8K easy、GSM8K medium 和 hard reasoning 的边界；
- 将正式效果模型转移到 Qwen-MathTutor 项目。

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
- After normalizing the overfit rows into official SFT schema and appending a
  final-answer marker, the 100-row overfit still stayed near zero:
  `exact_match=0.0`, `relaxed_match=0.0`, `answer_contains=0.01`, and
  `invalid_output_rate=0.91`.
- Debug predictions showed long reasoning cut off before the final-answer
  marker. Evaluation was still using `max_new_tokens=256`, which can truncate
  teacher-style math solutions before `答案是：...`.

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
- Keep the next Stage 3 run as a stronger 100-row overfit sanity test using
  dedicated debug learning-rate and gradient-accumulation settings before
  changing data size.
- Run debug evaluation with enough output budget and MiniMind-style sampling:
  `evaluation.max_new_tokens=1024`, `temperature=0.85`, `top_p=0.95`, and
  `do_sample=true`. Deterministic greedy generation with `temperature=0.0`
  collapsed into near-empty output or `<tool_call>` on both the overfit
  checkpoint and the official SFT baseline, so it is not a reliable Stage 3
  math-eval setting for this model.
- Do not implement token-level KL between Qwen and MiniMind.
- Use black-box candidate-level distillation data: Qwen answer, MiniMind
  current answer, gold final-answer response, and perturbed wrong answer.
- Convert candidates into `chosen` / `rejected` preference pairs for later DPO
  or ranking-loss work.

# 2026-06-16: Stage 3 compact target overfit breakthrough

Goal:

- Test whether MiniMind can memorize a tiny math SFT set when the assistant
  target is short and answer-focused instead of long Qwen-style reasoning.

Result:

- Long teacher target, sampling/open thinking experiments remained low:
  `answer_contains` around `0.06` to `0.07`.
- Compact 100-row target with sampling improved to about `0.27`.
- The same compact checkpoint evaluated greedily improved to about `0.49`.
- Strong compact overfit with 10 epochs and greedy evaluation reached:
  `exact_match=1.0`, `relaxed_match=1.0`, `answer_contains=1.0`,
  `invalid_output_rate=0.01`, and final loss around `0.0066`.

Interpretation:

- The Stage 3 SFT pipeline, checkpoint loading, loss mask, and evaluator are
  functioning.
- The main failure was the target contract: Qwen answers were too verbose and
  reasoning-heavy for the 63M MiniMind model.
- The next Stage 3 step is to convert the 5000-row teacher file to compact
  final-answer targets and run one reproducible compact math SFT experiment.

Implementation:

- `scripts/build_math_data.py --compact-final-target` now rewrites existing
  SFT rows to a short answer target:
  `解：最终结果为 <final_answer>。` followed by `答案是：<final_answer>`.
- Use `--official-sft-compatible` for files passed to
  `trainer/train_full_sft.py`.
