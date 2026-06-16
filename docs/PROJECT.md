# MiniMind-MathTutor 项目说明

## 项目目标

MiniMind-MathTutor 是一个基于 MiniMind 的数学题问答助手。当前目标不是重写 MiniMind，而是在原始 MiniMind 项目之上增加一层清晰、便于学习和继续开发的数学助手工程结构。

本项目保持 MiniMind 核心模型结构不变，只新增以下能力：

- 数学原始数据转换；
- 教师模型数据准备；
- 数学 SFT 与 LoRA 的命令封装；
- 候选答案级 KL 蒸馏的数据准备；
- RAG 索引构建；
- 统一评测；
- WebUI 脚手架。

## 当前已完成内容

- MiniMind 已经 clone 到本地。
- 已经使用 `pretrain_t2t_mini.jsonl` 完成 mini 预训练。
- mini pretrain checkpoint 已经生成在 `out/` 和 `checkpoints/`。
- 阶段 1 官方 SFT mini baseline 已经完成配置。
- 已经在本地单卡 RTX 5060 上跑通过 16 条样本的 SFT debug run。
- 原来分散的 MathTutor 配置和阶段文档已移动到 `archive/math_tutor_old/`。

## 重要本地文件

- 官方 SFT 使用的 mini 预训练权重：`out/pretrain_768.pth`
- mini 预训练续训 checkpoint：`checkpoints/pretrain_768_resume.pth`
- 阶段 1 debug SFT 权重：`out/debug_stage1_sft_768.pth`
- 阶段 1 debug 续训 checkpoint：`checkpoints/debug_stage1_sft_768_resume.pth`
- 官方 SFT 数据：`dataset/sft_t2t_mini.jsonl`
- 官方 mini 预训练数据：`dataset/pretrain_t2t_mini.jsonl`

这些大文件都应保留在本地，但不要提交到 Git。

## 简化后的目录结构

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

## 保持原样的 MiniMind 内容

以下目录和文件仍作为 MiniMind 原始工程保留：

- `model/`
- `trainer/`
- `scripts/` 中的原始 MiniMind 脚本；
- `dataset/lm_dataset.py`
- `eval_llm.py`
- `README.md`、`README_en.md`、`LICENSE`

## 风险和注意事项

- 本地 Windows 机器建议使用 RTX 5060 单进程训练。
- MiniMind 官方 DDP 初始化使用 NCCL，更适合 Linux 或 WSL2 环境。
- 课题组双 RTX 4090 机器可以在确认 CUDA、PyTorch、NCCL 后使用 `torchrun`。
- Qwen2.5-Math 到 MiniMind 的候选答案级 KL 蒸馏不同于 MiniMind 官方 token 级蒸馏，因为两者 tokenizer 和词表不同。
- 不要提交 checkpoint、完整数据集、模型权重、日志或训练输出。

## Stage 3 low-accuracy diagnosis and distillation direction

Stage 3 produced a valid math SFT baseline from Qwen-generated GSM8K answers,
but the first full evaluation was low (`answer_contains` about `0.04`). The
project now treats this as a diagnosis target inside Stage 3, not as a reason
to change MiniMind core code or start blind retraining.

The diagnosis checks:

- the SFT base checkpoint path and derived `--from_weight`;
- the evaluation checkpoint path, to avoid accidentally loading the official
  SFT baseline instead of the math SFT checkpoint;
- MiniMind conversations schema validity;
- `final_answer` coverage;
- max sequence length truncation risk for long Qwen explanations;
- tokenizer lengths for digits, math symbols, English problems, and Chinese
  final-answer text;
- MiniMind official `SFTDataset` label masks, which supervise assistant spans
  and mask non-assistant tokens with `-100`;
- robust final-answer extraction for `答案是`, English `answer is`, `####`,
  and LaTeX `\boxed{}` formats.

The first 100-row overfit debug run on the raw teacher format did not memorize
the training subset (`answer_contains` stayed around `0.01`). Before increasing
data volume, Stage 3 now normalizes teacher SFT rows so each record has a
top-level `final_answer` and each assistant response ends with a stable
`答案是：<final_answer>` marker.

White-box token-level KL from Qwen2.5-Math to MiniMind is not used. Qwen and
MiniMind have different tokenizers and vocabularies, so token-position logits
are not directly comparable without an additional alignment layer. Instead,
the project uses black-box sequence-level / candidate-level distillation:

- build four candidates per question: Qwen teacher answer, current MiniMind
  answer, gold final-answer response, and a perturbed wrong answer;
- score candidates with gold-answer rules when `final_answer` is available;
- optionally let Qwen act as a judge when a local Qwen model is available;
- fall back to rule-based mock scores when Qwen judge is unavailable;
- convert scored candidates to `chosen` / `rejected` preference pairs for
  later DPO or ranking-loss experiments.

This candidate-level design does not require tokenizer alignment and keeps the
current MiniMind model and official trainer code unchanged.
