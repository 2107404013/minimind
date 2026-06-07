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
