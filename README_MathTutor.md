# MiniMind-Math-Lab：小参数语言模型数学能力边界实验

MiniMind-Math-Lab 是一个基于 MiniMind 64M 的数学能力边界实验项目。项目价值不在于追求 GSM8K 高正确率，而在于完整展示一个小参数语言模型从零预训练、官方 SFT、数学 SFT、LoRA 准备、教师数据构建、分难度评测和实验分析的工程闭环。

## 项目简历描述

MiniMind-Math-Lab：基于 MiniMind 64M 构建小参数语言模型数学能力边界实验，完成 pretrain、SFT、数学数据构建和分难度评测；使用 Qwen2.5-Math-7B-Instruct 生成教师数据，分析 64M 小模型在 arithmetic、template word problem、GSM8K easy 等任务上的表现，并总结其在复杂多步数学推理上的容量瓶颈。

## 当前定位

MiniMind 64M 的主要价值是学习和复现 LLM 全流程。它适合用来观察小模型在不同数学任务上的能力边界，而不是直接作为复杂多步数学推理模型。

本项目保留 MiniMind 原始模型结构和官方训练脚本，只增加少量轻量组件：

- 数学数据转换与教师数据整理；
- SFT / LoRA 训练命令封装；
- answer extraction 与 debug predictions；
- arithmetic、template word problem、GSM8K easy/medium、hard reasoning 分难度评测；
- 实验记录和边界分析。

## 适合做什么

- 从零预训练 baseline；
- 官方 SFT / 数学 SFT / LoRA 流程复现；
- 小模型数学能力边界分析；
- 数据难度分层评测；
- 教师模型数据生成与小模型学习效果对比。

当前边界评测使用五档固定题集：`arithmetic`、`template_word_problem`、`gsm8k_easy`、`gsm8k_medium`、`hard_reasoning`。每个 checkpoint 都跑同一份边界集，报告中的 `boundary_summary.stable_boundary` 用来说明模型稳定通过的最高难度档。

## 不适合做什么

- 不再追求 GSM8K 高正确率；
- 不把 MiniMind 64M 直接定位为复杂多步数学推理模型；
- 不通过堆更多盲目训练来强行追分；
- 不修改 MiniMind 原始核心模型结构。

## 当前实验结论

已经完成 MiniMind 64M 的 mini 预训练，并进行过官方 SFT 与数学 SFT 尝试。500 条 Qwen2.5-Math-7B-Instruct 生成的 GSM8K SFT 数据能够让模型记住训练子集，但对未见过复杂 GSM8K 样本泛化较弱。当前结论是：MiniMind 64M 更适合作为小模型数学能力边界实验平台，而不是高准确率数学问答助手。

## 后续方向

后续会新建 Qwen-MathTutor 项目，将本项目中已经验证过的数据构建、教师生成、debug prediction 和评测框架迁移到更强的底座模型上。本仓库继续保留为 MiniMind 64M 小模型实验、复现和简历展示项目。

## 入口文件

- 项目说明：`docs/PROJECT.md`
- 运行手册：`docs/RUNBOOK.md`
- 实验记录：`docs/EXPERIMENT_LOG.md`
- 主配置：`configs/math_tutor.yaml`

不要提交 checkpoint、`outputs/`、`out/`、`checkpoints/` 或大规模 JSONL 数据。
