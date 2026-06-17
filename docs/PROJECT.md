# MiniMind-Math-Lab 项目说明

## 项目名称

MiniMind-Math-Lab：小参数语言模型数学能力边界实验

## 项目定位

MiniMind-Math-Lab 基于 MiniMind 64M，目标是研究小参数语言模型在数学问答任务上的能力边界。项目不再定位为“高准确率数学题问答助手”，也不继续原来的多阶段追分计划。

MiniMind 64M 的价值在于学习和复现 LLM 全流程：从零预训练、官方 SFT、数学数据构建、数学 SFT、LoRA 准备、教师数据生成、分难度评测和实验分析。它可以帮助观察小模型在不同数学任务上的能力、瓶颈和数据依赖。

## 适合的研究问题

- 从零预训练 baseline 能得到什么样的语言能力；
- 官方 SFT 后，小模型的指令跟随和输出格式能提升到什么程度；
- 数学 SFT 对 arithmetic、template word problem、GSM8K easy 的帮助有多大；
- 64M 参数规模在 GSM8K medium 和 hard reasoning 上的瓶颈在哪里；
- 长 CoT、短步骤、纯 final answer 等数据格式对小模型训练有什么影响；
- LoRA 是否能作为低成本任务适配工具，而不是容量扩展手段。

## 不追求的目标

- 不追求 GSM8K 高正确率；
- 不把 MiniMind 64M 作为复杂多步数学推理模型；
- 不通过盲目扩大训练轮数或数据量来强行追分；
- 不修改 MiniMind 原始核心模型结构；
- 不提交 checkpoint、outputs、大数据或模型权重。

## 当前实验状态

当前仓库基于 MiniMind 64M，已经完成 `pretrain_t2t_mini` 预训练，并做过官方 SFT 与数学 SFT 尝试。近期实验使用 Qwen2.5-Math-7B-Instruct 生成 GSM8K 教师答案，构造 MiniMind conversations 格式数据，并训练 500 条数学 SFT 样本。

观察结果：

- 500 条数据上的强过拟合可以让模型记住训练样本；
- 对未见过的复杂 GSM8K 样本，正确率仍然较低；
- 长 Qwen CoT 对 64M 模型过难；
- 纯 final answer 目标能改善格式和记忆，但不能带来可靠泛化；
- 当前更合理的结论是记录能力边界，而不是继续强行追求高分。

## 分难度评测

评测按以下难度桶汇总：

- `arithmetic`：简单四则运算；
- `template_word_problem`：固定模板应用题；
- `gsm8k_easy`：一到两步简单 GSM8K；
- `gsm8k_medium`：多步 GSM8K；
- `hard_reasoning`：复杂推理题。

核心指标：

- overall accuracy；
- accuracy by difficulty；
- answer contains；
- final answer format rate；
- invalid output rate；
- average output length；
- average latency。

评测报告和 debug predictions 写入 `outputs/`，不提交到 Git。

当前边界判定采用一个固定小评测集：每个难度档 10 道题，总计 50 道。每次训练或微调后都跑同一份 `outputs/eval_boundary_core.jsonl`，报告会生成 `boundary_summary`：

- `stable_boundary`：accuracy 达到 `0.8` 的最高稳定难度档；
- `first_unstable_bucket`：首次进入 partial 或 fail 的难度档；
- `partial`：accuracy 在 `0.5` 到 `0.8` 之间，说明模型有部分能力但不稳定；
- `fail`：accuracy 低于 `0.5`，说明该档已超过当前 checkpoint 的可靠能力范围。

这个边界集的目的不是替代 GSM8K 标准评测，而是快速、低成本地回答“MiniMind 64M 现在到底能稳到哪一级”。

## 后续迁移

后续会新建 Qwen-MathTutor 项目，将本项目的数据构建、教师生成、debug prediction 和评测框架迁移到更强底座模型上。本仓库继续作为 MiniMind 64M 小模型数学能力边界实验与简历展示项目保留。
