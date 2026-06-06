# MiniMind-MathTutor 当前状态与双 4090 训练计划

## 1. 当前项目状态

当前仓库位于：

```text
D:\pythonfile\minimind小项目\minimind
```

已确认的主要目录结构：

```text
minimind/
├── checkpoints/              # 本地训练 checkpoint，当前未跟踪
├── dataset/                  # 数据集目录，包含 mini 预训练和 SFT 数据
├── images/
├── model/                    # MiniMind 模型结构、Tokenizer、LoRA 实现
├── out/                      # 推理/训练使用的最终权重输出目录，已被 .gitignore 忽略
├── scripts/                  # 推理、WebUI、OpenAI API 服务、模型转换脚本
├── trainer/                  # 预训练、SFT、LoRA、蒸馏、RL 等训练脚本
├── eval_llm.py               # 命令行推理/简单评测入口
├── README.md
└── requirements.txt
```

当前已完成内容：

- MiniMind 项目已 clone 到本地。
- `dataset/pretrain_t2t_mini.jsonl` 已存在。
- `dataset/sft_t2t_mini.jsonl` 已存在。
- 已基于 `pretrain_t2t_mini.jsonl` 跑完 mini 预训练。
- 已生成 `pretrain_768` 权重和 resume checkpoint。
- 本次检查没有修改核心模型结构，没有下载大模型，没有启动训练。

## 2. Mini Pretrain Checkpoint

已找到的 mini pretrain checkpoint：

```text
checkpoints/pretrain_768.pth
checkpoints/pretrain_768_resume.pth
out/pretrain_768.pth
```

文件信息：

```text
checkpoints/pretrain_768.pth         约 131.31 MB
checkpoints/pretrain_768_resume.pth  约 619.00 MB
out/pretrain_768.pth                 约 131.31 MB
```

说明：

- `out/pretrain_768.pth` 是后续 `train_full_sft.py --from_weight pretrain` 默认会读取的权重。
- `checkpoints/pretrain_768_resume.pth` 是包含模型、优化器、epoch/step、world_size 等训练状态的续训 checkpoint。
- 本机当前默认 Python/Anaconda 环境未安装 `torch`，因此本次没有加载张量内容，只基于文件、PyTorch zip 容器结构、脚本保存规则和文件名确认 checkpoint。

是否建议继续使用 mini checkpoint：

- 建议继续使用。它已经是当前项目可复用的阶段产物，适合作为官方 SFT 打底、数学继续预训练小规模验证、数学 SFT 和 LoRA 实验的起点。
- 对 MiniMind-MathTutor 课题来说，先使用 mini checkpoint 可以快速打通完整流水线，尽早发现数据格式、评测、WebUI、RAG 和蒸馏接口问题。

是否建议后续补跑 `pretrain_t2t` full：

- 建议在 mini 流水线跑通、数学数据质量验证通过后再补跑 full。
- full 预训练更适合作为正式实验基座，可以提升通用语言能力和鲁棒性，但会增加数据、时间、显存、断点恢复和实验管理成本。
- 不建议现在直接跳到 full，否则会把训练成本压到流程验证之前。

## 3. 官方脚本位置

官方 pretrain 相关：

```text
trainer/train_pretrain.py
dataset/lm_dataset.py::PretrainDataset
```

官方 SFT 相关：

```text
trainer/train_full_sft.py
dataset/lm_dataset.py::SFTDataset
```

官方 LoRA 相关：

```text
trainer/train_lora.py
model/model_lora.py
scripts/convert_model.py
```

官方 distill 相关：

```text
trainer/train_distillation.py
```

官方 inference / serving / WebUI 相关：

```text
eval_llm.py
scripts/serve_openai_api.py
scripts/chat_api.py
scripts/web_demo.py
scripts/eval_toolcall.py
scripts/convert_model.py
```

## 4. 多 GPU 支持检查

是否使用 `torchrun`：

- README 官方示例支持 `torchrun --nproc_per_node N train_xxx.py`。
- `train_pretrain.py`、`train_full_sft.py`、`train_lora.py`、`train_distillation.py` 等脚本都通过环境变量判断是否处于分布式模式。

是否使用 `DistributedDataParallel`：

- `trainer/train_pretrain.py`
- `trainer/train_full_sft.py`
- `trainer/train_lora.py`
- `trainer/train_distillation.py`
- `trainer/train_dpo.py`
- `trainer/train_ppo.py`
- `trainer/train_grpo.py`
- `trainer/train_agent.py`

这些脚本均导入并在分布式初始化后使用 `torch.nn.parallel.DistributedDataParallel`，数据侧使用 `DistributedSampler`。

分布式初始化位置：

```text
trainer/trainer_utils.py::init_distributed_mode()
```

关键逻辑：

```text
如果环境变量 RANK 不存在，则走单进程。
如果 RANK 存在，则 dist.init_process_group(backend="nccl")。
随后读取 LOCAL_RANK，并 torch.cuda.set_device(local_rank)。
```

是否可以通过 `CUDA_VISIBLE_DEVICES` 指定 GPU：

- 可以。脚本本身不解析 `CUDA_VISIBLE_DEVICES`，但 PyTorch 会在进程启动前读取该环境变量。
- Linux 推荐：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 train_pretrain.py
```

- PowerShell 示例：

```powershell
$env:CUDA_VISIBLE_DEVICES="0,1"
torchrun --nproc_per_node 2 train_pretrain.py
```

注意：

- 当前脚本 DDP backend 写死为 `nccl`。NCCL 通常用于 Linux CUDA 环境；如果在 Windows 原生环境跑双卡，可能会遇到 backend 不可用问题。课题组双 RTX 4090 机器建议使用 Linux 或 WSL2/CUDA 环境验证双卡。

是否支持 resume checkpoint：

- 支持。主要训练脚本提供 `--from_resume 0/1`。
- checkpoint 工具函数为 `trainer/trainer_utils.py::lm_checkpoint()`。
- resume 文件命名规则：

```text
checkpoints/{weight}_{hidden_size}[_moe]_resume.pth
```

例如：

```text
checkpoints/pretrain_768_resume.pth
checkpoints/full_sft_768_resume.pth
```

恢复内容包括：

- model
- optimizer
- epoch
- step
- world_size
- wandb_id
- scaler 或 scheduler 等额外状态

并且当保存时 GPU 数量与当前 GPU 数量不同，`lm_checkpoint()` 会按 `saved_world_size/current_world_size` 转换 step。

## 5. 官方 SFT 数据格式

当前官方 SFT 数据为 JSONL，每行一个样本，核心字段为 `conversations`：

```json
{
  "conversations": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！"},
    {"role": "user", "content": "再见"},
    {"role": "assistant", "content": "再见！"}
  ]
}
```

支持的角色和字段：

- `role`: `system`、`user`、`assistant`、`tool`
- `content`: 当前消息正文
- `reasoning_content`: assistant 的可选思考内容
- `tools`: system 消息中的可选工具定义，字符串或 JSON
- `tool_calls`: assistant 消息中的可选工具调用，字符串或 JSON

训练时：

- `SFTDataset` 使用 tokenizer 的 `apply_chat_template()` 生成完整 prompt。
- loss 只打在 assistant 回复段上，user/system/tool 上的 label 会被置为 `-100`。
- `pre_processing_chat()` 可能以一定概率自动添加 system prompt。
- `post_processing_chat()` 可能处理空 `<think>` 片段。

## 6. 数学数据转换建议

数学继续预训练数据格式：

```json
{"text": "题目、解析、公式、证明或数学知识文本..."}
```

适用场景：

- 教材、讲义、定理说明、例题解析、竞赛题解析、问答拼接后的纯文本。
- 目标是让模型继续吸收数学语言、符号表达和解题语料分布。

数学 SFT 数据格式：

```json
{
  "conversations": [
    {
      "role": "system",
      "content": "你是一个严谨、耐心的数学题问答助手。请给出清晰步骤，并在最后给出答案。"
    },
    {
      "role": "user",
      "content": "解方程：2x + 3 = 11。"
    },
    {
      "role": "assistant",
      "content": "两边同时减去 3，得到 2x = 8。再两边同时除以 2，得到 x = 4。因此答案是 x = 4。"
    }
  ]
}
```

带解析/思考的数学 SFT 可选格式：

```json
{
  "conversations": [
    {"role": "user", "content": "一个等差数列首项为 3，公差为 2，求第 10 项。"},
    {
      "role": "assistant",
      "reasoning_content": "等差数列通项公式为 a_n = a_1 + (n-1)d。代入 a_1=3, d=2, n=10。",
      "content": "由等差数列通项公式 a_n = a_1 + (n-1)d，得到 a_10 = 3 + 9*2 = 21。因此第 10 项是 21。"
    }
  ]
}
```

转换规则建议：

- 每道题至少保留 `question`、`answer`，最好保留 `solution` 或 `analysis`。
- 用户侧只放题目、条件、选项和必要上下文。
- assistant 侧放标准解法和最终答案，避免把数据集元信息、评分标签、来源路径混入回答。
- 统一最终答案标记，例如“因此答案是 ...”或“最终答案：...”，方便后续评测。
- 选择题保留选项，并要求 assistant 说明理由后给出选项字母。
- 证明题保留完整证明链条，避免只有结论。
- 对教师模型生成数据，应额外保存原始教师输出、过滤状态、来源题 ID，但这些元数据不要放进 `conversations`，可放旁路字段或单独 manifest。

## 7. 双 RTX 4090 使用方案

总体原则：

- 先用 mini checkpoint 跑通全流程，再扩大数据和训练量。
- 能双卡稳定加速的阶段优先双卡。
- 数据生成、RAG 构建、WebUI 和部分评测更适合单卡或 CPU/GPU 混合。
- 每个阶段结束后只提交代码、文档、转换脚本和配置，不提交 checkpoint、dataset、outputs。

阶段规划：

| 阶段 | 建议 GPU | 输入 | 输出 | 推荐 commit |
|---|---:|---|---|---|
| 0. 当前状态文档 | 无 | 本地仓库、checkpoint、脚本 | `docs/current_status_and_2gpu_plan.md` | `docs: add current status and dual 4090 training plan` |
| 1. 官方 SFT 打底 | 双卡优先 | `out/pretrain_768.pth` + `dataset/sft_t2t_mini.jsonl` | `out/full_sft_768.pth`、`checkpoints/full_sft_768_resume.pth` | `train: finish official mini sft baseline` |
| 2. 数学数据转换 | 无或 CPU | 原始数学题库、教材/解析数据、教师生成原始结果 | MiniMind pretrain JSONL、math SFT JSONL、数据报告 | `data: add math dataset conversion pipeline` |
| 3. 数学继续预训练 | 双卡优先 | mini pretrain 或 full pretrain 权重 + math pretrain JSONL | `out/pretrain_math_768.pth`、resume checkpoint | `train: finish math continued pretraining` |
| 4. 数学 SFT | 双卡优先 | 数学继续预训练权重 + math SFT JSONL | `out/full_sft_math_768.pth`、resume checkpoint | `train: finish math sft` |
| 5. LoRA 微调 | 单卡优先，数据大时双卡 | SFT 基座 + 高质量垂域 math SFT JSONL | `out/lora_math_768.pth`、resume checkpoint | `train: finish math lora tuning` |
| 6. Qwen2.5-Math-7B-Instruct 教师生成数据 | 单卡可行，批量吞吐可双卡 | 原始题目池、prompt 模板、过滤规则 | teacher raw JSONL、clean SFT JSONL、过滤报告 | `data: generate qwen math tutor samples` |
| 7. 候选答案级 KL 蒸馏 | 需要新增实现后再定；建议双卡 | 候选答案、教师分数/概率、学生 logprob | 蒸馏后 math tutor 权重 | `train: add candidate answer kl distillation` |
| 8. RAG | 单卡或 CPU | 数学知识库、教材切片、向量模型/索引 | 文档切片、向量索引、检索 API | `feat: add math rag pipeline` |
| 9. 统一评测 | 单卡 | 固定评测集、模型权重、RAG 开关配置 | metrics JSON/CSV、评测报告 | `eval: add unified math tutor evaluation` |
| 10. WebUI 展示 | 单卡 | 最终模型、RAG API、评测样例 | 数学问答 WebUI | `feat: add math tutor webui` |

推荐训练入口示例：

```bash
# 在 trainer 目录下执行，双卡官方 SFT
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 train_full_sft.py \
  --from_weight pretrain \
  --data_path ../dataset/sft_t2t_mini.jsonl \
  --save_weight full_sft \
  --from_resume 0
```

```bash
# 数学继续预训练，后续有 math pretrain 数据后再执行
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 train_pretrain.py \
  --from_weight pretrain \
  --data_path ../dataset/math_pretrain.jsonl \
  --save_weight pretrain_math \
  --from_resume 0
```

```bash
# 数学 SFT，后续有 math SFT 数据后再执行
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 train_full_sft.py \
  --from_weight pretrain_math \
  --data_path ../dataset/math_sft.jsonl \
  --save_weight full_sft_math \
  --from_resume 0
```

## 8. 候选答案级 KL 蒸馏现状

当前官方 `trainer/train_distillation.py` 支持的是 token 级白盒蒸馏：

- 学生和教师均为 MiniMind 结构。
- 通过 `student_logits` 与 `teacher_logits` 计算 KL。
- 默认适合相同或可截断到相同 vocab 的 MiniMind teacher/student。

它不等价于 Qwen2.5-Math-7B-Instruct 到 MiniMind 的候选答案级 KL 蒸馏，原因：

- Qwen2.5-Math 与 MiniMind tokenizer/vocab 不同，不能直接逐 token 对齐 logits。
- 候选答案级 KL 需要先固定候选答案集合，再比较教师和学生对整段候选答案的序列级概率或归一化偏好分布。
- 当前仓库没有现成的数据结构和训练脚本直接支持候选答案级 KL。

后续建议：

- 先用 Qwen2.5-Math 生成多个候选答案或对候选答案打分。
- 对每道题构造候选答案分布，例如 `p_teacher(candidate_i | question)`。
- 让学生计算每个候选答案的平均 token logprob 或长度归一化 logprob。
- 在候选答案维度上做 KL / CE / ranking loss。
- 将此作为新脚本或新 loss 增量实现，不修改核心模型结构。

## 9. 风险点和注意事项

训练环境风险：

- 当前脚本 DDP 使用 `nccl`，Windows 原生环境可能无法双卡运行。建议在 Linux/WSL2 CUDA 环境验证。
- 仓库根目录存在一个名为 `python` 的空文件，可能影响命令查找顺序。运行脚本时建议显式确认 `python` 指向正确环境。
- 当前本机默认 Python/Anaconda 未安装 `torch`，正式训练前需要确认课题组机器环境依赖完整。

数据风险：

- 当前 `dataset/pretrain_t2t_mini.jsonl` 和 `dataset/sft_t2t_mini.jsonl` 很大且未跟踪，不要提交。
- 数学数据需要严格去重，尤其要避免评测集泄漏进 SFT/LoRA/蒸馏训练。
- 教师生成数据要做格式校验、答案校验、长度过滤和重复过滤。
- 数学题中 LaTeX、中文标点、选项编号、最终答案格式需要统一。

训练风险：

- mini checkpoint 适合打通流程，但数学能力上限有限。
- 数学继续预训练可能损伤对话能力，后面必须接官方 SFT 或混合 SFT 修复交互风格。
- 数学 SFT 数据如果只有短答案，模型会学成只报结论；如果全是长 CoT，又可能变得冗长。
- LoRA 适合低成本领域适配，但能力上限受基座影响。
- 蒸馏阶段如果 teacher 输出有错，学生会稳定学习错误模式。

工程与版本管理风险：

- 不提交 `checkpoints/`、`dataset/*.jsonl`、`out/`。
- 每个阶段结束后单独 commit，commit 中只包含代码、配置、文档、评测脚本或小型样例。
- 大文件建议通过外部存储、DVC、Git LFS 或课题组共享盘管理。
- 训练参数、数据版本、随机种子、GPU 数量、commit hash 需要写入实验记录。

## 10. 当前建议路线

推荐下一步顺序：

1. 使用当前 `out/pretrain_768.pth` 跑官方 `sft_t2t_mini.jsonl`，得到第一个可对话基线。
2. 编写数学数据转换和校验脚本，先生成小规模 `math_pretrain` 与 `math_sft` 样例。
3. 用 mini checkpoint 先完成数学继续预训练、数学 SFT、LoRA、评测、WebUI、RAG 的端到端链路。
4. 链路稳定后再补跑 `pretrain_t2t` full，并复现实验。
5. 最后实现 Qwen2.5-Math 教师数据生成和候选答案级 KL 蒸馏。
