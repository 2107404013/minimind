# MiniMind-MathTutor 运行手册

## 0. 环境检查

本机建议使用 MiniMind Conda 环境：

```powershell
D:\APP\Anaconda3\envs\minimind\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 1. 转换数学样例数据

```powershell
D:\APP\Anaconda3\envs\minimind\python.exe scripts\build_math_data.py `
  --input sample_data\math_raw_sample.jsonl `
  --output sample_data\math_sft_sample.jsonl
```

## 2. 准备教师模型请求

这一步不会下载或加载 Qwen，只会把数学题整理成教师模型生成所需的 prompt 记录。

```powershell
D:\APP\Anaconda3\envs\minimind\python.exe scripts\generate_teacher.py `
  --input sample_data\math_raw_sample.jsonl `
  --output outputs\teacher\qwen_math_requests.jsonl
```

## 3. 官方 SFT mini baseline

辅助脚本默认只打印官方训练命令：

```powershell
D:\APP\Anaconda3\envs\minimind\python.exe scripts\train_math_sft.py --mode official_sft --dry_run
```

本地 RTX 5060 的官方训练命令如下：

```powershell
cd trainer
$env:CUDA_VISIBLE_DEVICES = "0"
D:\APP\Anaconda3\envs\minimind\python.exe train_full_sft.py `
  --data_path ../dataset/sft_t2t_mini.jsonl `
  --save_dir ../out `
  --save_weight full_sft `
  --from_weight pretrain `
  --from_resume 0 `
  --epochs 2 `
  --batch_size 8 `
  --accumulation_steps 4 `
  --learning_rate 1e-5 `
  --max_seq_len 768 `
  --save_interval 1000 `
  --num_workers 4 `
  --dtype float16 `
  --device cuda:0
```

## 4. 数学 SFT

当真实数学 SFT 数据准备到 `data/processed/math_sft.jsonl` 后，先打印命令确认：

```powershell
D:\APP\Anaconda3\envs\minimind\python.exe scripts\train_math_sft.py --mode math_sft --dry_run
```

确认无误后再显式执行：

```powershell
D:\APP\Anaconda3\envs\minimind\python.exe scripts\train_math_sft.py --mode math_sft --run
```

## 5. 数学 LoRA

先打印命令：

```powershell
D:\APP\Anaconda3\envs\minimind\python.exe scripts\train_math_lora.py
```

只有准备开始训练时才加 `--run`。

## 6. 候选答案级 KL 数据准备

从已经打分的候选答案中准备概率分布 JSONL：

```powershell
D:\APP\Anaconda3\envs\minimind\python.exe scripts\train_math_kl.py `
  --input data\processed\candidate_scores.jsonl `
  --output data\processed\math_candidate_kl.jsonl
```

当前脚本只做后续 KL 实现所需的数据准备，不会启动蒸馏训练。

## 7. 构建样例 RAG 索引

```powershell
D:\APP\Anaconda3\envs\minimind\python.exe scripts\build_rag_index.py `
  --docs-dir sample_docs `
  --output outputs\rag\math_notes_sample.index.json
```

## 8. 评测

对包含 `prediction` 和 `answer` 字段的 JSONL 进行评测：

```powershell
D:\APP\Anaconda3\envs\minimind\python.exe scripts\eval_math.py `
  --input outputs\eval\predictions.jsonl `
  --output outputs\eval\math_eval_report.json
```

## 推荐阶段顺序

1. 官方 SFT mini baseline。
2. 数学原始数据转换与校验。
3. 教师模型 prompt 生成与教师数据清洗。
4. 如果有足够干净的数学文本，再做数学继续预训练。
5. 数学 SFT。
6. 数学 LoRA。
7. 候选答案级 KL 蒸馏。
8. RAG 索引和检索集成。
9. 统一评测。
10. WebUI。
# Stage 1: Official SFT mini baseline

Goal: train a general MiniMind chat baseline from the completed
`pretrain_t2t_mini` checkpoint with MiniMind official `sft_t2t_mini.jsonl`.
This is not math SFT.

The wrapper only builds and optionally runs the official MiniMind SFT command.
It does not rewrite `trainer/train_full_sft.py`.

Dry run, no training:

```powershell
D:\APP\Anaconda3\envs\minimind\python.exe scripts\train_math_sft.py `
  --config configs\math_tutor.yaml `
  --mode official_sft `
  --dry_run
```

Single-GPU local RTX 5060 command:

```powershell
$env:CUDA_VISIBLE_DEVICES = "0"
D:\APP\Anaconda3\envs\minimind\python.exe scripts\train_math_sft.py `
  --config configs\math_tutor.yaml `
  --mode official_sft `
  --run
```

Equivalent direct official MiniMind command:

```powershell
cd trainer
$env:CUDA_VISIBLE_DEVICES = "0"
D:\APP\Anaconda3\envs\minimind\python.exe train_full_sft.py `
  --data_path ../dataset/sft_t2t_mini.jsonl `
  --save_dir ../out `
  --save_weight full_sft `
  --from_weight pretrain `
  --from_resume 0 `
  --epochs 2 `
  --batch_size 8 `
  --accumulation_steps 4 `
  --learning_rate 1e-5 `
  --max_seq_len 768 `
  --save_interval 1000 `
  --num_workers 4 `
  --dtype float16 `
  --device cuda:0
```

Two-GPU command for the lab RTX 4090 Linux or WSL2 machine:

```bash
cd trainer
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 train_full_sft.py \
  --data_path ../dataset/sft_t2t_mini.jsonl \
  --save_dir ../out \
  --save_weight full_sft \
  --from_weight pretrain \
  --from_resume 0 \
  --epochs 2 \
  --batch_size 8 \
  --accumulation_steps 4 \
  --learning_rate 1e-5 \
  --max_seq_len 768 \
  --save_interval 1000 \
  --num_workers 4 \
  --dtype float16
```

MiniMind official SFT supports DDP because `train_full_sft.py` initializes
distributed mode when `RANK` exists and wraps the model with
`DistributedDataParallel`. On native Windows, the script uses NCCL for DDP, so
use the single-GPU command unless running inside Linux or WSL2 with CUDA/NCCL
working.

Stage 1 completion checks:

```powershell
Get-Item out\full_sft_768.pth
Get-Item checkpoints\full_sft_768_resume.pth
Get-Content outputs\logs\stage1_official_sft_*.log -Tail 40
Select-String -Path outputs\logs\stage1_official_sft_*.log `
  -Pattern "error","traceback","nan","out of memory","oom","RuntimeError"
```

The expected successful end state is:

- `out/full_sft_768.pth` exists.
- `checkpoints/full_sft_768_resume.pth` exists.
- The final log reaches `Epoch:[2/2](56608/56608)`.
- The error keyword check returns no matches.

Short local dialogue validation can be done by loading `full_sft` with
`eval_llm.py` or by running the same official MiniMind model loading logic in a
small one-shot Python snippet. The first validated prompt was:

```text
你好，请简单介绍一下你自己。
```

The model produced a coherent MiniMind self-introduction. One minor inference
artifact was observed: the answer included an extra `</think>` token before
repeating the response.

# Stage 2: Math data and Qwen teacher workflow

Goal: build compact math SFT data in MiniMind conversations format, then use a
local or already cached Qwen2.5-Math-7B-Instruct checkpoint to generate teacher
solutions. This stage does not modify MiniMind core model code and does not
start training.

Build the sample math SFT file:

```powershell
python scripts/build_math_data.py --config configs/math_tutor.yaml --sample
```

Build a full math SFT file from configured paths:

```powershell
python scripts/build_math_data.py --config configs/math_tutor.yaml
```

Write train/valid/test split files when a real raw dataset is ready:

```powershell
python scripts/build_math_data.py --config configs/math_tutor.yaml --write-splits
```

Generate Qwen teacher answers with a local path or cached Hugging Face model.
By default `local_files_only: true` is used, so this command will not download
model weights:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/generate_teacher.py \
  --config configs/math_tutor.yaml \
  --input sample_data/math_sft_sample.jsonl \
  --output outputs/sample_teacher.jsonl \
  --limit 3
```

Use 4bit loading when the environment has compatible `bitsandbytes` support:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/generate_teacher.py \
  --config configs/math_tutor.yaml \
  --input sample_data/math_sft_sample.jsonl \
  --output outputs/sample_teacher.jsonl \
  --limit 3 \
  --load-in-4bit
```

If the model is not already local or cached, the teacher command should fail
before generation instead of downloading weights. Failed per-sample generations
are appended to `outputs/failed_teacher.jsonl`.

Write Qwen teacher answers directly in the narrow schema required by MiniMind's
official `SFTDataset`:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/generate_teacher.py \
  --config configs/math_tutor.yaml \
  --input outputs/math_sft_100.jsonl \
  --output outputs/teacher_100_train.jsonl \
  --limit 100 \
  --official-sft-compatible
```

The resulting file can be passed directly to `trainer/train_full_sft.py`; no
manual `*_sft_compat.jsonl` conversion is needed.

# Stage 3: Math SFT

Goal: fine-tune MiniMind-MathTutor on math SFT data while reusing the official
MiniMind `trainer/train_full_sft.py` logic. This wrapper does not create a new
training framework.

Dry run only, no training:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_math_sft.py \
  --config configs/math_tutor.yaml \
  --mode math_sft \
  --dry_run
```

The dry run prints the delegated official command. For `math_sft`, the wrapper
maps:

- `training.math_sft.train_file` to `--data_path`
- `training.math_sft.output_dir` to `--save_dir`
- `training.math_sft.save_weight` to `--save_weight`
- `training.math_sft.from_weight` to `--from_weight`

`valid_file`, `warmup_ratio`, and `eval_steps` are recorded in the YAML for
experiment tracking, but MiniMind's official SFT script does not consume them.

When ready for a tiny sample run on the remote server, keep the dataset small
and run explicitly with `--run`:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_math_sft.py \
  --config configs/math_tutor.yaml \
  --mode math_sft \
  --run
```

Two-GPU remote command, only after the dry run command looks correct:

```bash
cd trainer
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 train_full_sft.py \
  --data_path ../outputs/teacher_100_train.jsonl \
  --save_dir ../out \
  --save_weight full_sft_math \
  --from_weight full_sft \
  --from_resume 0 \
  --epochs 1 \
  --batch_size 8 \
  --accumulation_steps 4 \
  --learning_rate 1e-5 \
  --max_seq_len 768 \
  --save_interval 1000 \
  --num_workers 4 \
  --dtype float16
```

Minimal math evaluation:

```bash
python scripts/eval_math.py \
  --config configs/math_tutor.yaml \
  --mode math_sft \
  --sample
```

In `--sample` mode, if `out/full_sft_math_768.pth` does not exist yet, the
evaluator scores the stored sample assistant answers instead of loading a
checkpoint. Once the math SFT checkpoint exists, the same command loads the
MiniMind checkpoint and generates answers.
