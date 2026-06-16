# MiniMind-Math-Lab 运行手册

本手册只保留 MiniMind 64M 数学能力边界实验所需的最少命令。不要启动盲目全量训练；checkpoint、`outputs/`、`out/`、`checkpoints/` 和大规模 JSONL 数据不提交。

## 1. 构建 sample 数学数据

```bash
python scripts/build_math_data.py \
  --config configs/math_tutor.yaml \
  --sample
```

也可以显式指定输入输出：

```bash
python scripts/build_math_data.py \
  --input sample_data/math_raw_sample.jsonl \
  --output sample_data/math_sft_sample.jsonl
```

## 2. 准备 500 条 SFT 数据

教师数据由 Qwen2.5-Math-7B-Instruct 生成，训练文件应保持 MiniMind conversations 或 official SFT compatible 格式。运行产物写入 `outputs/`。

将已有教师数据压缩成 compact final-answer 目标：

```bash
python scripts/build_math_data.py \
  --config configs/math_tutor.yaml \
  --input outputs/teacher_5000_train.jsonl \
  --output outputs/teacher_5000_train_compact.jsonl \
  --compact-final-target \
  --official-sft-compatible
```

截取 500 条实验数据：

```bash
head -n 500 outputs/teacher_5000_train_compact.jsonl \
  > outputs/teacher_500_train_compact.jsonl
```

## 3. 运行 sample SFT 或 500 条 SFT

先 dry run，只检查命令和路径：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_math_sft.py \
  --config configs/math_tutor.yaml \
  --mode math_sft \
  --dry_run
```

确认无误后再显式运行小规模 SFT：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_math_sft.py \
  --config configs/math_tutor.yaml \
  --mode math_sft \
  --run
```

如需在双 4090 Linux 环境直接调用 MiniMind 官方 SFT 脚本：

```bash
cd trainer
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 train_full_sft.py \
  --data_path ../outputs/teacher_500_train_compact.jsonl \
  --save_dir ../out \
  --save_weight full_sft_math_compact_500 \
  --from_weight full_sft_official \
  --from_resume 0 \
  --epochs 1 \
  --batch_size 1 \
  --accumulation_steps 4 \
  --learning_rate 1e-5 \
  --max_seq_len 768 \
  --save_interval 1000 \
  --num_workers 4 \
  --dtype float16
```

## 4. 运行分难度评测

评测会输出 overall accuracy、accuracy by difficulty、answer contains、final answer format rate、invalid output rate、average output length 和 average latency。

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python scripts/eval_math.py \
  --config configs/math_tutor.yaml \
  --mode math_sft \
  --input outputs/eval_difficulty_sample.jsonl \
  --output outputs/math_eval_report.json \
  --debug
```

如果 checkpoint 尚不存在，可以用 sample fallback 检查数据和答案抽取：

```bash
PYTHONPATH=src python scripts/eval_math.py \
  --config configs/math_tutor.yaml \
  --mode math_sft \
  --sample \
  --debug
```

## 5. 查看 debug predictions

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path("outputs/debug_predictions.jsonl")
for i, line in enumerate(p.open(encoding="utf-8")):
    if i >= 20:
        break
    obj = json.loads(line)
    print("=" * 80)
    print("CASE", i)
    print("DIFFICULTY:", obj.get("difficulty"))
    print("GOLD:", obj.get("gold_answer"))
    print("EXTRACTED:", repr(obj.get("extracted_answer")))
    print("CORRECT:", obj.get("is_correct"), "INVALID:", obj.get("invalid_output"))
    print("OUTPUT:", repr((obj.get("model_output") or "")[:500]))
PY
```

## 6. Git 注意事项

只提交代码、配置和文档。不要提交：

- `outputs/`
- `out/`
- `checkpoints/`
- `.pth` 权重
- 训练日志
- 大规模 JSONL 数据
