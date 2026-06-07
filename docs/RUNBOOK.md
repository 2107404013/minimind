# MiniMind-MathTutor Runbook

## 0. Setup check

Use the MiniMind Conda environment on this machine:

```powershell
D:\APP\Anaconda3\envs\minimind\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 1. Convert a raw math sample

```powershell
D:\APP\Anaconda3\envs\minimind\python.exe scripts\build_math_data.py `
  --input sample_data\math_raw_sample.jsonl `
  --output sample_data\math_sft_sample.jsonl
```

## 2. Prepare teacher prompts

This does not download or load Qwen. It only writes prompt records.

```powershell
D:\APP\Anaconda3\envs\minimind\python.exe scripts\generate_teacher.py `
  --input sample_data\math_raw_sample.jsonl `
  --output outputs\teacher\qwen_math_requests.jsonl
```

## 3. Official SFT mini baseline

The helper prints the official command by default:

```powershell
D:\APP\Anaconda3\envs\minimind\python.exe -m math_tutor.train --task stage1_sft
```

Direct official command for the local RTX 5060:

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

## 4. Math SFT

After real math SFT data exists at `data/processed/math_sft.jsonl`, print the
command:

```powershell
D:\APP\Anaconda3\envs\minimind\python.exe scripts\train_math_sft.py
```

Execute only when ready:

```powershell
D:\APP\Anaconda3\envs\minimind\python.exe scripts\train_math_sft.py --run
```

## 5. Math LoRA

```powershell
D:\APP\Anaconda3\envs\minimind\python.exe scripts\train_math_lora.py
```

Add `--run` only when training should start.

## 6. Candidate-answer KL preparation

Prepare candidate probability JSONL from scored candidate answers:

```powershell
D:\APP\Anaconda3\envs\minimind\python.exe scripts\train_math_kl.py `
  --input data\processed\candidate_scores.jsonl `
  --output data\processed\math_candidate_kl.jsonl
```

This only prepares data for the later KL implementation.

## 7. Build a sample RAG index

```powershell
D:\APP\Anaconda3\envs\minimind\python.exe scripts\build_rag_index.py `
  --docs-dir sample_docs `
  --output outputs\rag\math_notes_sample.index.json
```

## 8. Evaluation

For a prediction JSONL with `prediction` and `answer` fields:

```powershell
D:\APP\Anaconda3\envs\minimind\python.exe scripts\eval_math.py `
  --input outputs\eval\predictions.jsonl `
  --output outputs\eval\math_eval_report.json
```

## Recommended stage order

1. Official SFT mini baseline.
2. Math raw data conversion and validation.
3. Teacher prompt generation and teacher-data cleaning.
4. Math continued pretraining if enough clean math text exists.
5. Math SFT.
6. Math LoRA.
7. Candidate-answer KL distillation.
8. RAG index and retrieval integration.
9. Unified evaluation.
10. WebUI.
