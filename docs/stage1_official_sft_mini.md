# Stage 1: official SFT mini baseline

## Goal

Use MiniMind's official `train_full_sft.py` script to continue from the locally trained `pretrain_t2t_mini` checkpoint and run the official `sft_t2t_mini.jsonl` supervised fine-tuning stage. The target output is a `full_sft_768.pth` baseline that can follow the MiniMind chat template and produce normal assistant-style replies.

This stage does not modify the core model structure or rewrite the training framework.

## Current environment

- Repository: the MiniMind git repository root
- Training Python: `D:\APP\Anaconda3\envs\minimind\python.exe`
- Detected PyTorch: `torch 2.11.0+cu128`
- Detected GPU: `NVIDIA GeForce RTX 5060 Laptop GPU`
- Detected CUDA devices in this environment: `1`

The default `python` on PATH points to `D:\APP\Anaconda3\python.exe` and does not have `torch` installed, so commands in this stage should explicitly use the `minimind` Conda environment or activate it first.

## Official script read

The official SFT entry is:

```text
trainer/train_full_sft.py
```

Important implementation details:

- It uses `argparse`; there is no native YAML config loader.
- It loads SFT data with `SFTDataset` from `dataset/lm_dataset.py`.
- It loads the base weight through `init_model(lm_config, args.from_weight, device=args.device)`.
- With the default `--from_weight pretrain`, the loader reads `../out/pretrain_768.pth` when the script is launched from `trainer/`.
- It writes the normal output checkpoint to `../out/{save_weight}_768.pth`.
- It writes resumable checkpoints through `lm_checkpoint(..., save_dir='../checkpoints')`, for example `../checkpoints/full_sft_768_resume.pth`.
- It supports resume with `--from_resume 1`.
- It supports DDP when launched through `torchrun`, because the script initializes distributed mode when `RANK` exists and wraps the model with `DistributedDataParallel`.

## Official recommended command

The README recommends running all training scripts from the `trainer/` directory.

Official SFT commands:

```powershell
cd <repo-root>\trainer
python train_full_sft.py
```

or:

```powershell
cd <repo-root>\trainer
torchrun --nproc_per_node 1 train_full_sft.py
```

For this stage, use the single-card 5060 path below.

## Stage config

Stage parameters are recorded in:

```text
configs/sft_t2t_mini_1gpu_5060.yaml
```

The requested two-GPU template is also recorded for the later lab 4090 machine:

```text
configs/sft_t2t_mini_2gpu.yaml
```

The MiniMind script does not read YAML directly, so these YAML files are reproducibility records and should be translated to CLI flags. This stage's actual local run uses the single-5060 configuration.

Recommended full Stage 1 single-GPU command:

```powershell
cd <repo-root>\trainer
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
  --log_interval 100 `
  --num_workers 4 `
  --dtype float16 `
  --device cuda:0
```

Resume command if training is interrupted:

```powershell
cd <repo-root>\trainer
$env:CUDA_VISIBLE_DEVICES = "0"
D:\APP\Anaconda3\envs\minimind\python.exe train_full_sft.py `
  --data_path ../dataset/sft_t2t_mini.jsonl `
  --save_dir ../out `
  --save_weight full_sft `
  --from_weight pretrain `
  --from_resume 1 `
  --epochs 2 `
  --batch_size 8 `
  --accumulation_steps 4 `
  --learning_rate 1e-5 `
  --max_seq_len 768 `
  --save_interval 1000 `
  --log_interval 100 `
  --num_workers 4 `
  --dtype float16 `
  --device cuda:0
```

If the 5060 laptop GPU reports OOM, reduce `--batch_size` to `4`, then `2`, then `1`, and increase `--accumulation_steps` to keep the effective batch size near the original target.

## Torchrun support and this stage's GPU choice

`train_full_sft.py` supports `torchrun` in general:

```powershell
torchrun --nproc_per_node N train_full_sft.py
```

However, this stage should use one local 5060 GPU. On native Windows, the current MiniMind distributed initialization uses `backend="nccl"` when `RANK` is set. NCCL is the normal Linux CUDA backend and may fail on native Windows. Therefore, for this machine and this stage, prefer the non-DDP single-process command:

```powershell
D:\APP\Anaconda3\envs\minimind\python.exe train_full_sft.py --device cuda:0
```

On the two RTX 4090 Linux/WSL2 machine, the later dual-GPU command can use:

```bash
cd trainer
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 train_full_sft.py
```

## Data and checkpoints

Input SFT data:

```text
dataset/sft_t2t_mini.jsonl
```

Base checkpoint:

```text
out/pretrain_768.pth
```

Output checkpoint:

```text
out/full_sft_768.pth
```

Resume checkpoint:

```text
checkpoints/full_sft_768_resume.pth
```

Do not commit the dataset, checkpoints, `out/`, or `checkpoints/` files.

## Dry run performed

A small debug SFT run was performed without modifying the training script.

Temporary debug data:

```text
out/debug_sft_t2t_mini_16.jsonl
```

Debug command:

```powershell
cd <repo-root>\trainer
$env:CUDA_VISIBLE_DEVICES = "0"
D:\APP\Anaconda3\envs\minimind\python.exe train_full_sft.py `
  --data_path ../out/debug_sft_t2t_mini_16.jsonl `
  --save_dir ../out `
  --save_weight debug_stage1_sft `
  --from_weight pretrain `
  --from_resume 0 `
  --epochs 1 `
  --batch_size 1 `
  --accumulation_steps 4 `
  --learning_rate 1e-5 `
  --max_seq_len 128 `
  --save_interval 9999 `
  --log_interval 1 `
  --num_workers 0 `
  --dtype float16 `
  --device cuda:0
```

Observed dry-run output:

```text
Model Params: 63.91M
Trainable Params: 63.912M
Epoch:[1/1](1/16), loss: 3.1348, logits_loss: 3.1348, aux_loss: 0.0000, lr: 0.00000991, epoch_time: 0.0min
...
Epoch:[1/1](16/16), loss: 3.6446, logits_loss: 3.6446, aux_loss: 0.0000, lr: 0.00000100, epoch_time: 0.0min
```

Debug outputs created:

```text
out/debug_stage1_sft_768.pth
checkpoints/debug_stage1_sft_768.pth
checkpoints/debug_stage1_sft_768_resume.pth
```

These are debug artifacts only and should not be committed.

## How to verify the model can chat

After the full SFT run produces `out/full_sft_768.pth`, use the official inference script:

```powershell
cd <repo-root>
$env:CUDA_VISIBLE_DEVICES = "0"
D:\APP\Anaconda3\envs\minimind\python.exe eval_llm.py `
  --save_dir out `
  --weight full_sft `
  --hidden_size 768 `
  --num_hidden_layers 8 `
  --max_new_tokens 256 `
  --device cuda `
  --show_speed 1
```

Choose automatic mode with input `0`, or manual mode with input `1` and ask simple prompts such as:

```text
Explain what machine learning is.
Introduce yourself in three sentences.
Calculate 23 + 58 and explain the steps.
```

Expected baseline behavior:

- The model loads `out/full_sft_768.pth` successfully.
- The model uses the chat template path rather than raw pretrain completion.
- Replies should look like assistant answers instead of continuing arbitrary text.
- Mathematical reliability is not the goal of Stage 1; this is only the official SFT baseline before math continued pretraining and math SFT.

## Notes and risks

- The stage config YAML is not consumed by MiniMind automatically; keep the docs command and YAML synchronized.
- `sft_t2t_mini.jsonl` is about 1.6 GB, so full SFT is much longer than the debug run.
- Native Windows plus `torchrun` may fail because distributed mode uses NCCL. Use single-process Python on the 5060 machine.
- `--dtype float16` is chosen for broad CUDA compatibility on the 5060. If bf16 is verified as stable on the local stack, `--dtype bfloat16` can also be tested.
- Keep `out/`, `checkpoints/`, and `dataset/*.jsonl` out of commits.
