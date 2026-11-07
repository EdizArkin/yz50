# Week 12 — Training GPT-2 (124M) from Scratch and Evaluating It

The final week of YZ50: my Week 11 GPT-2 (124M), **randomly initialized**, trained on 39.3M tokens of FineWeb-Edu on a laptop GPU, with validation loss, text generation and checkpoints during training, and a HellaSwag comparison with OpenAI's pretrained GPT-2 (124M) using one shared evaluation implementation.

The main submission is **[`training_gpt2.ipynb`](training_gpt2.ipynb)**. The Python files are the supporting implementation used for the long-running steps.

## Results

| Metric | My trained GPT-2 | OpenAI GPT-2 |
|---|---:|---:|
| Parameters | 124,475,904 (vocab padded to 50,304) | 124,439,808 |
| Training data | FineWeb-Edu `sample-10BT`, 40M-token subset | WebText (pretrained reference) |
| Tokens processed | 39,321,600 (2,400 steps × 16,384; 0.98 epochs) | — |
| Training time / throughput | 51.0 min / 12,840 tokens/s (RTX 3060 Laptop, 6 GB) | — |
| Validation loss on my split | 5.186 | 3.636 |
| HellaSwag examples | 10,042 (full validation split) | 10,042 |
| HellaSwag acc (summed loss) | 26.05% | 28.58% |
| HellaSwag acc_norm (mean loss per token) | 23.98% | 29.55% |

Random chance is 25%. The random initialization of my model scores 25.46% / 24.86% with the same code, and my trained model is **not significantly different from it** (paired McNemar test). The notebook explains why: the summed-loss metric favors short endings, and 39M tokens taught the model English words, local grammar and the corpus's style, but not the event-level knowledge that separates HellaSwag's endings.

## Files

| File | Purpose |
|---|---|
| `training_gpt2.ipynb` | the educational notebook: method, checks, results, interpretation |
| `gpt2_model.py` | GPT-2 module, initialization, AdamW groups, LR schedule, sampling (Week 10/11 code) |
| `prepare_data.py` | streams FineWeb-Edu, splits documents train/val, writes `uint16` token shards and `metadata.json` |
| `train.py` | sharded data loader, training loop, validation, generation, verified atomic checkpoints, resume |
| `evaluate_hellaswag.py` | HellaSwag scoring for either model, and the paired comparison |
| `pilot.py` | batch-shape benchmark and learning-rate pilot |
| `sanity_checks.py` | tests of the loader, model, checkpoint/resume and HellaSwag scoring |
| `results/` | saved histories, summaries, per-example HellaSwag records, plots, pilot results |

`data/` (token shards, HellaSwag download) and `checkpoints/` (6 × 1.39 GiB) are not in the repository; see `.gitignore`.

## Reproducing

Requirements: the repository's `requirements.txt` (PyTorch with CUDA, `tiktoken`, `transformers`, `datasets==4.8.5`, `huggingface-hub==0.36.2`). All commands are run from the repository root.

```bash
# 1. data: 40M train + 1M val tokens, ~13 min, ~80 MB on disk
python week-12/prepare_data.py --train_tokens 40_000_000 --val_tokens 1_000_000 --val_fraction 0.025 --shard_size 5_000_000

# 2. checks (~4 min on the GPU)
python week-12/sanity_checks.py

# 3. optional pilot (~15 min)
python week-12/pilot.py bench
python week-12/pilot.py lr --micro_batch_size 8 --seq_len 256 --grad_accum_steps 8 --max_steps 2400 --stop_at_step 250

# 4. training (~56 min; add --resume latest to continue after an interruption)
python -u week-12/train.py --run_name main --micro_batch_size 8 --seq_len 256 --grad_accum_steps 8 \
  --max_steps 2400 --warmup_steps 100 --max_lr 6e-4 --min_lr 6e-5 \
  --val_every 100 --val_batches 100 --val_batch_size 8 --gen_every 400 --ckpt_every 400 --time_budget_min 58

# 5. HellaSwag (~5 min per model)
python week-12/evaluate_hellaswag.py --model mine --checkpoint week-12/checkpoints/main/ckpt_step_002400.pt --n_examples 10042
python week-12/evaluate_hellaswag.py --model gpt2 --n_examples 10042
python week-12/evaluate_hellaswag.py --compare week-12/results/hellaswag_mine.json week-12/results/hellaswag_gpt2.json

# 6. the notebook (reads the results; computes four small cached analyses once, ~12 min on the GPU)
jupyter nbconvert --to notebook --execute --inplace week-12/training_gpt2.ipynb
```

What each step produces:

| Step | Output |
|---|---|
| 1. Data | `data/fineweb_edu/`: 8 train shards + 1 val shard (`uint16 .npy`), `metadata.json` (source revision, counts, crawl distribution), `split_documents.json`. The output directory must be empty; the script refuses to overwrite existing shards. |
| 4. Training | `checkpoints/main/ckpt_step_*.pt` and `latest.json`; `results/main/history.json` (every step, every evaluation, all samples), `summary.json`, `train_log.txt` |
| 5. HellaSwag | the validation split is downloaded automatically on first use to `data/hellaswag/` (pinned revision, recorded in `source.json`); `results/hellaswag_mine.json`, `results/hellaswag_gpt2.json` (per-example losses and predictions), `results/hellaswag_comparison.json` |
| 6. Notebook | `results/val_loss_comparison.json`, `generation_metrics.json`, `hellaswag_random_init.json`, `hellaswag_trajectory/`, `main/loss_curves.png`; once these exist, the notebook only reads them |

To stop training early and keep a checkpoint, create `week-12/checkpoints/main/STOP` or press Ctrl+C once.

## Notes

- **FineWeb-Edu** is read through `datasets` streaming at a pinned revision. `sample-10BT` is stored in 14 files ordered by crawl date, so `prepare_data.py` reads them round-robin instead of taking the start of the stream.
- **HellaSwag** is read from the dataset author's Hugging Face repository, [`Rowan/hellaswag`](https://huggingface.co/datasets/Rowan/hellaswag) (revision `218ec52e09a7`), because the GitHub repository `rowanz/hellaswag` used by build-nanogpt has been blocked by a DMCA notice since 2026-09-14. `acc_norm` here is normalized by the number of tokens, as in build-nanogpt, not by characters as in EleutherAI's evaluation harness.
- **Environment:** Windows, PyTorch 2.14 + CUDA 12.6, BF16 autocast and the memory-efficient SDPA kernel. Flash Attention and `torch.compile` were not available in this environment (Week 11).
