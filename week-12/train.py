"""Train my randomly initialized GPT-2 (124M) on FineWeb-Edu token shards.

The optimization is the Week 11 setup: AdamW (0.9, 0.95, eps 1e-8, fused), weight decay 0.1 on
2D parameters only, gradient clipping at 1.0, linear warmup + cosine decay, gradient
accumulation, TF32 + BF16 autocast, SDPA attention, and the vocabulary padded to 50,304.

Training stops at --max_steps optimizer steps or after --time_budget_min minutes of training,
whichever comes first, and can be stopped early by creating the file STOP in the checkpoint
directory or with Ctrl+C. Every stop saves a verified checkpoint.

    python week-12/train.py --run_name main --max_steps 2000 --time_budget_min 50
    python week-12/train.py --run_name main --max_steps 2000 --time_budget_min 50 --resume latest
"""
import argparse
import json
import math
import os
import random
import shutil
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import tiktoken
import torch

sys.path.insert(0, str(Path(__file__).parent))
from gpt2_model import (GPT, GPTConfig, GPT2_VOCAB_SIZE, build_model, configure_optimizers, generate,
                        get_lr, set_tf32)

WEEK_DIR = Path(__file__).parent
PROMPTS = ['The history of science', 'In the future, technology', 'Once upon a time']

# arguments that define the optimization; a resumed run must use the same values
SCHEDULE_KEYS = ['micro_batch_size', 'seq_len', 'grad_accum_steps', 'max_lr', 'min_lr', 'warmup_steps',
                 'max_steps', 'weight_decay', 'grad_clip', 'seed']


# ----------------------------------------------------------------------------------------------
# data

class ShardedDataLoader:
    # reads one split as a single token stream that continues across its shard files

    def __init__(self, data_dir, split, B, T):
        self.data_dir, self.split, self.B, self.T = Path(data_dir), split, B, T
        self.files = sorted(self.data_dir.glob(f'*_{split}_*.npy'))
        assert self.files, f'no {split} shards in {self.data_dir}'
        assert all(f'_{split}_' in f.name for f in self.files)
        meta_path = self.data_dir / 'metadata.json'
        if meta_path.exists():
            # the shards on disk must be exactly the ones prepare_data.py wrote for this split
            listed = [s['file'] for s in json.loads(meta_path.read_text(encoding='utf-8'))['splits'][split]['shards']]
            assert [f.name for f in self.files] == listed, f'{split} shards differ from metadata.json'
        self.shards = [np.load(f, mmap_mode='r') for f in self.files] # memory-mapped, not read into RAM
        sizes = [len(s) for s in self.shards]
        self.starts = np.cumsum([0] + sizes) # global index of each shard's first token
        self.n_tokens = int(self.starts[-1])
        assert self.n_tokens >= B * T + 1, f'{split} split is shorter than one batch'
        self.reset()

    def reset(self):
        self.position = 0 # global index of the next input token
        self.epoch = 0

    @property
    def current_shard(self):
        return int(np.searchsorted(self.starts, self.position, side='right') - 1)

    def read(self, start, n):
        # n consecutive tokens of the split, gathered across shard boundaries
        out = np.empty(n, dtype=np.int64)
        filled = 0
        shard = int(np.searchsorted(self.starts, start, side='right') - 1)
        offset = start - int(self.starts[shard])
        while filled < n:
            take = min(n - filled, len(self.shards[shard]) - offset)
            out[filled:filled + take] = self.shards[shard][offset:offset + take]
            filled += take
            shard, offset = shard + 1, 0
        return out

    def next_batch(self):
        B, T = self.B, self.T
        buf = torch.from_numpy(self.read(self.position, B * T + 1))
        x = buf[:-1].view(B, T) # inputs
        y = buf[1:].view(B, T) # targets
        self.position += B * T
        # if the next batch would be out of bounds, wrap to the beginning
        if self.position + B * T + 1 > self.n_tokens:
            self.position = 0
            self.epoch += 1
        return x, y

    def state_dict(self):
        return dict(split=self.split, B=self.B, T=self.T, position=self.position, epoch=self.epoch,
                    files=[f.name for f in self.files], n_tokens=self.n_tokens)

    def load_state_dict(self, state):
        for key in ['split', 'B', 'T', 'files', 'n_tokens']:
            assert state[key] == self.state_dict()[key], f'loader {key} changed: {state[key]} != {self.state_dict()[key]}'
        self.position, self.epoch = state['position'], state['epoch']


# ----------------------------------------------------------------------------------------------
# evaluation and generation

@torch.no_grad()
def estimate_loss(model, data_dir, split, B, T, n_batches, device):
    # the mean loss over the first n_batches of a split: the same batches every time
    loader = ShardedDataLoader(data_dir, split, B, T)
    was_training = model.training
    model.eval()
    total = 0.0
    for _ in range(n_batches):
        x, y = loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
            _, loss = model(x, y)
        total += loss.item()
    model.train(was_training)
    return total / n_batches


def generate_samples(model, enc, device, max_new_tokens, temperature, top_k, seed):
    samples = []
    for i, prompt in enumerate(PROMPTS):
        ids = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)
        # one generator per prompt, seeded identically at every stage of training
        generator = torch.Generator(device=device).manual_seed(seed + i)
        out = generate(model, ids, max_new_tokens, temperature=temperature, top_k=top_k, generator=generator)
        assert out.max().item() < GPT2_VOCAB_SIZE
        samples.append(dict(prompt=prompt, text=enc.decode(out[0].tolist())))
    return samples


# ----------------------------------------------------------------------------------------------
# checkpoints

def write_json_atomic(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def rng_states():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def set_rng_states(states):
    random.setstate(states['python'])
    np.random.set_state(states['numpy'])
    torch.set_rng_state(states['torch'])
    if states['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states['cuda'])


def save_checkpoint(ckpt_dir, state, keep_last):
    step = state['step']
    path = ckpt_dir / f'ckpt_step_{step:06d}.pt'
    tmp = ckpt_dir / f'ckpt_step_{step:06d}.pt.tmp'
    free = shutil.disk_usage(ckpt_dir).free
    if free < 4 * 2**30:
        print(f'WARNING: only {free / 2**30:.1f} GiB free, checkpoint at step {step} not written')
        return None
    t0 = time.time()
    with open(tmp, 'wb') as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    # verify the written file before it gets its final name or becomes "latest"
    loaded = torch.load(tmp, map_location='cpu', weights_only=False)
    assert loaded['step'] == step
    assert loaded['model'].keys() == state['model'].keys()
    assert all(torch.isfinite(v).all() for v in loaded['model'].values())
    assert torch.equal(loaded['model']['lm_head.weight'], state['model']['lm_head.weight'].cpu())
    assert len(loaded['optimizer']['state']) == len(state['optimizer']['state'])
    del loaded
    os.replace(tmp, path)
    write_json_atomic(ckpt_dir / 'latest.json', dict(path=path.name, step=step, tokens_processed=state['tokens_processed']))
    print(f'saved and verified {path.name} ({path.stat().st_size / 2**30:.2f} GiB, {time.time() - t0:.1f} s)')
    if keep_last > 0:
        # older checkpoints are deleted only after the new one is verified and marked latest
        for old in sorted(ckpt_dir.glob('ckpt_step_*.pt'))[:-keep_last]:
            old.unlink()
    return path


def find_checkpoint(ckpt_dir, resume):
    if resume == 'latest':
        pointer = ckpt_dir / 'latest.json'
        assert pointer.exists(), f'no latest.json in {ckpt_dir}'
        return ckpt_dir / json.loads(pointer.read_text(encoding='utf-8'))['path']
    return Path(resume)


# ----------------------------------------------------------------------------------------------
# training

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run_name', default='main')
    p.add_argument('--data_dir', default=str(WEEK_DIR / 'data' / 'fineweb_edu'))
    p.add_argument('--ckpt_root', default=str(WEEK_DIR / 'checkpoints'))
    p.add_argument('--results_root', default=str(WEEK_DIR / 'results'))
    # batch: micro_batch_size * seq_len * grad_accum_steps tokens per optimizer step
    p.add_argument('--micro_batch_size', type=int, default=2)
    p.add_argument('--seq_len', type=int, default=256)
    p.add_argument('--grad_accum_steps', type=int, default=4)
    # optimization
    p.add_argument('--max_steps', type=int, required=True, help='optimizer steps; also the cosine horizon')
    p.add_argument('--time_budget_min', type=float, default=None, help='stop after this many minutes of training')
    p.add_argument('--stop_at_step', type=int, default=None, help='stop early without changing the lr schedule (pilot runs)')
    p.add_argument('--max_lr', type=float, default=6e-4)
    p.add_argument('--min_lr', type=float, default=None, help='default: 0.1 * max_lr')
    p.add_argument('--warmup_steps', type=int, default=100)
    p.add_argument('--weight_decay', type=float, default=0.1)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=1337)
    # evaluation, generation, checkpoints
    p.add_argument('--val_every', type=int, default=100)
    p.add_argument('--val_batches', type=int, default=20)
    p.add_argument('--val_batch_size', type=int, default=8)
    p.add_argument('--gen_every', type=int, default=500)
    p.add_argument('--gen_tokens', type=int, default=48)
    p.add_argument('--gen_temperature', type=float, default=1.0)
    p.add_argument('--gen_top_k', type=int, default=50)
    p.add_argument('--ckpt_every', type=int, default=500)
    p.add_argument('--keep_last', type=int, default=0, help='0 keeps every checkpoint')
    p.add_argument('--no_checkpoints', action='store_true', help='pilot runs: write no checkpoint files')
    p.add_argument('--log_every', type=int, default=10)
    p.add_argument('--resume', default=None, help="'latest' or a checkpoint path")
    args = p.parse_args(argv)
    if args.min_lr is None:
        args.min_lr = 0.1 * args.max_lr
    return args


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert device == 'cuda', 'training this model needs the GPU'
    ckpt_dir = Path(args.ckpt_root) / args.run_name
    results_dir = Path(args.results_root) / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    stop_file = ckpt_dir / 'STOP'
    if stop_file.exists():
        stop_file.unlink()

    set_tf32(True)
    enc = tiktoken.get_encoding('gpt2')
    B, T, accum = args.micro_batch_size, args.seq_len, args.grad_accum_steps
    tokens_per_step = B * T * accum
    train_loader = ShardedDataLoader(args.data_dir, 'train', B, T)
    val_loader_check = ShardedDataLoader(args.data_dir, 'val', args.val_batch_size, T)
    assert set(train_loader.files).isdisjoint(val_loader_check.files) # no validation shard in the train loader
    assert args.val_batches * args.val_batch_size * T + 1 <= val_loader_check.n_tokens, 'val batches exceed the val split'
    model_config = GPTConfig(block_size=1024, vocab_size=50304, n_layer=12, n_head=12, n_embd=768, sdpa=True)
    assert T <= model_config.block_size

    if args.resume:
        path = find_checkpoint(ckpt_dir, args.resume)
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        for key in SCHEDULE_KEYS:
            assert ckpt['args'][key] == vars(args)[key], f'{key} differs from the checkpoint: {ckpt["args"][key]} != {vars(args)[key]}'
        assert ckpt['model_config'] == asdict(model_config)
        model = GPT(model_config).to(device)
        model.load_state_dict(ckpt['model'])
        optimizer = configure_optimizers(model, args.max_lr, args.weight_decay, fused=True)
        optimizer.load_state_dict(ckpt['optimizer'])
        train_loader.load_state_dict(ckpt['train_loader'])
        set_rng_states(ckpt['rng'])
        history, step, tokens_processed = ckpt['history'], ckpt['step'], ckpt['tokens_processed']
        train_seconds = ckpt['train_seconds']
        history['resumes'].append(dict(step=step, from_checkpoint=path.name, time=time.strftime('%Y-%m-%d %H:%M:%S')))
        print(f'resumed from {path.name}: step {step}, {tokens_processed:,} tokens, {train_seconds / 60:.1f} min of training')
        del ckpt
    else:
        model = build_model(model_config, args.seed, device) # random initialization, no pretrained weights
        optimizer = configure_optimizers(model, args.max_lr, args.weight_decay, fused=True)
        history = dict(train=[], eval=[], samples=[], resumes=[])
        step, tokens_processed, train_seconds = 0, 0, 0.0

    n_params = sum(p.numel() for p in model.parameters())
    print(f'model: {n_params:,} parameters | vocab {model_config.vocab_size} | block_size {model_config.block_size}')
    print(f'batch: {B} x {T} x {accum} accumulation = {tokens_per_step:,} tokens per optimizer step')
    print(f'train split: {train_loader.n_tokens:,} tokens in {len(train_loader.files)} shards | '
          f'planned: {args.max_steps * tokens_per_step:,} tokens = {args.max_steps * tokens_per_step / train_loader.n_tokens:.2f} epochs')
    print(f'lr: warmup {args.warmup_steps}, cosine to step {args.max_steps}, max {args.max_lr:.2e}, min {args.min_lr:.2e}')

    interrupted = {'flag': False}
    def on_sigint(signum, frame):
        if interrupted['flag']:
            raise KeyboardInterrupt
        print('\ninterrupt received: finishing this step and saving a checkpoint (press again to abort)')
        interrupted['flag'] = True
    signal.signal(signal.SIGINT, on_sigint)

    def checkpoint_state():
        return dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                    tokens_processed=tokens_processed, train_seconds=train_seconds, history=history,
                    train_loader=train_loader.state_dict(), rng=rng_states(), args=vars(args),
                    model_config=asdict(model_config))

    def run_eval():
        t0 = time.time()
        val_loss = estimate_loss(model, args.data_dir, 'val', args.val_batch_size, T, args.val_batches, device)
        train_eval_loss = estimate_loss(model, args.data_dir, 'train', args.val_batch_size, T, args.val_batches, device)
        recent = [r['loss'] for r in history['train'] if r['step'] >= step - args.val_every]
        record = dict(step=step, tokens_processed=tokens_processed, val_loss=val_loss, train_eval_loss=train_eval_loss,
                      train_loss_recent_mean=sum(recent) / len(recent) if recent else None,
                      peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                      peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20, seconds=time.time() - t0)
        history['eval'] = [r for r in history['eval'] if r['step'] != step] + [record]
        print(f'eval step {step:5d} | val {val_loss:.4f} | train (same batches procedure) {train_eval_loss:.4f} | {record["seconds"]:.1f} s')

    def run_generation():
        samples = generate_samples(model, enc, device, args.gen_tokens, args.gen_temperature, args.gen_top_k, args.seed)
        history['samples'] = [r for r in history['samples'] if r['step'] != step] + [dict(step=step, samples=samples)]
        for s in samples:
            print(f'  sample @ {step}: {s["text"]!r}')

    def save_results(stop_reason=None):
        write_json_atomic(results_dir / 'history.json', history)
        if stop_reason:
            train_steps = history['train']
            summary = dict(run_name=args.run_name, stop_reason=stop_reason, completed_steps=step, max_steps=args.max_steps,
                           tokens_per_step=tokens_per_step, tokens_processed=tokens_processed,
                           train_split_tokens=train_loader.n_tokens, epochs=tokens_processed / train_loader.n_tokens,
                           train_seconds=train_seconds, tokens_per_sec_train=tokens_processed / max(train_seconds, 1e-9),
                           final_train_loss=train_steps[-1]['loss'] if train_steps else None,
                           final_eval=history['eval'][-1] if history['eval'] else None, parameters=n_params,
                           peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                           peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                           args=vars(args), model_config=asdict(model_config), resumes=history['resumes'],
                           gpu=torch.cuda.get_device_name(0), torch=torch.__version__)
            write_json_atomic(results_dir / 'summary.json', summary)

    model.train()
    if step == 0 and not history['eval']:
        run_eval()
        run_generation() # the randomly initialized model, before any update
        save_results()

    torch.cuda.reset_peak_memory_stats()
    stop_reason = 'max_steps'
    while step < args.max_steps:
        if args.time_budget_min is not None and train_seconds >= args.time_budget_min * 60:
            stop_reason = 'time_budget'
            break
        if interrupted['flag'] or stop_file.exists():
            stop_reason = 'interrupted'
            break
        if args.stop_at_step is not None and step >= args.stop_at_step:
            stop_reason = 'stop_at_step'
            break

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for micro_step in range(accum):
            x, y = train_loader.next_batch()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / accum # the mean over the k microbatches, not their sum
            loss_accum += loss.detach()
            loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip) # the norm before clipping
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f'non-finite gradient norm at step {step}; the last checkpoint is unaffected')
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.max_lr, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        loss_value = float(loss_accum)
        if not math.isfinite(loss_value):
            raise FloatingPointError(f'non-finite loss at step {step}; the last checkpoint is unaffected')

        train_seconds += dt
        tokens_processed += tokens_per_step
        history['train'].append(dict(step=step, loss=loss_value, lr=lr, grad_norm=grad_norm.item(), step_ms=dt * 1000,
                                     tokens_per_sec=tokens_per_step / dt, tokens_processed=tokens_processed,
                                     shard=train_loader.current_shard, epoch=train_loader.epoch))
        if step % args.log_every == 0:
            print(f'step {step:5d} | loss {loss_value:.4f} | lr {lr:.2e} | norm {grad_norm.item():6.3f} | '
                  f'{dt * 1000:6.1f} ms | {tokens_per_step / dt:7,.0f} tok/s | shard {train_loader.current_shard} | '
                  f'{train_seconds / 60:5.1f} min')
        step += 1

        if step % args.val_every == 0 or step == args.max_steps:
            run_eval()
        if step % args.gen_every == 0 or step == args.max_steps:
            run_generation()
        if step % args.ckpt_every == 0 and step < args.max_steps and not args.no_checkpoints:
            save_checkpoint(ckpt_dir, checkpoint_state(), args.keep_last)
            save_results()

    # the final state always gets an evaluation, samples, and a checkpoint
    if not history['eval'] or history['eval'][-1]['step'] != step:
        run_eval()
    if not history['samples'] or history['samples'][-1]['step'] != step:
        run_generation()
    if not args.no_checkpoints:
        save_checkpoint(ckpt_dir, checkpoint_state(), args.keep_last)
    save_results(stop_reason)
    print(f'stopped ({stop_reason}) at step {step}: {tokens_processed:,} tokens in {train_seconds / 60:.1f} min of training '
          f'({tokens_processed / max(train_seconds, 1e-9):,.0f} tokens/s) | peak allocated {torch.cuda.max_memory_allocated() / 2**20:,.0f} MiB')
    return dict(step=step, stop_reason=stop_reason, history=history, model=model)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True) # log lines appear immediately, also when redirected to a file
    train(parse_args())
