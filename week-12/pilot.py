"""Phase C pilot: batch-shape throughput benchmark and a short learning-rate comparison.

Every run is a separate train.py process on the prepared FineWeb-Edu shards, without checkpoint files.

    python week-12/pilot.py bench
    python week-12/pilot.py lr --micro_batch_size 8 --seq_len 256 --grad_accum_steps 8 --max_steps 2400 --stop_at_step 250
"""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

WEEK_DIR = Path(__file__).parent
RESULTS = WEEK_DIR / 'results' / 'pilot'
TOKENS_PER_STEP = 16_384
BENCH_CONFIGS = [(2, 256), (8, 256), (16, 256), (4, 512), (8, 512), (2, 1024), (4, 1024)] # (micro batch, sequence length)


def run_train(run_name, extra):
    cmd = [sys.executable, str(WEEK_DIR / 'train.py'), '--run_name', run_name, '--results_root', str(RESULTS),
           '--ckpt_root', str(WEEK_DIR / 'checkpoints' / '_pilot'), '--no_checkpoints'] + extra
    out = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    log = out.stdout + out.stderr
    (RESULTS / f'{run_name}.log').write_text(log, encoding='utf-8')
    return out.returncode, log


def bench(args):
    rows = []
    for B, T in BENCH_CONFIGS:
        accum = TOKENS_PER_STEP // (B * T)
        name = f'bench_{B}x{T}x{accum}'
        code, log = run_train(name, ['--micro_batch_size', str(B), '--seq_len', str(T), '--grad_accum_steps', str(accum),
                                     '--max_steps', '1000', '--stop_at_step', str(args.steps), '--warmup_steps', '100',
                                     '--val_every', '1000000', '--gen_every', '1000000', '--val_batches', '1',
                                     '--val_batch_size', str(B), '--log_every', '1'])
        if code != 0:
            reason = 'out of memory' if 'OutOfMemoryError' in log else log.strip().splitlines()[-1][:200]
            rows.append(dict(config=name, B=B, T=T, accum=accum, failed=reason))
            print(f'{name:22s} failed: {reason}')
            continue
        history = json.loads((RESULTS / name / 'history.json').read_text(encoding='utf-8'))
        summary = json.loads((RESULTS / name / 'summary.json').read_text(encoding='utf-8'))
        step_ms = [r['step_ms'] for r in history['train'][args.warmup:]] # the first steps include allocation and kernel setup
        median_ms = statistics.median(step_ms)
        rows.append(dict(config=name, B=B, T=T, accum=accum, tokens_per_step=B * T * accum,
                         median_step_ms=median_ms, min_step_ms=min(step_ms), max_step_ms=max(step_ms),
                         tokens_per_sec=B * T * accum / (median_ms / 1000), measured_steps=len(step_ms),
                         peak_allocated_mib=summary['peak_allocated_mib'], peak_reserved_mib=summary['peak_reserved_mib'],
                         losses=[r['loss'] for r in history['train']]))
        r = rows[-1]
        print(f"{name:22s} | {median_ms:7.1f} ms/step ({r['min_step_ms']:.0f}-{r['max_step_ms']:.0f}) | {r['tokens_per_sec']:7,.0f} tok/s | "
              f"peak allocated {r['peak_allocated_mib']:6,.0f} MiB | reserved {r['peak_reserved_mib']:6,.0f} MiB")
    (RESULTS / 'bench.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')


def lr(args):
    for max_lr in args.lrs:
        name = f'lr_{max_lr:.0e}_{args.micro_batch_size}x{args.seq_len}x{args.grad_accum_steps}'
        code, log = run_train(name, ['--micro_batch_size', str(args.micro_batch_size), '--seq_len', str(args.seq_len),
                                     '--grad_accum_steps', str(args.grad_accum_steps), '--max_lr', str(max_lr),
                                     '--max_steps', str(args.max_steps), '--stop_at_step', str(args.stop_at_step),
                                     '--warmup_steps', str(args.warmup_steps), '--val_every', str(args.val_every),
                                     '--val_batches', str(args.val_batches), '--val_batch_size', '8',
                                     '--gen_every', str(args.stop_at_step), '--log_every', '25'])
        assert code == 0, log[-3000:]
        history = json.loads((RESULTS / name / 'history.json').read_text(encoding='utf-8'))
        print(name, '| val:', ', '.join(f"{r['step']}: {r['val_loss']:.3f}" for r in history['eval']))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('bench')
    b.add_argument('--steps', type=int, default=12)
    b.add_argument('--warmup', type=int, default=2)
    l = sub.add_parser('lr')
    l.add_argument('--lrs', type=float, nargs='+', default=[3e-4, 6e-4])
    l.add_argument('--micro_batch_size', type=int, required=True)
    l.add_argument('--seq_len', type=int, required=True)
    l.add_argument('--grad_accum_steps', type=int, required=True)
    l.add_argument('--max_steps', type=int, required=True, help='the cosine horizon of the planned final run')
    l.add_argument('--stop_at_step', type=int, default=250)
    l.add_argument('--warmup_steps', type=int, default=100)
    l.add_argument('--val_every', type=int, default=25)
    l.add_argument('--val_batches', type=int, default=40)
    args = p.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    {'bench': bench, 'lr': lr}[args.command](args)
