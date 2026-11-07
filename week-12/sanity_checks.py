"""Phase A checks: data loader, model, training step, checkpoint/resume, generation, HellaSwag scoring.

    python week-12/sanity_checks.py --data_dir week-12/data/fineweb_edu            # everything
    python week-12/sanity_checks.py --data_dir week-12/data/fineweb_edu --only loader hellaswag
"""
import argparse
import gc
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import tiktoken
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from gpt2_model import (GPT, GPTConfig, GPT2_VOCAB_SIZE, PADDED_VOCAB_SIZE, build_model, from_pretrained,
                        generate, set_tf32)
from train import ShardedDataLoader, parse_args, train
import evaluate_hellaswag as hs

WEEK_DIR = Path(__file__).parent
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def check_loader(data_dir):
    print('\n== sharded data loader ==')
    # synthetic shards with known content: token value = its position in the stream
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        sizes = [100, 37, 64] # the middle shard is shorter than some batches
        stream = np.arange(sum(sizes), dtype=np.uint16)
        start = 0
        for k, n in enumerate(sizes):
            np.save(tmp / f'synthetic_train_{k:06d}.npy', stream[start:start + n])
            start += n
        np.save(tmp / 'synthetic_val_000000.npy', np.arange(1000, 1100, dtype=np.uint16))
        B, T = 2, 8
        loader = ShardedDataLoader(tmp, 'train', B, T)
        assert loader.n_tokens == 201 and all('_train_' in f.name for f in loader.files)
        n_batches = (201 - 1) // (B * T) # 12 full batches; the tail of 8 tokens cannot fill one
        inputs, crossed = [], 0
        for b in range(n_batches):
            assert loader.epoch == 0
            first_shard = loader.current_shard
            pos = loader.position
            x, y = loader.next_batch()
            assert x.shape == y.shape == (B, T)
            # y is x shifted by exactly one token, also across rows and shard files
            assert torch.equal(x.flatten()[1:], y.flatten()[:-1])
            assert torch.equal(x.flatten(), torch.from_numpy(stream[pos:pos + B * T].astype(np.int64)))
            assert y.flatten()[-1].item() == stream[pos + B * T]
            last_shard = int(np.searchsorted(loader.starts, pos + B * T, side='right') - 1)
            crossed += last_shard != first_shard
            inputs.append(x.flatten())
        assert torch.equal(torch.cat(inputs), torch.from_numpy(stream[:n_batches * B * T].astype(np.int64))) # in order, nothing skipped
        assert loader.epoch == 1 and loader.position == 0 # the final batch of the epoch wrapped the loader
        x_wrapped, _ = loader.next_batch()
        assert torch.equal(x_wrapped, inputs[0].view(B, T)) # the second epoch starts from the first batch
        print(f'synthetic shards {sizes}: {n_batches} batches per epoch, {crossed} cross a shard boundary, '
              f'{201 - 1 - n_batches * B * T} tail tokens dropped at the wrap')
        assert crossed >= 2

        # reset and state_dict give the same batches again
        loader.reset()
        assert torch.equal(loader.next_batch()[0], inputs[0].view(B, T))
        for _ in range(5):
            loader.next_batch()
        state = loader.state_dict()
        expected = loader.next_batch()
        restored = ShardedDataLoader(tmp, 'train', B, T)
        restored.load_state_dict(json.loads(json.dumps(state))) # survives serialization
        assert all(torch.equal(a, b) for a, b in zip(restored.next_batch(), expected))
        try:
            ShardedDataLoader(tmp, 'train', 4, 8).load_state_dict(state)
            raise RuntimeError('a state with a different batch shape was accepted')
        except AssertionError:
            pass
        val = ShardedDataLoader(tmp, 'val', B, T)
        assert set(val.files).isdisjoint(loader.files) and val.next_batch()[0].min().item() >= 1000
        del loader, restored, val # Windows keeps memory-mapped files locked until they are released
        gc.collect()

    # the real prepared shards
    meta = json.loads((Path(data_dir) / 'metadata.json').read_text(encoding='utf-8'))
    train_loader = ShardedDataLoader(data_dir, 'train', 2, 256)
    val_loader = ShardedDataLoader(data_dir, 'val', 2, 256)
    assert set(train_loader.files).isdisjoint(val_loader.files)
    assert train_loader.n_tokens == meta['splits']['train']['tokens']
    assert val_loader.n_tokens == meta['splits']['val']['tokens']
    docs = json.loads((Path(data_dir) / 'split_documents.json').read_text(encoding='utf-8'))
    assert not set(docs['train']) & set(docs['val'])
    # near-duplicates: different documents can still open with the same text (site boilerplate)
    eot = 50256
    def doc_starts(loader, n=32):
        tokens = loader.read(0, loader.n_tokens)
        starts = np.flatnonzero(tokens == eot)
        return {tokens[s + 1:s + 1 + n].tobytes() for s in starts if s + 1 + n <= len(tokens)}
    val_starts = doc_starts(val_loader)
    shared_starts = doc_starts(train_loader) & val_starts
    assert len(shared_starts) <= 0.01 * len(val_starts), f'{len(shared_starts)} val documents share an opening with train'
    # every batch of one epoch keeps its shape and its shift, including across real shard files
    n_batches = (train_loader.n_tokens - 1) // (2 * 256)
    shards_seen = set()
    for _ in range(n_batches):
        shards_seen.add(train_loader.current_shard)
        x, y = train_loader.next_batch()
        assert x.shape == (2, 256) and torch.equal(x.flatten()[1:], y.flatten()[:-1])
    assert train_loader.epoch == 1 and shards_seen == set(range(len(train_loader.files)))
    print(f'real shards: train {train_loader.n_tokens:,} tokens in {len(train_loader.files)} files, '
          f'val {val_loader.n_tokens:,} in {len(val_loader.files)}; '
          f'{len(docs["train"])} + {len(docs["val"])} documents, 0 shared ids, '
          f'{len(shared_starts)} of {len(val_starts)} val documents open with the same 32 tokens as a train document')
    print('Loader checks passed.')


def check_model():
    print('\n== model ==')
    model = GPT(GPTConfig())
    n_padded = sum(p.numel() for p in model.parameters())
    n_unpadded = sum(p.numel() for p in GPT(GPTConfig(vocab_size=GPT2_VOCAB_SIZE)).parameters())
    assert n_padded == 124_475_904 and n_unpadded == 124_439_808
    assert model.lm_head.weight is model.transformer.wte.weight
    assert (model.config.n_layer, model.config.n_head, model.config.n_embd) == (12, 12, 768)
    print(f'parameters: {n_padded:,} with the padded vocabulary ({n_unpadded:,} at 50,257)')
    del model

    # SDPA and the explicit formula compute the same function
    small = dict(block_size=64, vocab_size=200, n_layer=2, n_head=4, n_embd=64)
    manual, fast = build_model(GPTConfig(**small, sdpa=False), 1337, device), build_model(GPTConfig(**small, sdpa=True), 1337, device)
    idx = torch.randint(200, (4, 64), device=device)
    with torch.no_grad():
        assert torch.allclose(manual(idx)[0], fast(idx)[0], atol=1e-4)
    del manual, fast

    # pretrained weights in my module reproduce Hugging Face's logits (float32, no TF32)
    from transformers import GPT2LMHeadModel
    set_tf32(False)
    mine = from_pretrained('gpt2').to(device).eval()
    ref = GPT2LMHeadModel.from_pretrained('openai-community/gpt2').to(device).eval()
    enc = tiktoken.get_encoding('gpt2')
    x = torch.tensor([enc.encode('The history of science is the study of the development of science and scientific knowledge')],
                     device=device)
    with torch.no_grad():
        diff = (mine(x)[0] - ref(x).logits).abs().max().item()
    print(f'my GPT (SDPA) with OpenAI weights vs Hugging Face GPT2LMHeadModel: max |logit diff| {diff:.2e}')
    assert diff < 1e-3
    del mine, ref

    # the padded rows: a random model can rank them in its top 50, so generation must exclude them
    model = build_model(GPTConfig(), 1337, device)
    ids = torch.tensor([enc.encode('Once upon a time')], device=device)
    with torch.no_grad():
        top = model(ids)[0][0, -1].topk(50).indices
    n_pad_in_top = (top >= GPT2_VOCAB_SIZE).sum().item()
    out = generate(model, ids.repeat(8, 1), 64, top_k=50, generator=torch.Generator(device=device).manual_seed(0))
    assert out.max().item() < GPT2_VOCAB_SIZE
    a = generate(model, ids, 20, generator=torch.Generator(device=device).manual_seed(5))
    b = generate(model, ids, 20, generator=torch.Generator(device=device).manual_seed(5))
    c = generate(model, ids, 20, generator=torch.Generator(device=device).manual_seed(6))
    assert torch.equal(a, b) and not torch.equal(a, c)
    print(f'random model: {n_pad_in_top} padded ids in the raw top 50; 8 x 64 generated tokens all < 50,257; same seed, same text')
    print('sample from the random model:', repr(enc.decode(a[0].tolist())))
    del model
    torch.cuda.empty_cache()
    print('Model checks passed.')


def check_training_and_resume(data_dir):
    print('\n== training step, checkpoint, resume ==')
    ckpt_root = WEEK_DIR / 'checkpoints' / '_sanity'
    results_root = WEEK_DIR / 'results' / '_sanity'
    for d in (ckpt_root, results_root):
        shutil.rmtree(d, ignore_errors=True)
    common = ['--data_dir', str(data_dir), '--ckpt_root', str(ckpt_root), '--results_root', str(results_root),
              '--micro_batch_size', '2', '--seq_len', '256', '--grad_accum_steps', '4', '--max_steps', '12',
              '--warmup_steps', '4', '--val_every', '6', '--gen_every', '6', '--ckpt_every', '6', '--val_batches', '2',
              '--val_batch_size', '2', '--gen_tokens', '16', '--log_every', '1']
    torch.cuda.reset_peak_memory_stats()
    full = train(parse_args(common + ['--run_name', 'continuous']))
    hist_a = full['history']
    del full
    torch.cuda.empty_cache()

    # resume a second run from the step-6 checkpoint written by the first one
    (ckpt_root / 'resumed').mkdir(parents=True)
    shutil.copy(ckpt_root / 'continuous' / 'ckpt_step_000006.pt', ckpt_root / 'resumed' / 'ckpt_step_000006.pt')
    (ckpt_root / 'resumed' / 'latest.json').write_text(json.dumps(dict(path='ckpt_step_000006.pt', step=6, tokens_processed=6 * 2048)))
    resumed = train(parse_args(common + ['--run_name', 'resumed', '--resume', 'latest']))
    hist_b = resumed['history']
    del resumed
    torch.cuda.empty_cache()

    a = torch.load(ckpt_root / 'continuous' / 'ckpt_step_000012.pt', map_location='cpu', weights_only=False)
    b = torch.load(ckpt_root / 'resumed' / 'ckpt_step_000012.pt', map_location='cpu', weights_only=False)
    losses_a = [r['loss'] for r in hist_a['train']]
    losses_b = [r['loss'] for r in hist_b['train']]
    assert len(losses_a) == len(losses_b) == 12 and losses_a[:6] == losses_b[:6] # history restored
    loss_diff = max(abs(x - y) for x, y in zip(losses_a[6:], losses_b[6:]))
    param_diff = max((a['model'][k] - b['model'][k]).abs().max().item() for k in a['model'])
    assert a['train_loader'] == b['train_loader'] and a['tokens_processed'] == b['tokens_processed'] == 12 * 2048
    assert [r['lr'] for r in hist_a['train']] == [r['lr'] for r in hist_b['train']]
    assert hist_b['resumes'][0]['step'] == 6 and a['step'] == b['step'] == 12
    print(f'steps 6-11 after resume: max |loss diff| {loss_diff:.2e} | final max |param diff| {param_diff:.2e} | '
          f'loader position {a["train_loader"]["position"]} in both | samples at steps {[s["step"] for s in hist_b["samples"]]}')
    assert loss_diff < 1e-2
    print('first losses:', [round(v, 4) for v in losses_a[:4]], '... last:', round(losses_a[-1], 4))
    print('peak allocated during the two runs:', f'{torch.cuda.max_memory_allocated() / 2**20:,.0f} MiB')
    del a, b

    # graceful stop with the STOP file in a separate process, then resume
    cmd = [sys.executable, str(WEEK_DIR / 'train.py')] + [v if v != '12' else '1000' for v in common] + ['--run_name', 'stopfile', '--ckpt_every', '500', '--val_every', '500', '--gen_every', '500']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
    log = []
    for line in proc.stdout:
        log.append(line)
        if line.startswith('step     3'):
            (ckpt_root / 'stopfile' / 'STOP').touch()
    proc.wait()
    assert proc.returncode == 0, ''.join(log[-20:])
    summary = json.loads((results_root / 'stopfile' / 'summary.json').read_text(encoding='utf-8'))
    pointer = json.loads((ckpt_root / 'stopfile' / 'latest.json').read_text(encoding='utf-8'))
    assert summary['stop_reason'] == 'interrupted' and pointer['step'] == summary['completed_steps'] >= 4
    assert summary['completed_steps'] <= 8, 'the STOP file was noticed late'
    resume_cmd = cmd + ['--resume', 'latest', '--time_budget_min', str((summary['train_seconds'] + 3) / 60)]
    out = subprocess.run(resume_cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    summary2 = json.loads((results_root / 'stopfile' / 'summary.json').read_text(encoding='utf-8'))
    assert summary2['stop_reason'] == 'time_budget' and summary2['completed_steps'] > summary['completed_steps']
    assert summary2['resumes'][0]['step'] == summary['completed_steps']
    print(f'STOP file: stopped at step {summary["completed_steps"]} with a verified checkpoint; resumed and '
          f'stopped by the time budget at step {summary2["completed_steps"]}')
    assert not list(ckpt_root.rglob('*.tmp')) # no incomplete files left behind
    shutil.rmtree(ckpt_root)
    shutil.rmtree(results_root)
    print('Training and resume checks passed.')


def check_hellaswag(n_small=20):
    print('\n== HellaSwag scoring ==')
    set_tf32(False)
    enc = tiktoken.get_encoding('gpt2')
    examples, source = hs.load_hellaswag_val()
    print(f'{len(examples)} validation examples from {source["repo"]} @ {source["revision"][:10]}')
    lengths = [hs.render_example(e, enc, 1024)[0].size(1) for e in examples]
    print(f'row length (context + longest ending): median {int(np.median(lengths))}, 99th pct {int(np.percentile(lengths, 99))}, '
          f'max {max(lengths)} tokens; > 256: {sum(l > 256 for l in lengths)}, > 512: {sum(l > 512 for l in lengths)}')

    model = from_pretrained('gpt2').to(device).eval()
    e = examples[0]
    tokens, mask, _ = hs.render_example(e, enc, 1024)
    unlabeled = {k: v for k, v in e.items() if k != 'label'} # the renderer works without the label
    assert all(torch.equal(a, b) for a, b in zip(hs.render_example(unlabeled, enc, 1024)[:2], (tokens, mask)))
    sum_loss, avg_loss = hs.score_example(model, tokens, mask, device, autocast=False)

    # 1) the first ending token is scored by the logits at the last context position
    n_ctx = len(enc.encode(e['ctx']))
    with torch.no_grad():
        logits = model(tokens.to(device))[0][..., :GPT2_VOCAB_SIZE].float()
    logp = torch.log_softmax(logits, dim=-1)
    for i in range(4):
        n_end = int(mask[i].sum())
        row = tokens[i].to(device)
        manual = -sum(logp[i, t - 1, row[t]].item() for t in range(n_ctx, n_ctx + n_end))
        assert abs(manual - sum_loss[i].item()) < 1e-3, (manual, sum_loss[i].item())
        assert mask[i, n_ctx - 1] == 0 and mask[i, n_ctx] == 1
    # 2) padding does not change any score: each ending alone, without padding
    for i in range(4):
        n = int(mask[i].nonzero().max()) + 1
        s, a = hs.completion_losses(model(tokens[i:i + 1, :n].to(device))[0], tokens[i:i + 1, :n].to(device), mask[i:i + 1, :n].to(device))
        assert abs(s.item() - sum_loss[i].item()) < 1e-3 and abs(a.item() - avg_loss[i].item()) < 1e-4
    # 3) reordering the endings reorders the losses: the score depends on the ending, not on its slot
    permuted = dict(e, endings=[e['endings'][j] for j in (2, 0, 3, 1)])
    s2, _ = hs.score_example(model, *hs.render_example(permuted, enc, 1024)[:2], device, autocast=False)
    assert torch.allclose(s2, sum_loss[[2, 0, 3, 1]], atol=1e-3)
    # 4) context truncation keeps the endings whole
    t_short, m_short, truncated = hs.render_example(e, enc, n_ctx // 2 + int(mask.sum(1).max()))
    assert truncated and torch.equal(m_short.sum(1), mask.sum(1))
    print(f'example 0: {n_ctx} context tokens | sum losses {[round(v, 3) for v in sum_loss.tolist()]} | label {e["label"]}')
    print('first-token alignment, padding invariance, ending permutation, and truncation checks passed')

    # 5) a small run on the same indices for OpenAI's GPT-2 and a random-init model
    indices = hs.select_indices(n_small, 1337)
    assert indices == hs.select_indices(n_small, 1337) and indices != hs.select_indices(n_small, 1338)
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.time()
        r_gpt2 = hs.evaluate(model, examples, indices, Path(tmp) / 'gpt2.json', device, dict(name='gpt2'), save_every=10)
        per_example = (time.time() - t0) / n_small
        del model
        torch.cuda.empty_cache()
        random_model = build_model(GPTConfig(), 1337, device)
        r_rand = hs.evaluate(random_model, examples, indices, Path(tmp) / 'random.json', device, dict(name='random-init'), save_every=10)
        del random_model
    print(f'{per_example * 1000:.0f} ms per example in float32 -> about {per_example * 1000 / 60:.1f} min for 1,000 examples, '
          f'{per_example * 10042 / 60:.1f} min for all 10,042')
    print('HellaSwag checks passed.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default=str(WEEK_DIR / 'data' / 'fineweb_edu'))
    p.add_argument('--only', nargs='*', default=['loader', 'model', 'train', 'hellaswag'])
    args = p.parse_args()
    if 'loader' in args.only:
        check_loader(args.data_dir)
    if 'model' in args.only:
        check_model()
    if 'train' in args.only:
        check_training_and_resume(args.data_dir)
    if 'hellaswag' in args.only:
        check_hellaswag()
    print('\nAll selected checks passed.')
