"""HellaSwag evaluation of my GPT-2 and of OpenAI's GPT-2 with one scoring implementation.

Adapted from Karpathy's build-nanogpt hellaswag.py. For each example, the context is followed
by each of the four endings (with a leading space), the model scores every row, and only the
ending tokens contribute to the loss. The prediction is the ending with the lowest average
per-token loss (acc_norm here) or the lowest summed loss (acc). The label is only used after
the four losses are computed.

The official GitHub copy (rowanz/hellaswag) is blocked by a DMCA notice since 2026-09-14, so the
validation split is read from the dataset author's Hugging Face repository, Rowan/hellaswag.

    python week-12/evaluate_hellaswag.py --model mine --checkpoint week-12/checkpoints/main/ckpt_step_002000.pt
    python week-12/evaluate_hellaswag.py --model gpt2
    python week-12/evaluate_hellaswag.py --compare week-12/results/hellaswag_mine.json week-12/results/hellaswag_gpt2.json
"""
import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import tiktoken
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).parent))
from gpt2_model import GPT, GPTConfig, GPT2_VOCAB_SIZE, from_pretrained, set_tf32

WEEK_DIR = Path(__file__).parent
HF_REPO = 'Rowan/hellaswag'
HF_FILE = 'data/validation-00000-of-00001.parquet'
N_VALIDATION = 10042


def load_hellaswag_val(cache_dir=WEEK_DIR / 'data' / 'hellaswag'):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    jsonl = cache_dir / 'hellaswag_val.jsonl'
    source = cache_dir / 'source.json'
    if not jsonl.exists():
        import pyarrow.parquet as pq
        from huggingface_hub import HfApi, hf_hub_download
        revision = HfApi().dataset_info(HF_REPO).sha # pin the exact version that was read
        parquet = hf_hub_download(HF_REPO, HF_FILE, repo_type='dataset', revision=revision, local_dir=cache_dir)
        rows = pq.read_table(parquet).to_pylist()
        with open(jsonl, 'w', encoding='utf-8') as f:
            for row in rows:
                row['label'] = int(row['label'])
                f.write(json.dumps(row) + '\n')
        source.write_text(json.dumps(dict(repo=HF_REPO, file=HF_FILE, revision=revision, rows=len(rows),
                                          downloaded_utc=datetime.now(timezone.utc).isoformat(timespec='seconds')),
                                     indent=2), encoding='utf-8')
    examples = [json.loads(line) for line in open(jsonl, encoding='utf-8')]
    assert len(examples) == N_VALIDATION, f'expected {N_VALIDATION} validation examples, found {len(examples)}'
    assert all(len(e['endings']) == 4 and e['label'] in (0, 1, 2, 3) for e in examples)
    return examples, json.loads(source.read_text(encoding='utf-8'))


def select_indices(n_examples, sample_seed):
    # a seeded sample of validation indices, sorted; the same for every model
    if n_examples >= N_VALIDATION:
        return list(range(N_VALIDATION))
    return sorted(random.Random(sample_seed).sample(range(N_VALIDATION), n_examples))


def render_example(example, enc, max_context):
    # (4, L) tokens and a (4, L) mask that is 1 on the ending tokens; no label in the output
    ctx_tokens = enc.encode(example['ctx'])
    rows, masks, truncated = [], [], False
    for ending in example['endings']:
        end_tokens = enc.encode(' ' + ending) # GPT-2 tokens carry their leading space
        row = ctx_tokens + end_tokens
        mask = [0] * len(ctx_tokens) + [1] * len(end_tokens)
        if len(row) > max_context:
            # drop the earliest context tokens; the ending is always kept whole
            assert len(end_tokens) < max_context
            row, mask, truncated = row[-max_context:], mask[-max_context:], True
        rows.append(row)
        masks.append(mask)
    L = max(len(r) for r in rows)
    tokens = torch.zeros((4, L), dtype=torch.long) # padding id 0 is never scored and only follows a row
    mask = torch.zeros((4, L), dtype=torch.long)
    for i, (r, m) in enumerate(zip(rows, masks)):
        tokens[i, :len(r)] = torch.tensor(r)
        mask[i, :len(m)] = torch.tensor(m)
    return tokens, mask, truncated


def completion_losses(logits, tokens, mask):
    # logits (4, L, V) -> summed and average loss over each row's ending tokens
    logits = logits[..., :GPT2_VOCAB_SIZE].float() # the same 50,257 real tokens for both models
    shift_logits = logits[:, :-1, :] # position t predicts token t + 1
    shift_tokens = tokens[:, 1:]
    shift_mask = mask[:, 1:] # aligned with the targets, so the first ending token is scored at the last context position
    losses = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), shift_tokens.reshape(-1), reduction='none')
    losses = losses.view(tokens.size(0), -1) * shift_mask
    sum_loss = losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    return sum_loss, avg_loss


@torch.no_grad()
def score_example(model, tokens, mask, device, autocast):
    tokens, mask = tokens.to(device), mask.to(device)
    try:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=autocast):
            logits, _ = model(tokens)
        return completion_losses(logits, tokens, mask)
    except torch.cuda.OutOfMemoryError:
        # fall back to one ending at a time; causal attention makes the result the same
        torch.cuda.empty_cache()
        sums, avgs = [], []
        for i in range(tokens.size(0)):
            n = int(mask[i].nonzero().max()) + 1 # this row's length without padding
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=autocast):
                logits, _ = model(tokens[i:i + 1, :n])
            s, a = completion_losses(logits, tokens[i:i + 1, :n], mask[i:i + 1, :n])
            sums.append(s)
            avgs.append(a)
        return torch.cat(sums), torch.cat(avgs)


def evaluate(model, examples, indices, out_path, device, model_info, max_context=1024, autocast=False, save_every=100):
    enc = tiktoken.get_encoding('gpt2')
    model.eval()
    records = []
    if out_path.exists():
        # continue an interrupted evaluation of the same model on the same indices
        previous = json.loads(out_path.read_text(encoding='utf-8'))
        if previous['indices'] == indices and previous['model'] == model_info and previous['autocast'] == autocast:
            records = previous['records']
            print(f'continuing from {len(records)} saved examples')
    done = {r['index'] for r in records}

    def write(final):
        n = len(records)
        correct = sum(r['pred_sum'] == r['label'] for r in records)
        correct_norm = sum(r['pred_norm'] == r['label'] for r in records)
        result = dict(model=model_info, dataset=dict(repo=HF_REPO, file=HF_FILE), n_examples=n, indices=indices,
                      complete=final and n == len(indices), max_context=max_context, autocast=autocast,
                      truncated_examples=sum(r['truncated'] for r in records),
                      correct=correct, accuracy=correct / max(n, 1),
                      correct_norm=correct_norm, accuracy_norm=correct_norm / max(n, 1), records=records)
        tmp = out_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(result), encoding='utf-8')
        tmp.replace(out_path)
        return result

    t0 = time.time()
    for k, i in enumerate(indices):
        if i in done:
            continue
        example = examples[i]
        tokens, mask, truncated = render_example(example, enc, max_context)
        sum_loss, avg_loss = score_example(model, tokens, mask, device, autocast)
        records.append(dict(index=i, label=example['label'], # the label is read only here, after scoring
                            pred_sum=int(sum_loss.argmin()), pred_norm=int(avg_loss.argmin()),
                            sum_loss=sum_loss.tolist(), avg_loss=avg_loss.tolist(),
                            length=tokens.size(1), truncated=truncated))
        if len(records) % save_every == 0:
            r = write(final=False)
            print(f'{len(records):5d}/{len(indices)} | acc {r["accuracy"]:.4f} | acc_norm {r["accuracy_norm"]:.4f} | {time.time() - t0:.0f} s')
    result = write(final=True)
    print(f'{model_info["name"]}: {result["n_examples"]} examples | acc {result["correct"]}/{result["n_examples"]} = {result["accuracy"]:.4f} '
          f'| acc_norm {result["correct_norm"]}/{result["n_examples"]} = {result["accuracy_norm"]:.4f} | {time.time() - t0:.0f} s')
    return result


def load_my_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = GPT(GPTConfig(**ckpt['model_config']))
    model.load_state_dict(ckpt['model'])
    info = dict(name='mine', checkpoint=str(Path(checkpoint).name), step=ckpt['step'],
                tokens_processed=ckpt['tokens_processed'], seq_len=ckpt['args']['seq_len'])
    return model.to(device), info


def wilson_interval(k, n, z=1.96):
    p = k / n
    center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return center - half, center + half


def compare(path_a, path_b):
    a, b = (json.loads(Path(p).read_text(encoding='utf-8')) for p in (path_a, path_b))
    assert a['indices'] == b['indices'], 'the two evaluations used different examples'
    assert a['complete'] and b['complete']
    assert a['max_context'] == b['max_context'] and a['autocast'] == b['autocast']
    ra, rb = ({r['index']: r for r in x['records']} for x in (a, b))
    n = len(a['indices'])
    out = dict(n_examples=n, random_baseline=0.25)
    for key, pred in [('accuracy', 'pred_sum'), ('accuracy_norm', 'pred_norm')]:
        ca = [ra[i][pred] == ra[i]['label'] for i in a['indices']]
        cb = [rb[i][pred] == rb[i]['label'] for i in a['indices']]
        only_a = sum(x and not y for x, y in zip(ca, cb))
        only_b = sum(y and not x for x, y in zip(ca, cb))
        out[key] = {
            a['model']['name']: dict(correct=sum(ca), value=sum(ca) / n, wilson_95=wilson_interval(sum(ca), n)),
            b['model']['name']: dict(correct=sum(cb), value=sum(cb) / n, wilson_95=wilson_interval(sum(cb), n)),
            'difference_pp': 100 * (sum(cb) - sum(ca)) / n,
            # paired comparison on the same examples: only the discordant pairs matter (McNemar)
            'only_first_correct': only_a, 'only_second_correct': only_b,
            'mcnemar_chi2': (abs(only_a - only_b) - 1) ** 2 / (only_a + only_b) if only_a + only_b else 0.0,
        }
    print(json.dumps(out, indent=2))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', choices=['mine', 'gpt2'])
    p.add_argument('--checkpoint', help="my checkpoint, for --model mine")
    p.add_argument('--n_examples', type=int, default=1000)
    p.add_argument('--sample_seed', type=int, default=1337)
    p.add_argument('--max_context', type=int, default=1024)
    p.add_argument('--autocast', action='store_true', help='BF16 autocast; default is float32 without TF32')
    p.add_argument('--out', default=None)
    p.add_argument('--compare', nargs=2, metavar=('MINE_JSON', 'GPT2_JSON'))
    args = p.parse_args()
    if args.compare:
        out = compare(*args.compare)
        (WEEK_DIR / 'results' / 'hellaswag_comparison.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_tf32(False) # both models in full float32 unless --autocast
    examples, source = load_hellaswag_val()
    indices = select_indices(args.n_examples, args.sample_seed)
    if args.model == 'mine':
        model, info = load_my_model(args.checkpoint, device)
    else:
        model = from_pretrained('gpt2').to(device)
        info = dict(name='gpt2', source='openai-community/gpt2 via transformers, loaded into gpt2_model.GPT')
    info.update(sample_seed=args.sample_seed, dataset_revision=source['revision'])
    out = Path(args.out) if args.out else WEEK_DIR / 'results' / f'hellaswag_{info["name"]}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    evaluate(model, examples, indices, out, device, info, args.max_context, args.autocast)


if __name__ == '__main__':
    main()
