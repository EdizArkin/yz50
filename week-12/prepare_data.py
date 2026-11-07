"""Stream FineWeb-Edu, tokenize it with the GPT-2 tokenizer, and write uint16 token shards.

Every document is assigned to train or val by a seeded hash of its FineWeb id, before it is
tokenized, so the two splits are disjoint at the document level. Each split has its own
token stream and its own shards: a document that crosses a shard limit continues in the next
shard of the same split, never in the other split.

    python week-12/prepare_data.py --train_tokens 40_000_000 --val_tokens 1_000_000 --val_fraction 0.025
"""
import argparse
import hashlib
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tiktoken

DATASET = 'HuggingFaceFW/fineweb-edu'
SUBSET = 'sample-10BT'


def is_validation(doc_id, seed, val_fraction):
    # a deterministic coin flip per document: the same id always goes to the same split
    h = hashlib.sha256(f'{seed}:{doc_id}'.encode('utf-8')).digest()
    return int.from_bytes(h[:8], 'big') / 2**64 < val_fraction


def save_atomic(path, array):
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'wb') as f:
        np.save(f, array)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path) # the final name only ever points to a complete file


class ShardWriter:
    # one continuous token stream per split, cut into shards of exactly shard_size tokens

    def __init__(self, out_dir, prefix, split, shard_size, budget):
        self.out_dir, self.prefix, self.split = out_dir, prefix, split
        self.shard_size, self.budget = shard_size, budget
        self.buffer = np.empty(shard_size, dtype=np.uint16)
        self.fill = 0
        self.total = 0 # tokens written to this split
        self.shards = [] # (file name, number of tokens)
        self.documents = 0
        self.truncated_last_document = False

    @property
    def full(self):
        return self.total >= self.budget

    def add_document(self, tokens):
        # returns the number of tokens kept; the document that reaches the budget is cut there
        keep = min(len(tokens), self.budget - self.total)
        if keep < len(tokens):
            self.truncated_last_document = True
        tokens = tokens[:keep]
        i = 0
        while i < len(tokens):
            n = min(self.shard_size - self.fill, len(tokens) - i)
            self.buffer[self.fill:self.fill + n] = tokens[i:i + n]
            self.fill += n
            i += n
            if self.fill == self.shard_size:
                self.flush()
        self.total += keep
        self.documents += 1
        if self.full:
            self.flush()
        return keep

    def flush(self):
        if self.fill == 0:
            return
        name = f'{self.prefix}_{self.split}_{len(self.shards):06d}.npy'
        save_atomic(self.out_dir / name, self.buffer[:self.fill].copy())
        self.shards.append((name, self.fill))
        print(f'  wrote {name}: {self.fill:,} tokens')
        self.fill = 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out_dir', default=str(Path(__file__).parent / 'data' / 'fineweb_edu'))
    parser.add_argument('--train_tokens', type=int, default=40_000_000)
    parser.add_argument('--val_tokens', type=int, default=1_000_000)
    parser.add_argument('--shard_size', type=int, default=5_000_000)
    parser.add_argument('--val_fraction', type=float, default=0.025, help='probability that a document goes to val')
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--batch_docs', type=int, default=256, help='documents tokenized together')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assert not list(out_dir.glob('*.npy')), f'{out_dir} already has shards; use a new directory'

    from datasets import load_dataset # imported here so that --help works without it
    from huggingface_hub import HfApi
    info = HfApi().dataset_info(DATASET)
    revision = info.sha # pin the exact version that was read
    # sample-10BT is stored in 14 files ordered by crawl date; reading one document from each
    # file in turn spreads the sample over all of them instead of only the earliest crawl
    files = sorted(s.rfilename for s in info.siblings if s.rfilename.startswith('sample/10BT/') and s.rfilename.endswith('.parquet'))
    assert len(files) == 14, files
    streams = [iter(load_dataset(DATASET, data_files=f, split='train', streaming=True, revision=revision)) for f in files]

    def round_robin():
        while True:
            for k, s in enumerate(streams):
                doc = next(s, None)
                if doc is None:
                    return
                doc['source_file'] = files[k]
                yield doc
    stream = round_robin()
    file_counts = {'train': Counter(), 'val': Counter()}

    enc = tiktoken.get_encoding('gpt2')
    eot = enc._special_tokens['<|endoftext|>'] # 50256, the document separator
    assert enc.n_vocab == 50257 and eot == 50256 < 2**16 # every id fits in uint16

    writers = {split: ShardWriter(out_dir, 'fineweb_edu', split, args.shard_size, budget)
               for split, budget in [('train', args.train_tokens), ('val', args.val_tokens)]}
    split_docs = {'train': [], 'val': []} # FineWeb ids, in stream order
    split_hashes = {'train': set(), 'val': set()} # sha1 of each document's text
    skipped = Counter() # documents assigned to a split that was already full
    dumps = {'train': Counter(), 'val': Counter()}
    streamed = 0
    t0 = time.time()

    def process(batch):
        texts = [doc['text'] for doc in batch]
        for doc, ids in zip(batch, enc.encode_ordinary_batch(texts, num_threads=8)):
            split = 'val' if is_validation(doc['id'], args.seed, args.val_fraction) else 'train'
            writer = writers[split]
            if writer.full:
                skipped[split] += 1 # dropped, not moved to the other split
                continue
            # every document starts with <|endoftext|>, so documents are separated by it
            writer.add_document(np.array([eot] + ids, dtype=np.uint16))
            split_docs[split].append(doc['id'])
            split_hashes[split].add(hashlib.sha1(doc['text'].encode('utf-8')).hexdigest())
            dumps[split][doc.get('dump', 'unknown')] += 1
            file_counts[split][doc['source_file']] += 1

    batch = []
    for doc in stream:
        streamed += 1
        batch.append(doc)
        if len(batch) == args.batch_docs:
            process(batch)
            batch = []
            if all(w.full for w in writers.values()):
                break
            if streamed % (args.batch_docs * 40) == 0:
                rate = writers['train'].total / (time.time() - t0)
                print(f'{streamed:,} documents streamed | train {writers["train"].total:,} | val {writers["val"].total:,} tokens | {rate:,.0f} train tokens/s')
    if batch and not all(w.full for w in writers.values()):
        process(batch)
    for w in writers.values():
        w.flush()
    elapsed = time.time() - t0

    # disjointness: no FineWeb id and no identical text appears in both splits
    id_overlap = set(split_docs['train']) & set(split_docs['val'])
    text_overlap = split_hashes['train'] & split_hashes['val']
    assert not id_overlap, f'{len(id_overlap)} documents are in both splits'
    for split, w in writers.items():
        assert sum(n for _, n in w.shards) == w.total
        for name, n in w.shards:
            arr = np.load(out_dir / name, mmap_mode='r')
            assert arr.dtype == np.uint16 and arr.shape == (n,) and int(arr.max()) < 50257

    meta = {
        'source_dataset': DATASET,
        'subset': SUBSET,
        'source_url': f'https://huggingface.co/datasets/{DATASET}',
        'revision': revision,
        'source_files': files,
        'acquisition': 'datasets.load_dataset(streaming=True) per parquet file; no full download',
        'sampling_method': ('round-robin over the 14 sample/10BT parquet files (one document from each file in turn), '
                            'each file read from its start without shuffling; '
                            'each document assigned to val if sha256(f"{seed}:{id}") < val_fraction, else to train; '
                            'a split that reached its budget drops further documents assigned to it'),
        'tokenizer': 'tiktoken gpt2 (encode_ordinary), <|endoftext|> = 50256 prepended to every document',
        'dtype': 'uint16',
        'seed': args.seed,
        'val_fraction': args.val_fraction,
        'shard_size_tokens': args.shard_size,
        'documents_streamed': streamed,
        'documents_skipped_split_full': dict(skipped),
        'exact_text_duplicates_across_splits': len(text_overlap),
        'preparation_date_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'preparation_seconds': round(elapsed, 1),
        'splits': {split: {
            'tokens': w.total,
            'budget': w.budget,
            'documents': w.documents,
            'last_document_truncated_at_budget': w.truncated_last_document,
            'shards': [{'file': name, 'tokens': n} for name, n in w.shards],
            'crawl_dumps': dict(sorted(dumps[split].items())),
            'documents_per_source_file': dict(sorted(file_counts[split].items())),
        } for split, w in writers.items()},
    }
    (out_dir / 'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    (out_dir / 'split_documents.json').write_text(json.dumps(split_docs), encoding='utf-8')

    print(f'\nstreamed {streamed:,} documents in {elapsed:.0f} s')
    for split, w in writers.items():
        print(f'{split:5s}: {w.total:>12,} tokens | {w.documents:,} documents | {len(w.shards)} shards')
    print(f'documents skipped because their split was full: {dict(skipped)}')
    print(f'shared FineWeb ids: {len(id_overlap)} | identical texts in both splits: {len(text_overlap)}')
    print('metadata:', out_dir / 'metadata.json')


if __name__ == '__main__':
    main()
