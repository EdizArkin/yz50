"""GPT-2 (124M) shared by train.py, evaluate_hellaswag.py and training_gpt2.ipynb.

The modules are my Week 10/11 definitions, unchanged: fused c_attn, GELU with the tanh
approximation, pre-normalized blocks, lm_head tied to wte, std 0.02 initialization with
1/sqrt(2 * n_layer) for the residual projections, AdamW groups that decay only 2D
parameters, and the warmup + cosine learning rate. Week 12 only collects them in one file.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

GPT2_VOCAB_SIZE = 50257 # 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
PADDED_VOCAB_SIZE = 50304 # 2^7 * 393, the hardware-friendly size from Week 11


@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = PADDED_VOCAB_SIZE # padded; ids >= 50257 are never inputs or targets
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    sdpa: bool = True # True: F.scaled_dot_product_attention, False: explicit attention as in Week 10


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.sdpa = config.sdpa
        if not self.sdpa:
            # causal mask, not a learned weight and not saved in the state dict
            tril = torch.tril(torch.ones(config.block_size, config.block_size))
            self.register_buffer('tril', tril.view(1, 1, config.block_size, config.block_size), persistent=False)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        qkv = self.c_attn(x) # (B,T,3C)
        q, k, v = qkv.split(self.n_embd, dim=2) # each (B,T,C)
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B,nh,T,hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B,nh,T,hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B,nh,T,hs)
        if self.sdpa:
            # one optimized kernel; PyTorch chooses the backend
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # attention (materializes the large (T,T) matrix for all the queries and keys)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B,nh,T,T)
            att = att.masked_fill(self.tril[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v # (B,nh,T,T) @ (B,nh,T,hs) -> (B,nh,T,hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x) # (B,T,4C)
        x = self.gelu(x)
        x = self.c_proj(x) # (B,T,C)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # communication
        x = x + self.mlp(self.ln_2(x)) # computation
        return x


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme: one (vocab_size, n_embd) matrix for the input and the output
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def init_gpt2_weights(model):
    n_layer = model.config.n_layer
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            std = 0.02
            if name.endswith('c_proj'): # projections that add into the residual stream
                std *= (2 * n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)


def build_model(config, seed, device):
    torch.manual_seed(seed) # the same seed always gives the same initial weights
    model = GPT(config)
    init_gpt2_weights(model)
    return model.to(device)


def from_pretrained(model_type='gpt2', sdpa=True):
    # Loads pretrained GPT-2 model weights from huggingface, as in Week 10
    from transformers import GPT2LMHeadModel
    assert model_type == 'gpt2' # GPT-2 small (124M), the reference model
    config = GPTConfig(n_layer=12, n_head=12, n_embd=768, vocab_size=GPT2_VOCAB_SIZE, block_size=1024, sdpa=sdpa)
    model = GPT(config)
    sd = model.state_dict() # tensors share storage with the model's parameters
    sd_keys = set(sd.keys())

    model_hf = GPT2LMHeadModel.from_pretrained('openai-community/gpt2')
    sd_hf = model_hf.state_dict()
    sd_keys_hf = set(sd_hf.keys())
    assert sd_keys == sd_keys_hf, f"missing: {sorted(sd_keys_hf - sd_keys)}, unexpected: {sorted(sd_keys - sd_keys_hf)}"

    # the openai checkpoints use a "Conv1D" module, but I only want to use a vanilla Linear
    # this means that I have to transpose these weights when I import them
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    for k in sorted(sd_keys_hf):
        if any(k.endswith(w) for w in transposed):
            # special treatment for the Conv1D weights we need to transpose
            assert sd_hf[k].shape[::-1] == sd[k].shape, k
            with torch.no_grad():
                sd[k].copy_(sd_hf[k].t())
        else:
            # vanilla copy over the other parameters
            assert sd_hf[k].shape == sd[k].shape, k
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])
    assert model.lm_head.weight is model.transformer.wte.weight
    return model


def configure_optimizers(model, learning_rate, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8, fused=False):
    param_dict = {name: p for name, p in model.named_parameters() if p.requires_grad} # one entry per unique parameter
    decay_params = [p for p in param_dict.values() if p.dim() >= 2]
    nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ]
    return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps, fused=fused)


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    # 1) linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # 2) after the horizon, stay at the minimum
    if step > max_steps:
        return min_lr
    # 3) in between, cosine decay down to the minimum
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # goes from 1 to 0
    return min_lr + coeff * (max_lr - min_lr)


def set_tf32(enabled):
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    torch.set_float32_matmul_precision('high' if enabled else 'highest')


def sample_next(logits, temperature=1.0, top_k=None, generator=None):
    # logits: (B, vocab_size) scores for the next token -> (B, 1) sampled token IDs
    logits = logits / temperature
    if top_k is not None:
        topk_logits, topk_ids = torch.topk(logits, top_k, dim=-1) # (B, k), sorted
        probs = F.softmax(topk_logits, dim=-1) # renormalize over the k candidates only
        ix = torch.multinomial(probs, num_samples=1, generator=generator) # (B, 1) index into the top-k
        return torch.gather(topk_ids, -1, ix) # (B, 1) token IDs
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=50, generator=None, autocast=True):
    # idx is (B, T) array of indices in the current context
    was_training = model.training
    model.eval()
    for _ in range(max_new_tokens):
        # crop idx to the last block_size tokens
        idx_cond = idx[:, -model.config.block_size:]
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=autocast and idx.is_cuda):
            logits, _ = model(idx_cond)
        # focus only on the last time step; the padded ids >= 50257 are not real tokens and
        # tiktoken cannot decode them, so they are never candidates
        logits = logits[:, -1, :GPT2_VOCAB_SIZE].float() # (B, 50257)
        idx_next = sample_next(logits, temperature, top_k, generator) # (B, 1)
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    model.train(was_training)
    return idx
