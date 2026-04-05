import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import sys
import os
from config.config import *

print("=" * 60)
print(" Core Transformer-Engine - Training Pipeline")
print("Eamon2009")
print("=" * 60)
print(f"\n[INFO] Starting at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"[INFO] Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

start = time.time()

# Hyperparameters
batch_size    = 16
block_size    = 128
max_iters     = 3000
eval_interval = 200
learning_rate = 3e-4
device        = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters    = 50
n_embd        = 128
n_head        = 4
n_layer       = 4
dropout       = 0.2

print(f"\n[CONFIG] Hyperparameters loaded:")
print(f"         batch_size={batch_size}, block_size={block_size}")
print(f"         max_iters={max_iters}, learning_rate={learning_rate}")
print(f"         n_embd={n_embd}, n_head={n_head}, n_layer={n_layer}, dropout={dropout}")

torch.manual_seed(seed)

# ── Data loading ───────────────────────────────────────────
print(f"\n[DATA]  Loading text from: {cleaned_path}")
with open(cleaned_path, 'r', encoding='utf-8') as f:
    text2 = f.read()

chars2     = sorted(list(set(text2)))
vocab_size = len(chars2)
stri       = {ch: i for i, ch in enumerate(chars2)}
it         = {i: ch for i, ch in enumerate(chars2)}
encode     = lambda s: [stri[c] for c in s]
decode     = lambda l: ''.join([it[i] for i in l])

data = torch.tensor(encode(text2), dtype=torch.long)
n          = int(train_split * len(data))
train_data = data[:n]
val_data   = data[n:]

print(f"[DATA]  Total characters : {len(text2):,}")
print(f"[DATA]  Vocabulary size  : {vocab_size}")
print(f"[DATA]  Train tokens     : {len(train_data):,}")
print(f"[DATA]  Val   tokens     : {len(val_data):,}")


# ── Batch / loss helpers ───────────────────────────────────
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix   = torch.randint(len(data) - block_size, (batch_size,))
    x    = torch.stack([data[i:i + block_size]         for i in ix])
    y    = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y      = get_batch(split)
            _, loss   = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Model 
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads   = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj    = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa   = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks                   = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f                     = nn.LayerNorm(n_embd)
        self.lm_head                  = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T    = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x       = tok_emb + pos_emb
        x       = self.blocks(x)
        x       = self.ln_f(x)
        logits  = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits  = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss    = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond  = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits    = logits[:, -1, :]
            probs     = F.softmax(logits, dim=-1)
            idx_next  = torch.multinomial(probs, num_samples=1)
            idx       = torch.cat((idx, idx_next), dim=1)
        return idx


# ── Build model ────────────────────────────────────────────
print(f"\n[MODEL] Building GPTLanguageModel...")
model    = GPTLanguageModel().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"[MODEL] Parameters  : {n_params / 1e6:.2f} M  ({n_params:,} total)")
print(f"[MODEL] Architecture: {n_layer} layers x {n_head} heads x {n_embd} embedding dim")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
print(f"[OPTIM] AdamW optimizer, lr={learning_rate}")


# ── Training loop ──────────────────────────────────────────
print(f"\n{'─' * 60}")
print(f"  TRAINING  ({max_iters} iterations, eval every {eval_interval})")
print(f"{'─' * 60}")

best_val_loss = float('inf')
train_start   = time.time()

for iter in range(max_iters):

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses   = estimate_loss()
        elapsed  = time.time() - train_start
        pct      = 100 * iter / max_iters
        eta_secs = (elapsed / (iter + 1)) * (max_iters - iter - 1) if iter > 0 else 0
        improved = " << best!" if losses['val'] < best_val_loss else ""

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save(model.state_dict(), 'best_model.pt')

        print(f"[{iter:>5}/{max_iters}] {pct:5.1f}%  "
              f"train={losses['train']:.4f}  val={losses['val']:.4f}  "
              f"elapsed={elapsed:.0f}s  ETA={eta_secs:.0f}s{improved}")
        sys.stdout.flush()

    xb, yb       = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

total_time = time.time() - train_start
print(f"\n[DONE]  Training finished in {total_time:.1f}s "
      f"({total_time / 60:.1f} min)  |  Best val loss: {best_val_loss:.4f}")
print(f"[SAVE]  Best weights saved to best_model.pt")


# ── Infinite generation in same terminal ───────────────────
print(f"\n{'─' * 60}")
print(f"  MODEL OUTPUT  (Ctrl+C to stop)")
print(f"{'─' * 60}\n")

model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)

try:
    with torch.no_grad():
        while True:
            idx_cond = context[:, -block_size:]
            logits, _ = model(idx_cond)
            logits    = logits[:, -1, :]
            probs     = F.softmax(logits, dim=-1)
            idx_next  = torch.multinomial(probs, num_samples=1)
            context   = torch.cat((context, idx_next), dim=1)
            if context.shape[1] > block_size:
                context = context[:, -block_size:]
            print(decode([idx_next[0].item()]), end='', flush=True)
except KeyboardInterrupt:
    print("\n\n[Stopped by user]")

end = time.time()
print(f"\n[TOTAL] Wall-clock time: {end - start:.1f}s  ({(end - start) / 60:.1f} min)")