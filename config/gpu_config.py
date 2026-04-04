# ── Cell 5: Configure hyperparameters ─────────────────────
# These are GPU-optimised settings (much larger than CPU defaults)
# Adjust if you want to experiment

BATCH_SIZE    = 64     # (was 16 on CPU)
BLOCK_SIZE    = 256    # (was 128 on CPU)
MAX_ITERS     = 5000   # (was 3000 on CPU)
EVAL_INTERVAL = 250
LEARNING_RATE = 3e-4
EVAL_ITERS    = 100
N_EMBD        = 384    # (was 128 on CPU)
N_HEAD        = 6      # (was 4 on CPU)
N_LAYER       = 6      # (was 4 on CPU)
DROPOUT       = 0.2

# Estimated param count
approx_params = (N_EMBD * 256 + N_EMBD * 256 +   # embeddings
                 N_LAYER * (4 * N_EMBD**2 * 3 + N_EMBD**2)) / 1e6
print(f"   Hyperparameters set")
print(f"   batch_size={BATCH_SIZE}, block_size={BLOCK_SIZE}")
print(f"   n_embd={N_EMBD}, n_head={N_HEAD}, n_layer={N_LAYER}")
print(f"   max_iters={MAX_ITERS}, lr={LEARNING_RATE}")
print(f"   Approx parameters: ~10M (GPU-scale model)")