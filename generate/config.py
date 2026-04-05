import torch

batch_size    = 16
block_size    = 128
n_embd        = 128
n_head        = 4
n_layer       = 4
dropout       = 0.2
device        = 'cuda' if torch.cuda.is_available() else 'cpu'