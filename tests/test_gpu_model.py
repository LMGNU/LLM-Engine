import torch
import pytest
import os
from gpu_train import GPTLanguageModel 

# ── Test Configuration ──────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VOCAB_SIZE = 65  # Standard for character-level Shakespeare
N_EMBD = 384
BLOCK_SIZE = 256

@pytest.fixture
def model():
    """Initializes the model on the correct device."""
    # We mock the global variables expected by your GPTLanguageModel class
    import gpu_train
    gpu_train.n_embd = N_EMBD
    gpu_train.n_head = 6
    gpu_train.n_layer = 6
    gpu_train.block_size = BLOCK_SIZE
    gpu_train.dropout = 0.0 # Disable dropout for deterministic testing
    gpu_train.vocab_size = VOCAB_SIZE
    gpu_train.device = DEVICE
    
    model = GPTLanguageModel().to(DEVICE)
    model.eval()
    return model

# ── Accuracy & Logic Tests ──────────────────────────────────
class TestTransformerLogic:

    def test_output_shape(self, model):
        """Verify the model returns the correct logit dimensions."""
        B, T = 4, 16
        idx = torch.randint(0, VOCAB_SIZE, (B, T)).to(DEVICE)
        logits, loss = model(idx)
        
        assert logits.shape == (B, T, VOCAB_SIZE), f"Expected (B,T,V), got {logits.shape}"
        assert loss is None

    def test_causality(self, model):
        """
        Verify that changing a future token does not affect past logits.
        This confirms the triangular mask (tril) is working.
        """
        B, T = 1, 10
        idx = torch.randint(0, VOCAB_SIZE, (B, T)).to(DEVICE)
        
        # Get original logits
        logits_orig, _ = model(idx)
        
        # Change the VERY LAST token
        idx_changed = idx.clone()
        idx_changed[0, -1] = (idx[0, -1] + 1) % VOCAB_SIZE
        logits_new, _ = model(idx_changed)
        
        # All logits except the last one should be identical
        assert torch.allclose(logits_orig[:, :-1, :], logits_new[:, :-1, :], atol=1e-5), \
            "Causality violation: Future token affected the past!"

    def test_loss_calculation(self, model):
        """Verify loss is a scalar and is reasonably positive."""
        B, T = 4, 32
        idx = torch.randint(0, VOCAB_SIZE, (B, T)).to(DEVICE)
        targets = torch.randint(0, VOCAB_SIZE, (B, T)).to(DEVICE)
        
        logits, loss = model(idx, targets)
        
        assert loss.item() > 0
        assert not torch.isnan(loss), "Loss is NaN!"

# ── GPU Performance Tests ──────────────────────────────────
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
class TestGPUSpecifics:

    def test_mixed_precision_compatibility(self, model):
        """Verify the model runs under torch.cuda.amp (Mixed Precision)."""
        B, T = 8, 64
        idx = torch.randint(0, VOCAB_SIZE, (B, T)).to(DEVICE)
        targets = torch.randint(0, VOCAB_SIZE, (B, T)).to(DEVICE)
        
        with torch.cuda.amp.autocast():
            logits, loss = model(idx, targets)
            
        assert logits.dtype == torch.float16 or logits.dtype == torch.bfloat16
        assert not torch.isnan(loss)

    def test_memory_usage(self, model):
        """Ensure model and data are actually on the GPU."""
        B, T = 2, 32
        idx = torch.randint(0, VOCAB_SIZE, (B, T)).to(DEVICE)
        
        # Check model parameter location
        param_device = next(model.parameters()).device.type
        assert param_device == 'cuda', f"Model is on {param_device}, expected cuda"
        
        # Trigger forward pass to check for memory errors
        logits, _ = model(idx)
        assert logits.is_cuda

# ── Training Flow Test ─────────────────────────────────────
def test_optimization_step(model):
    """Verify that one step of AdamW actually updates weights."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    B, T = 2, 8
    idx = torch.randint(0, VOCAB_SIZE, (B, T)).to(DEVICE)
    targets = torch.randint(0, VOCAB_SIZE, (B, T)).to(DEVICE)
    
    # Capture weight before update
    initial_weight = model.lm_head.weight.clone()
    
    # Simple train step
    logits, loss = model(idx, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Check that weights moved
    assert not torch.equal(initial_weight, model.lm_head.weight), "Weights did not update after step"

if __name__ == "__main__":
    print(f"Testing Model on: {DEVICE.upper()}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        
    pytest.main([__file__, "-v"])