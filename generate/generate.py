import torch
from torch.nn import functional as F
from transformer import GPTLanguageModel, it, decode, block_size, device

# --- CONFIGURATION ---
MODEL_PATH = 'best_model.pt'
MAX_CONTEXT = block_size # 128 as per your training script

def load_model():
    print(f"[INFO] Loading model from {MODEL_PATH}...")
    # Initialize the architecture
    model = GPTLanguageModel()
    
    # Load the saved weights
    # map_location ensures it works on CPU even if trained on GPU
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval() # Switch to evaluation mode
    return model

def run_infinite_generation(model):
    print(f"\n{'─' * 60}")
    print(f"  MODEL OUTPUT - INFINITE GENERATION  (Ctrl+C to stop)")
    print(f"{'─' * 60}\n")

    # Start with a single '0' token (usually newline or start token)
    context = torch.zeros((1, 1), dtype=torch.long, device=device)

    try:
        with torch.no_grad():
            while True:
                # Crop context to the max block size the model supports
                idx_cond = context[:, -MAX_CONTEXT:]
                
                # Get predictions
                logits, _ = model(idx_cond)
                
                # Focus only on the last time step
                logits = logits[:, -1, :] 
                
                # Apply softmax to get probabilities
                probs = F.softmax(logits, dim=-1)
                
                # Sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                context = torch.cat((context, idx_next), dim=1)
                
                # Decode and print the character immediately
                char = decode([idx_next[0].item()])
                print(char, end='', flush=True)

    except KeyboardInterrupt:
        print("\n\n[INFO] Generation stopped by user.")

if __name__ == "__main__":
    # 1. Load the model
    gpt_model = load_model()
    
    # 2. Start generating
    run_infinite_generation(gpt_model)