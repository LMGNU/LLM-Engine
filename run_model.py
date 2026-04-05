import torch
from torch.nn import functional as F
from transformer import GPTLanguageModel, decode, device, block_size
import os

def run_inference():
    model_path = 'best_model.pt'

    if not os.path.exists(model_path):
        print(f"[Error] Model file '{model_path}' not found.")
        print("Make sure 'best_model.pt' is in the same directory as this script.")
        return

    print("--- Loading Pre-trained Model ---")
    model = GPTLanguageModel().to(device)

    # weights_only=True avoids a security warning in newer PyTorch versions
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    print("Model loaded successfully. Generating text (Ctrl+C to stop)...\n")
    print("-" * 50)

    # Start from a blank context token
    context = torch.zeros((1, 1), dtype=torch.long, device=device)

    try:
        with torch.no_grad():
            while True:
                idx_cond = context[:, -block_size:]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                context = torch.cat((context, idx_next), dim=1)
                print(decode([idx_next[0].item()]), end='', flush=True)

    except KeyboardInterrupt:
        print("\n\n[Stopped by user]")

if __name__ == "__main__":
    run_inference()