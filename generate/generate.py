import torch
from torch.nn import functional as F
from model import GPTLanguageModel, decode, device, block_size
import os

def run_inference():
    model_path = '../best_model.pt'  # points up one level to where it was saved

    if not os.path.exists(model_path):
        print(f"[Error] '{model_path}' not found.")
        return

    print("--- Loading Pre-trained Model ---")
    model = GPTLanguageModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print("Model loaded. Generating text (Ctrl+C to stop)...\n")
    print("-" * 50)

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

if __name__ == "__main__":
    run_inference()