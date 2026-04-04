import os
import torch
from transformer import GPTLanguageModel, block_size # for local train 


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = "best_model.pt"
    data_path = "data.txt"

    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return 1

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    if len(text) < block_size + 1:
        print(
            f"Error: data too small for block_size={block_size}. "
            f"Need at least {block_size + 1} characters."
        )
        return 1

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]

    print(f"Detected Vocab Size: {vocab_size}")

    model = GPTLanguageModel(vocab_size).to(device)

    if not os.path.exists(save_path):
        print(f"Error: {save_path} not found.")
        return 1

    try:
        state = torch.load(save_path, map_location=device)
        model.load_state_dict(state)
    except RuntimeError as e:
        print("Error: failed to load weights into the current model.")
        print("This usually means the saved weights were created with")
        print("different hyperparameters or a different architecture.")
        print(f"Details: {e}")
        return 1

    model.eval()
    print(f"Weights loaded successfully from {save_path}")

    data = torch.tensor(encode(text), dtype=torch.long)

    # Simple random-batch evaluation to estimate loss
    batch_size = 64
    eval_iters = 50

    def get_batch():
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    losses = []
    with torch.no_grad():
        for _ in range(eval_iters):
            xb, yb = get_batch()
            _, loss = model(xb, yb)
            losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)
    print(f"Evaluation loss (avg over {eval_iters} iters): {avg_loss:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())