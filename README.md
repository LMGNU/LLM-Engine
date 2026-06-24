# Core Transformer-Engine 

A character-level GPT transformer built from scratch in PyTorch, trained on Linux kernel C source code to generate kernel-style C code character by character. No pre-trained weights. No fine-tuning. Pure architecture and training from zero.
---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Project Structure](#project-structure)
3. [How It Works](#how-it-works)
4. [Setup & Requirements](#setup--requirements)
5. [How to Run](#how-to-run)
6. [Configuration](#configuration)
7. [What to Expect as Output](#what-to-expect-as-output)
8. [Training Results — CPU Run](#training-results--cpu-run)
9. [Training Results — GPU Run](#training-results--gpu-run)
10. [CPU vs GPU — Head to Head](#cpu-vs-gpu--head-to-head)
11. [Overfitting — What Happened and Why](#overfitting--what-happened-and-why)
12. [How Weights Produce Output](#how-weights-produce-output)
13. [Scaling Laws — And Where Your Model Sits](#scaling-laws--and-where-your-model-sits)
14. [Known Limitations](#known-limitations)

---

## What This Project Does

This project trains a small GPT-style transformer model on Linux kernel C source code and then generates new kernel-like C code character by character — infinitely — in a separate terminal window.

It is a learning project. The goal is not to produce production-quality code, but to understand how language models learn patterns from text and to see that process happen live on your own machine.

---

## Project Structure

```
LLM-Engine/
│
├── transformer.py        # Main training script — GPT model
├── train.py              # Secondary script — simple Bigram model (for comparison)
│
├── config/
│   └── config.py         # Central config: paths, seed, train split
│
├── cleaned.txt           # Training data — Linux kernel C source
├── best_model.pt         # Best weights saved during training (lowest val loss)
│
├── _gen_weights.pt       # Auto-generated after training (deleted on exit)
├── _gen_vocab.pt         # Auto-generated after training (deleted on exit)
└── _gen_worker.py        # Auto-generated after training (deleted on exit)
```

> The three `_gen_*` files are created automatically when training finishes and cleaned up automatically when the main script exits. You do not need to manage them manually.

---

## How It Works

The model is a **character-level transformer**. This means:

- It reads your text file one character at a time
- It learns which characters tend to follow other characters in which contexts
- At generation time it predicts the next character, then the next, then the next — forever

It is the same core architecture as GPT, just much smaller and trained on much less data.

**The pipeline in order:**

```
cleaned.txt
    ↓
Characters encoded as integers (vocab size: 95–105)
    ↓
Model trains on sequences of 128–256 characters at a time
    ↓
Every 200–250 steps: loss is measured and printed
    ↓
Best weights saved to best_model.pt whenever val loss improves
    ↓
After training: weights saved, new CMD window opens
    ↓
New window loads weights and streams generated code forever
```

---

## Setup & Requirements

**Python version:** 3.8 or higher

**Install dependencies:**

```bash
pip install torch
```

No other dependencies needed. The project uses only PyTorch and Python standard library modules.

**Your `config/config.py` should look like this:**

```python
cleaned_path = "cleaned.txt"
train_split  = 0.9
seed         = 42
```

---

## How to Run

```bash
python transformer.py
```

That is it. The script will:

1. Print a startup banner with device and timestamp
2. Load and report stats on your dataset
3. Build the model and print parameter count
4. Train for the configured number of steps, printing progress at each eval interval
5. Automatically open a new CMD window with infinite generation when done

**To stop generation:** close the CMD window or press `Ctrl+C` inside it.  
**To stop training early:** press `Ctrl+C` in the main window.

---

## Configuration

All hyperparameters are at the top of `transformer.py`.

### CPU Configuration (no GPU)

Tuned for a CPU-only machine with limited RAM:

```python
batch_size    = 16      # How many sequences to train on at once
block_size    = 128     # How many characters the model sees at once
max_iters     = 3000    # Total training steps
eval_interval = 200     # Print progress every N steps
eval_iters    = 50      # Batches averaged for loss estimate
learning_rate = 3e-4    # How fast the model updates
n_embd        = 128     # Size of internal representations
n_head        = 4       # Number of attention heads
n_layer       = 4       # Number of transformer blocks
dropout       = 0.2     # Regularization — prevents memorization
```

**Parameter count with these settings: 0.83M parameters**

### GPU Configuration (CUDA)

```python
batch_size    = 64
block_size    = 256
n_embd        = 384
n_head        = 6
n_layer       = 6
max_iters     = 5000
eval_interval = 250
eval_iters    = 200
```

**Parameter count with GPU settings: ~10.8M parameters**

---

## What to Expect as Output

After training, generated output looks roughly like:

```c
static int module_get(struct module *mod)
{
    if (!mod->state == MODULE_STATE_LIVE)
        return -ENOENT;
    mutex_lock(&module_mutex);
    list_for_each_entry(mod, &modules
```

It will look like real kernel C — keywords, brackets, struct patterns, function signatures — all plausible. But the logic will be wrong and it will not compile. This is expected and normal for a model of this size trained on this amount of data.

| Loss Value  | What the output looks like          |
|-------------|-------------------------------------|
| 4.0+        | Random characters                   |
| 2.5 – 3.5   | Recognizable keywords, messy        |
| 1.8 – 2.5   | Plausible structure, wrong logic    |
| Below 1.8   | Coherent style, still not runnable  |

---

## Training Results — CPU Run

Trained on: AMD Ryzen 5 PRO 3500U (CPU only, no GPU)

```
Parameters    : 0.83M
Dataset       : 117,076 characters of Linux kernel C
Vocabulary    : 95 unique characters
Training time : 36 minutes
Best val loss : 2.3924  (reached at step 1400)
Final train   : 0.7820
```

**Full training log:**

```
[    0/3000]   0.0%   train=4.5517   val=4.5657   elapsed=13s     ETA=0s      << best!
[  200/3000]   6.7%   train=2.5428   val=2.8737   elapsed=161s    ETA=2239s   << best!
[  400/3000]  13.3%   train=2.3188   val=2.7972   elapsed=321s    ETA=2083s   << best!
[  600/3000]  20.0%   train=2.0389   val=2.6563   elapsed=479s    ETA=1910s   << best!
[  800/3000]  26.7%   train=1.6357   val=2.6205   elapsed=631s    ETA=1732s   << best!
[ 1000/3000]  33.3%   train=1.4412   val=2.5216   elapsed=782s    ETA=1561s   << best!
[ 1200/3000]  40.0%   train=1.2984   val=2.4317   elapsed=924s    ETA=1384s   << best!
[ 1400/3000]  46.7%   train=1.1895   val=2.3924   elapsed=1069s   ETA=1220s   << best!
[ 1600/3000]  53.3%   train=1.0938   val=2.4409   elapsed=1210s   ETA=1058s
[ 1800/3000]  60.0%   train=1.0317   val=2.4111   elapsed=1352s   ETA=900s
[ 2000/3000]  66.7%   train=0.9991   val=2.4207   elapsed=1487s   ETA=743s
[ 2200/3000]  73.3%   train=0.9201   val=2.4054   elapsed=1615s   ETA=586s
[ 2400/3000]  80.0%   train=0.8548   val=2.4371   elapsed=1750s   ETA=437s
[ 2600/3000]  86.7%   train=0.8339   val=2.4127   elapsed=1888s   ETA=290s
[ 2800/3000]  93.3%   train=0.8137   val=2.4009   elapsed=2021s   ETA=144s
[ 2999/3000] 100.0%   train=0.7820   val=2.4854   elapsed=2159s   ETA=0s

[DONE] Training finished in 2159.5s (36.0 min) | Best val loss: 2.3924
```

---

## Training Results — GPU Run

Trained on: CUDA GPU (Google Colab)

```
Parameters    : 10.82M  (10,819,689 total)
Architecture  : 6 layers × 6 heads × 384 embd dim
Dataset       : 1,896,893 characters
Vocabulary    : 105 unique characters
Train tokens  : 1,707,203
Val tokens    : 189,690
Training time : 55.8 minutes
Best val loss : 1.1177  (reached at step 3750)
Final train   : 0.5069
```

**Full training log:**

```
[    0/5000]   0.0%   train=4.7662   val=4.7716   elapsed=30s     ETA=0s      best!
[  250/5000]   5.0%   train=2.3460   val=2.3397   elapsed=189s    ETA=3579s   best!
[  500/5000]  10.0%   train=1.5358   val=1.6658   elapsed=356s    ETA=3196s   best!
[  750/5000]  15.0%   train=1.1732   val=1.3981   elapsed=523s    ETA=2957s   best!
[ 1000/5000]  20.0%   train=1.0084   val=1.2819   elapsed=690s    ETA=2755s   best!
[ 1250/5000]  25.0%   train=0.9052   val=1.2211   elapsed=857s    ETA=2567s   best!
[ 1500/5000]  30.0%   train=0.8301   val=1.1800   elapsed=1023s   ETA=2385s   best!
[ 1750/5000]  35.0%   train=0.7846   val=1.1740   elapsed=1190s   ETA=2208s   best!
[ 2000/5000]  40.0%   train=0.7444   val=1.1552   elapsed=1356s   ETA=2032s   best!
[ 2250/5000]  45.0%   train=0.7075   val=1.1357   elapsed=1522s   ETA=1859s   best!
[ 2500/5000]  50.0%   train=0.6747   val=1.1300   elapsed=1688s   ETA=1687s   best!
[ 2750/5000]  55.0%   train=0.6502   val=1.1319   elapsed=1854s   ETA=1516s
[ 3000/5000]  60.0%   train=0.6350   val=1.1298   elapsed=2020s   ETA=1345s   best!
[ 3250/5000]  65.0%   train=0.6068   val=1.1198   elapsed=2186s   ETA=1176s   best!
[ 3500/5000]  70.0%   train=0.5871   val=1.1364   elapsed=2352s   ETA=1007s
[ 3750/5000]  75.0%   train=0.5724   val=1.1177   elapsed=2517s   ETA=838s    best!
[ 4000/5000]  80.0%   train=0.5613   val=1.1249   elapsed=2683s   ETA=670s
[ 4250/5000]  85.0%   train=0.5443   val=1.1206   elapsed=2849s   ETA=502s
[ 4500/5000]  90.0%   train=0.5292   val=1.1387   elapsed=3015s   ETA=334s
[ 4750/5000]  95.0%   train=0.5156   val=1.1426   elapsed=3180s   ETA=167s
[ 4999/5000] 100.0%   train=0.5069   val=1.1298   elapsed=3345s   ETA=0s

[DONE] Training finished in 3345.6s (55.8 min) | Best val loss: 1.1177
[SAVE] Best weights saved to: /content/best_model.pt
```

---

## CPU vs GPU — Head to Head

| Metric | CPU Run | GPU Run |
|---|---|---|
| Device | AMD Ryzen 5 PRO 3500U | CUDA GPU |
| Parameters | 0.83M | 10.82M |
| Dataset size | 117,076 chars | 1,896,893 chars |
| Vocabulary | 95 chars | 105 chars |
| Best val loss | 2.3924 | **1.1177** |
| Training time | 36 min | 55.8 min |
| Best step | 1400 / 3000 | 3750 / 5000 |
| Overfitting? | Yes — after step 1400 | Mild — val stays stable |

The GPU run demonstrates exactly what scaling laws predict: **13× more parameters + 16× more data = dramatically better loss** with only 1.5× the training time. A val loss of 1.1177 puts the output firmly in the *coherent style* range, a full tier above the CPU run's *plausible but wrong* range.

---

## Overfitting — What Happened and Why

### CPU Run — Severe Overfitting

Looking at the CPU training log, something important happened after step 1400:

```
Step 1400:  train=1.1895   val=2.3924   ← val loss at its lowest (best)
Step 1600:  train=1.0938   val=2.4409   ← val loss starts rising
Step 3000:  train=0.7820   val=2.4854   ← train keeps falling, val keeps rising
```

This is textbook overfitting. Up to step 1400 the model was learning general patterns. After step 1400, the train loss kept falling but the val loss started rising — the model stopped generalizing and started memorizing specific lines from the training text.

**Why this happened:** The dataset (117K characters) is too small for even a 0.83M parameter model. The model has more capacity than the data can fill.

### GPU Run — Much Better Generalization

The GPU run tells a very different story:

```
Step 2500:  train=0.6747   val=1.1300   ← best at this point
Step 2750:  train=0.6502   val=1.1319   ← tiny uptick, recovers
Step 3750:  train=0.5724   val=1.1177   ← new best, still improving
Step 4999:  train=0.5069   val=1.1298   ← mild divergence at the end
```

The val loss stays tightly clustered between 1.11 and 1.14 for the last 2500 steps. The model never truly overfit because the dataset (1.9M characters) was large enough to keep it generalizing throughout training.

### The Fix — Save Best Weights, Not Final Weights

The current script saves weights at the end of training. But the best weights are always at the lowest val loss checkpoint. Add this to always keep the best version:

```python
# Inside the eval checkpoint block:
if losses['val'] < best_val_loss:
    best_val_loss = losses['val']
    torch.save(model.state_dict(), 'best_model.pt')  # add this line
    improved = " << best!"
```

Then load `best_model.pt` for generation instead of the final weights. This gives noticeably better output quality — especially on the CPU run where the gap between best and final is large.

**Other ways to reduce overfitting:**

- Add more training data — most effective, aim for 1M+ characters
- Increase dropout from 0.2 to 0.3 or 0.4
- Reduce model size further
- Use early stopping — stop training when val loss stops improving for N evals

---

## How Weights Produce Output

After training, the model is frozen. The weights file is just a collection of numbers — 0.83M or 10.82M of them — that encode everything the model learned from your kernel C code.

**The generation loop step by step:**

```
Step 1 — Start with a seed token (zero = start of text)
              ↓
Step 2 — Feed it through all transformer layers
         Each layer does matrix multiplications
         using the saved weight numbers
              ↓
Step 3 — Output is N numbers (one per vocab character)
         Each number = probability of that character being next
         e.g.  's' = 0.34   '{' = 0.21   'i' = 0.18
              ↓
Step 4 — Sample randomly from those probabilities
         Higher probability = more likely to be picked
              ↓
Step 5 — That character becomes the new input
         Go back to Step 2
              ↓
Step 6 — Repeat forever
```

**Why output is different every run:** The sampling step (`torch.multinomial`) picks randomly from the probability distribution. Same weights, different random draws = different output each time. To get reproducible output, add `torch.manual_seed(42)` before generation.

The weights are a compressed snapshot of every pattern seen in the training data — stored as millions of floating point numbers.

---

## Scaling Laws — And Where Model Sits

### What Are Scaling Laws?

Scaling laws describe a predictable relationship between model size, dataset size, compute, and output quality:

> The more parameters, the more data, and the more compute you use — the better the model gets. And this improvement follows a consistent, measurable curve.

The key finding from research (Chinchilla, 2022) is that model size and dataset size must grow together. The optimal ratio is roughly **20 tokens of training data per parameter.**

### The Three Axes of Scaling

```
Parameters (N)  →  How much the model can remember
Data (D)        →  How much it has learned from
Compute (C)     →  Parameters × Data × Training steps
```

All three need to grow together. Improving only one gives diminishing returns.

### Where These Models Sit

```
Model                    Parameters    Data              Best Val Loss   Quality
────────────────────────────────────────────────────────────────────────────────
This project (CPU)       0.83M         117K chars        2.3924          Plausible style
This project (GPU)       10.82M        1.9M chars        1.1177          Coherent style
GPT-2 Small              117M          ~40GB text        —               Coherent English
GPT-2 Large              774M          ~40GB text        —               Strong English
GPT-3                    175B          ~600GB text       —               Near-human text
```

### Data Efficiency — CPU vs GPU Run

```
                    CPU Run         GPU Run
────────────────────────────────────────────────────────
Parameters        : 0.83M           10.82M
Training data     : 117K tokens     1.9M tokens
Optimal data      : 16.6M tokens    216M tokens
Data you have     : 0.7% optimal    0.9% optimal
```

Both runs are well below optimal data — but the GPU run has far more of both, which is exactly why it generalizes much better. The improvement from val loss 2.39 → 1.12 is a direct demonstration of scaling laws in action on your own hardware.

---

## Known Limitations

- **CPU run is slow** — no CUDA GPU means training is slow and larger configs are impractical
- **Both runs are data-starved** — even the GPU run at 1.9M characters is under 1% of what would be optimal for a 10.82M parameter model
- **CPU run overfits after step 1400** — use `best_model.pt`, not the final weights
- **Character-level** — the model learns characters not words or concepts, so it cannot reason about what the code does
- **Output will not compile** — this is a style learner, not a functional code generator
- **No memory between runs** — each generation starts from scratch with no context

---

*Built with PyTorch. Architecture inspired by Andrej Karpathy's makemore / Let's build GPT series.*  
*Trained on Linux kernel source — `kernel/module/core.c` and related files.*
