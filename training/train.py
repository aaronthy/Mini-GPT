import sys
sys.path.append('.')

import torch
import torch.nn as nn
from torch.optim import Adam
from model.gpt import MiniGPT

# ── settings ──────────────────────────────────────────
d_model    = 64
num_heads  = 8
d_ff       = 256
num_layers = 3
max_len    = 128
dropout    = 0.1
batch_size = 16
epochs     = 3000
lr         = 3e-4

# ── load and tokenize data ─────────────────────────────
with open('data/input.txt', 'r') as f:
    text = f.read()

# build character level vocabulary
chars = sorted(set(text))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size} unique characters")

# char to index and index to char lookups
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# encode full text into numbers
data = torch.tensor([char_to_idx[ch] for ch in text], dtype=torch.long)
print(f"Total tokens: {len(data):,}")

# ── create training batches ────────────────────────────
def get_batch(data, batch_size, max_len):
    ix = torch.randint(len(data) - max_len, (batch_size,))
    x = torch.stack([data[i:i+max_len] for i in ix])
    y = torch.stack([data[i+1:i+max_len+1] for i in ix])
    return x, y

# ── build model ────────────────────────────────────────
model = MiniGPT(vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout)
optimizer = Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── training loop ──────────────────────────────────────
model.train()
for epoch in range(epochs):
    x, y = get_batch(data, batch_size, max_len)

    # forward pass
    logits = model(x)

    # reshape for loss function
    loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))

    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}  Loss: {loss.item():.4f}")

print("Training done!")

# ── save model ─────────────────────────────────────────
torch.save(model.state_dict(), 'model/minigpt_weights.pth')
print("Model saved to model/minigpt_weights.pth")