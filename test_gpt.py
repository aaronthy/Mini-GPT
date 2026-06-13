import torch
from model.gpt import MiniGPT

# settings
vocab_size = 1000
d_model = 64
num_heads = 8
d_ff = 256
num_layers = 3
max_len = 128

# create full model
model = MiniGPT(vocab_size, d_model, num_heads, d_ff, num_layers, max_len)

# count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# fake "I love you"
x = torch.tensor([[1, 204, 89]])

# run through full model
logits = model(x)

print("Input shape:", x.shape)           # [1, 3]
print("Output shape:", logits.shape)     # [1, 3, 1000]
print("Logits for first token:", logits[0][0])  # 1000 scores for "I"

# predict next word
next_token = logits[0, -1, :].argmax()
print("Predicted next token ID:", next_token.item())