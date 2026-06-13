import torch
from model.embeddings import Embeddings
from model.transformer import TransformerBlock

# settings
vocab_size = 1000
d_model = 64
num_heads = 8
d_ff = 256      # 4x d_model
max_len = 128

# create layers
embed = Embeddings(vocab_size, d_model, max_len)
transformer_block = TransformerBlock(d_model, num_heads, d_ff)

# fake "I love you"
x = torch.tensor([[1, 204, 89]])

# step 1 — embed
x = embed(x)
print("After embedding shape:", x.shape)       # [1, 3, 64]

# step 2 — transformer block
output, weights = transformer_block(x)
print("After transformer shape:", output.shape) # [1, 3, 64]
print("Attention weights shape:", weights.shape) # [1, 8, 3, 3]
print("Output first token:", output[0][0])       # 64 numbers