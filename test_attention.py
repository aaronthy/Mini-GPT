import torch
from model.embeddings import Embeddings
from model.attention import MultiHeadAttention

# same settings as before
vocab_size = 1000
d_model = 64
num_heads = 8
max_len = 128

# create embedding and attention layers
embed = Embeddings(vocab_size, d_model, max_len)
attention = MultiHeadAttention(d_model, num_heads)

# fake "I love you" tokenized
x = torch.tensor([[1, 204, 89]])

# run through embeddings first
x = embed(x)
print("After embedding shape:", x.shape)    # [1, 3, 64]

# run through attention
output, weights = attention(x)
print("After attention shape:", output.shape)   # [1, 3, 64]
print("Attention weights shape:", weights.shape) # [1, 8, 3, 3]
print("Attention weights:", weights[0][0])       # 3x3 score table for head 1