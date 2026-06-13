import torch
from model.embeddings import Embeddings

# small numbers for testing
vocab_size = 1000
d_model = 64
max_len = 128

# create the embedding layer
embed = Embeddings(vocab_size, d_model, max_len)

# fake "I love you" tokenized to [1, 204, 89]
x = torch.tensor([[1, 204, 89]])

# run it
output = embed(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("First token vector:", output[0][0])