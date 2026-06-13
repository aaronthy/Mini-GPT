import torch
import torch.nn as nn
import math


def scaled_dot_product_attention(Q, K, V, mask=None):
    # step 1 — get the vector size
    d_k = Q.size(-1)

    # step 2 — compute scores: how much each word attends to every other word
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # step 3 — apply mask if provided (stops words peeking at future words)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # step 4 — softmax turns scores into probabilities (all add up to 1)
    weights = torch.softmax(scores, dim=-1)

    # step 5 — multiply weights by values to get final output
    output = torch.matmul(weights, V)

    return output, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # linear layers to project input into Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # final linear layer after combining all heads
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch_size = x.size(0)
        # reshape into [batch, num_heads, seq_len, d_k]
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, x, mask=None):
        batch_size = x.size(0)

        # step 1 — project x into Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # step 2 — split into multiple heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # step 3 — run attention on each head
        output, weights = scaled_dot_product_attention(Q, K, V, mask)

        # step 4 — combine all heads back together
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.num_heads * self.d_k)

        # step 5 — final linear projection
        output = self.W_o(output)

        return output, weights

