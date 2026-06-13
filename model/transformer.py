import torch
import torch.nn as nn
from model.attention import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),   # expand
            nn.GELU(),                   # non-linearity
            nn.Dropout(dropout),         # regularization
            nn.Linear(d_ff, d_model),   # compress back
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # step 1 — attention + residual connection
        attention_output, weights = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attention_output))

        # step 2 — FFN + residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x, weights