import torch
import torch.nn as nn
from model.embeddings import Embeddings
from model.transformer import TransformerBlock


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, 
                 num_layers, max_len, dropout=0.1):
        super().__init__()

        # front door — text to vectors
        self.embeddings = Embeddings(vocab_size, d_model, max_len, dropout)

        # stack N transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # final layer norm
        self.norm = nn.LayerNorm(d_model)

        # output head — vectors to word probabilities
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        # step 1 — embeddings
        x = self.embeddings(x)

        # step 2 — pass through each transformer block
        for block in self.blocks:
            x, weights = block(x, mask)

        # step 3 — final layer norm
        x = self.norm(x)

        # step 4 — predict next word probabilities
        logits = self.output_head(x)

        return logits