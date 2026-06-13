import sys
sys.path.append('.')
import torch
from model.gpt import MiniGPT

# same settings as train.py
d_model    = 64
num_heads  = 8
d_ff       = 256
num_layers = 3
max_len    = 128
dropout    = 0.0

# load vocabulary
with open('data/input.txt', 'r') as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# load trained model
model = MiniGPT(vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout)
model.load_state_dict(torch.load('model/minigpt_weights.pth'))
model.eval()

# generate text
def generate(prompt, max_new_tokens=200, temperature=0.8):
    tokens = [char_to_idx[ch] for ch in prompt]
    x = torch.tensor([tokens])

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(x[:, -max_len:])
            # divide by temperature then sample
            logits = logits[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_token.unsqueeze(0)], dim=1)

    return ''.join([idx_to_char[i.item()] for i in x[0]])

print(generate("To be or not", temperature=0.8))