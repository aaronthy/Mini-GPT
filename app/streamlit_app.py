import sys
sys.path.append('.')
import torch
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from model.gpt import MiniGPT

d_model    = 64
num_heads  = 8
d_ff       = 256
num_layers = 3
max_len    = 128
dropout    = 0.0

@st.cache_resource
def load_model():
    with open('data/input.txt', 'r') as f:
        text = f.read()
    chars = sorted(set(text))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    model = MiniGPT(vocab_size, d_model, num_heads,
                    d_ff, num_layers, max_len, dropout)
    model.load_state_dict(torch.load('model/minigpt_weights.pth',
                          map_location='cpu'))
    model.eval()
    return model, char_to_idx, idx_to_char

model, char_to_idx, idx_to_char = load_model()

def generate(prompt, max_new_tokens=200, temperature=0.8):
    tokens = [char_to_idx.get(ch, 0) for ch in prompt]
    x = torch.tensor([tokens])
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(x[:, -max_len:])
            logits = logits[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_token.unsqueeze(0)], dim=1)
    return ''.join([idx_to_char[i.item()] for i in x[0]])

st.title("Mini GPT")
st.caption("Built from scratch - embeddings, attention, transformer, trained on Shakespeare")
st.divider()

st.sidebar.header("Settings")
temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 0.8, 0.1)
max_tokens = st.sidebar.slider("Max tokens", 50, 500, 200, 50)

mode = st.selectbox("Mode", ["Generate", "Chat", "Summarize", "Translate"])
prompt = st.text_area("Prompt", value="To be or not to be")

if st.button("Generate", type="primary"):
    with st.spinner("Generating..."):
        if mode == "Generate":
            output = generate(prompt, max_tokens, temperature)
            st.subheader("Output")
            st.write(output)
        elif mode == "Chat":
            output = generate(f"Q: {prompt}\nA:", max_tokens, temperature)
            st.subheader("Response")
            st.write(output)
        elif mode == "Summarize":
            output = generate(f"Summary of: {prompt}\n", max_tokens, temperature)
            st.subheader("Summary")
            st.write(output)
        elif mode == "Translate":
            output = generate(f"Translation: {prompt}\n", max_tokens, temperature)
            st.subheader("Translation")
            st.write(output)

st.divider()
st.subheader("Attention weights visualization")
st.caption("See which characters attend to which")

vis_prompt = st.text_input("Visualize attention for:", value="To be")

if st.button("Show attention"):
    tokens = [char_to_idx.get(ch, 0) for ch in vis_prompt]
    x = torch.tensor([tokens])
    with torch.no_grad():
        embed_out = model.embeddings(x)
        _, weights = model.blocks[0](embed_out)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    fig.suptitle(f'Attention heads for "{vis_prompt}"')
    chars_list = list(vis_prompt)
    for head in range(num_heads):
        ax = axes[head // 4][head % 4]
        w = weights[0, head].numpy()
        sns.heatmap(w, ax=ax, cmap='Blues',
                    xticklabels=chars_list,
                    yticklabels=chars_list,
                    cbar=False)
        ax.set_title(f'Head {head+1}', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)