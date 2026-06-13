# Mini GPT — Transformer Built From Scratch

A minimal GPT implementation built from scratch in PyTorch, trained on the Tiny Shakespeare dataset. Every component is hand-coded to demonstrate a deep understanding of transformer architecture.

## What I Built

This project implements the full transformer pipeline from scratch:

- **Token + Positional Embeddings** — converts raw text into positional-aware vectors
- **Multi-Head Self-Attention** — scaled dot-product attention with 8 parallel heads
- **Feed-Forward Network** — per-token processing with GELU activation
- **GPT Assembly** — stacked transformer blocks with residual connections and layer norm
- **Character-level Training** — trained on 1.1M tokens of Shakespeare text
- **Streamlit UI** — interactive app with text generation and attention visualization

## Architecture

Input text
↓
Token Embedding + Positional Encoding
↓
Transformer Block × 3
└── Multi-Head Attention (8 heads)
└── Feed-Forward Network (d_ff=256)
└── Residual + Layer Norm
↓
Output Head → next character prediction

## Model Config

| Parameter | Value |
|---|---|
| d_model | 64 |
| num_heads | 8 |
| num_layers | 3 |
| d_ff | 256 |
| max_len | 128 |
| vocab_size | 65 |
| Parameters | ~158k |

## Project Structure

mini-gpt/
├── model/
│   ├── embeddings.py     # token + positional encoding
│   ├── attention.py      # multi-head self-attention
│   ├── transformer.py    # transformer block + FFN
│   └── gpt.py            # full GPT model
├── training/
│   ├── dataset.py        # data loading + tokenization
│   └── train.py          # training loop
├── app/
│   └── streamlit_app.py  # interactive UI
├── data/
│   └── input.txt         # tiny shakespeare dataset
└── README.md

## How to Run

**Install dependencies**
```bash
pip install torch streamlit matplotlib seaborn
```

**Train the model**
```bash
python training/train.py
```

**Run the app**
```bash
streamlit run app/streamlit_app.py
```
## Screenshot
![Mini GPT](images/minigpt1.png)
![Mini GPT](images/minigpt2.png)
![Mini GPT](images/minigpt3.png)

## What I Learned

- How scaled dot-product attention works mathematically (Q, K, V matrices)
- Why positional encoding is needed in transformers
- How residual connections and layer norm stabilize training
- How cross-entropy loss and backpropagation tune 158k parameters
- Why multi-head attention captures different linguistic relationships simultaneously

## Stack

Python · PyTorch · Streamlit · Matplotlib · Seaborn

## References

- Attention Is All You Need (Vaswani et al., 2017)
- Andrej Karpathy's nanoGPT



