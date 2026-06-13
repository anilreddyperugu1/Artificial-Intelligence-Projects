# 🤖 GPT from Scratch using PyTorch

A complete implementation of a GPT-style Language Model built entirely from scratch using PyTorch. This project covers every core component of the Transformer architecture, including token embeddings, positional embeddings, multi-head self-attention, feed-forward networks, transformer blocks, training, validation, and autoregressive text generation.

The model was trained on *The Verdict* dataset and successfully learned the writing style, sentence structure, punctuation patterns, and contextual relationships within the text.

---

# 📑 Index

1. 📖 [Project Overview](#project-overview)
2. 🎯 [Problem Statement](#problem-statement)
3. 🧠 [Key Features & Terminologies](#key-features--terminologies)
4. 🛠️ [Libraries Used](#libraries-used)
5. ⚙️ [Workflow Summary](#workflow-summary)
6. 🏗️ [Model Architecture](#model-architecture)
7. 🚀 [Training Process](#training-process)
8. 📊 [Results & Evaluation](#results--evaluation)
9. ✨ [Text Generation Examples](#text-generation-examples)
10. 🧠 [Key Takeaways](#key-takeaways)
11. 🔮 [Future Improvements](#future-improvement)

---

# 📖 Project Overview

Large Language Models (LLMs) such as GPT are built upon the Transformer architecture. The goal of this project was to understand and implement every major component of GPT from scratch rather than relying on pre-built libraries.

This project demonstrates:

- Tokenization
- Input Embeddings
- Positional Embeddings
- Multi-Head Self-Attention
- Feed Forward Networks
- Transformer Blocks
- GPT Architecture
- Training Loop
- Validation Pipeline
- Autoregressive Text Generation
- Temperature Sampling

---

# 🎯 Problem Statement

Build a GPT-style language model capable of predicting the next token in a sequence using only PyTorch and fundamental Transformer concepts.

The model should:

- Learn contextual relationships between tokens.
- Generate coherent text autoregressively.
- Implement attention mechanisms from scratch.
- Validate performance on unseen validation data.
- Support inference and text generation from custom prompts.

---

# 🧠 Key Features & Terminologies

## 🔹 Token Embedding

Converts token IDs into dense vector representations.

```text
Token ID → Embedding Vector
```

---

## 🔹 Positional Embedding

Adds positional information to embeddings so the model understands token order.

```text
Token Embedding + Position Embedding
```

---

## 🔹 Multi-Head Self Attention

Allows tokens to attend to other relevant tokens in the sequence.

Implemented:

- Query (Q)
- Key (K)
- Value (V)
- Scaled Dot Product Attention
- Causal Masking
- Multi-Head Attention

---

## 🔹 Feed Forward Network (FFN)

Two-layer neural network applied after attention.

```python
Linear
↓
GELU
↓
Linear
```

---

## 🔹 Transformer Block

Contains:

- Layer Normalization
- Multi-Head Attention
- Feed Forward Network
- Residual Connections
- Dropout

---

## 🔹 GPT Architecture

Stack of Transformer Blocks followed by:

- Final LayerNorm
- Output Projection Layer

---

## 🔹 Cross Entropy Loss

Measures how accurately the model predicts the next token.

---

## 🔹 Perplexity

Standard evaluation metric for language models.

Lower perplexity indicates better predictions.

---

# ⚙️ Workflow Summary

## Step 1: Read Dataset

```text
Text File
↓
Raw Text
```

---

## Step 2: Tokenization

```text
Raw Text
↓
Token IDs
```

Tokenizer Used:

```text
cl100k_base
```

---

## Step 3: Dataset Creation

Created input-target pairs using a sliding context window.

```text
Input Sequence
↓
Target Sequence (Shifted by 1)
```

---

## Step 4: Embedding Layer

Implemented:

- Token Embedding
- Positional Embedding

---

## Step 5: Multi-Head Self Attention

Implemented:

- Query Layer
- Key Layer
- Value Layer
- Attention Scores
- Causal Mask
- Attention Weights
- Context Vector

---

## Step 6: Feed Forward Network

```text
Embedding Dim
↓
4 × Embedding Dim
↓
Embedding Dim
```

---

## Step 7: Transformer Block

```text
LayerNorm
↓
Multi-Head Attention
↓
Residual Connection
↓
LayerNorm
↓
Feed Forward
↓
Residual Connection
```

---

## Step 8: GPT Model

```text
Input Tokens
↓
Embeddings
↓
Transformer Blocks
↓
Final LayerNorm
↓
Output Projection
↓
Logits
```

---

## Step 9: Training

Optimizer:

```text
AdamW
```

Parameters:

```text
Learning Rate : 3e-4
Betas         : (0.9, 0.95)
Weight Decay  : 0.01
```

Gradient clipping was used for stable training.

---

## Step 10: Validation

Implemented:

- Validation Dataset
- Validation Dataloader
- Validation Loss
- Perplexity

---

## Step 11: Text Generation

Implemented autoregressive generation:

```text
Prompt
↓
Predict Next Token
↓
Append Token
↓
Repeat
```

Supports:

- Greedy Decoding
- Temperature Sampling

---

# 🏗️ Model Architecture

## Hyperparameters

```python
context_length = 256
vocab_size = 100256
embedding_dim = 128
num_heads = 4
n_layers = 5
dropout = 0.2
batch_size = 4
```

---

## Architecture Overview

```text
Input Tokens
      ↓
Token Embedding
      ↓
Position Embedding
      ↓
Dropout
      ↓
5 Transformer Blocks
      ↓
LayerNorm
      ↓
Linear Projection
      ↓
Vocabulary Logits
```

---

# 📊 Training & Validation

## Training Results

```text
Initial Loss : 11.66
Final Loss   : ~0.06
```

---

## Validation Results

```text
Validation Loss : 0.0305
Perplexity      : 1.03
```

---

# ✨ Text Generation Examples

### Prompt

```text
Once upon a time
```

### Generated Output

```text
Once upon a time, he managed to live without that sketch of the florid vista...
```

---

### Prompt

```text
The old man walked into the room and
```

### Generated Output

```text
The old man walked into the room and sensation was square and brown and leathery...
```

---

# 📈 Model Evaluation

### What Worked Well

✅ Learned sentence structure

✅ Learned punctuation patterns

✅ Learned contextual relationships

✅ Generated coherent text

✅ Successfully reproduced writing style

---

### Observations

- Limited world knowledge due to training on a single text corpus.
- Strong style imitation capabilities.

---

# 🚀 Key Takeaways

Through this project, I gained hands-on experience with:

- Transformer Architecture
- Self-Attention Mechanism
- Multi-Head Attention
- GPT Design
- Language Model Training
- Validation & Perplexity
- Autoregressive Text Generation
- Temperature Sampling
- PyTorch Model Development

---

# 🔮 Future Improvements

- Top-K Sampling
- Top-P (Nucleus) Sampling
- Learning Rate Scheduling
- Larger Training Corpus
- Checkpointing Support
- Fine-Tuning Pipeline
- Streamlit/Gradio Deployment
- Model Quantization
- Mixed Precision Training

---

# 🙌 Conclusion

This project successfully demonstrates the end-to-end implementation of a GPT-style Language Model from scratch using PyTorch. Every major Transformer component was implemented manually to gain a deep understanding of how modern Large Language Models work internally.

The final model was capable of learning language patterns from text data and generating coherent continuations through autoregressive decoding.

---

## 📇 Author

Anil Reddy Perugu💝

📧 Email: peruguanilreddy6@gmail.com

📍 Feel free to reach out for queries, suggestions, or collaborations!
