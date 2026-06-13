import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_length):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.positional_embedding_layer = nn.Embedding(context_length, embedding_dim)

    def forward(self, x):
        token_embedding = self.embedding_layer(x)
        position_ids = torch.arange(0, x.size(1), device=x.device)
        pos_embedding = self.positional_embedding_layer(position_ids)
        
        input_embedding = token_embedding + pos_embedding
        return input_embedding
    
class MultiheadAttention(nn.Module):
    def __init__(self, num_heads, embedding_dim, context_length, dropout):
        super().__init__()
        assert (embedding_dim % num_heads == 0), "embedding_dim must be divisible by num_heads"
        
        self.q_layer = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=False)
        self.k_layer = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=False)
        self.v_layer = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=False)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)  # Linear layer to combine head outputs
        self.register_buffer("mask", torch.tril(torch.ones(context_length, context_length)))

    def forward(self, x):
        B, T, C = x.shape 

        Q = self.q_layer(x)
        K = self.k_layer(x)
        V = self.v_layer(x)

        queries = Q.view(B, T, self.num_heads, self.head_dim)
        keys = K.view(B, T, self.num_heads, self.head_dim)
        values = V.view(B, T, self.num_heads, self.head_dim)
        
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        

        K_transpose = keys.transpose(2, 3)
        attention_score = queries @ K_transpose
        mask = self.mask[:T, :T]
        masked_attn_scores = attention_score.masked_fill(mask==0, float('-inf'))
        scaled_attn_scores = masked_attn_scores / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scaled_attn_scores, dim=3)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        context_vec = context_vec.transpose(1, 2)
        context_vec = context_vec.contiguous().view(B, T, self.num_heads * self.head_dim)
        context_vector = self.out_proj(context_vec)


        return context_vector

class FeedForward(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )
    
    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, num_heads, context_length, dropout, embedding_dim):
        super().__init__()

        self.att = MultiheadAttention(num_heads=num_heads, embedding_dim=embedding_dim, context_length=context_length, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.attn_dropout = nn.Dropout(dropout)


        self.ff = FeedForward(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.ff_dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        shortcut = x
        x = self.layer_norm1(x)
        x = self.att(x)
        x = self.attn_dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.layer_norm2(x)
        x = self.ff(x)
        x = self.ff_dropout(x)
        x = x + shortcut

        return x

class GPTModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_length, dropout, num_heads, n_layers):
        super().__init__()

        self.input_embedding = InputEmbedding(vocab_size, embedding_dim, context_length)
        self.dropout = nn.Dropout(dropout)
        self.trans_block = nn.Sequential(*[TransformerBlock(num_heads, context_length, dropout, embedding_dim) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, x):
        
        x = self.input_embedding(x)
        x = self.dropout(x)
        x = self.trans_block(x)
        x = self.final_norm(x)
        logits = self.output_layer(x)

        return logits