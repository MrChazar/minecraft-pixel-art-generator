import math
import torch.nn as nn
import torch
import torch.nn.functional as F



class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self, x):
        seq_len = x.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x, attn_mask=causal_mask)
        x = x + attn_output
        x = self.norm1(x)

        # Feed-forward network with residual connection
        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.norm2(x)
        return x

class Encoder(nn.Module):
    def __init__(self, hidden_size=128, num_layers=3, num_heads=4):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Transformer(nn.Module):
    def __init__(self, num_emb, hidden_size=256, num_layers=3, num_heads=8,
                 cond_type_dim=8, cond_colors_dim=8, latent_channels=256):
        super(Transformer, self).__init__()

        # Create an embedding layer for tokens
        self.embedding = nn.Embedding(num_emb, hidden_size)

        # Initialize sinusoidal positional embeddings
        self.pos_emb = SinusoidalPosEmb(hidden_size)

        # Condition projection and expansion layers
        self.block_proj = nn.Linear(1, latent_channels)
        self.type_proj = nn.Linear(cond_type_dim, latent_channels)
        self.color_proj = nn.Linear(cond_colors_dim, latent_channels)

        self.block_expand = nn.Linear(latent_channels, hidden_size)
        self.type_expand = nn.Linear(latent_channels, hidden_size)
        self.color_expand = nn.Linear(latent_channels, hidden_size)


        self.cond_norm = nn.LayerNorm(hidden_size)

        # Create an encoder with specified parameters
        self.encoder = Encoder(hidden_size=hidden_size, num_layers=num_layers,
                               num_heads=num_heads)

        # Define a linear layer for output prediction
        self.fc_out = nn.Linear(hidden_size, num_emb)

        # Num of head might need some changing
        self.block_attention = nn.MultiheadAttention(hidden_size, max(1,num_heads//4), batch_first=True,dropout=0.1)
        self.type_attention = nn.MultiheadAttention(hidden_size, max(1,num_heads//4), batch_first=True,dropout=0.1)
        self.color_attention = nn.MultiheadAttention(hidden_size, max(1,num_heads//2), batch_first=True,dropout=0.1)

        # Define weights
        self.block_weight = nn.Parameter(torch.tensor(1.0))
        self.type_weight = nn.Parameter(torch.tensor(1.0))
        self.color_weight = nn.Parameter(torch.tensor(0.5))

    def embed(self, input_seq, is_block=None, type_=None, colors=None):
        is_block = is_block.unsqueeze(1)
        # Embed the input sequence
        input_embs = self.embedding(input_seq)
        bs, l, h = input_embs.shape

        # Add positional embeddings to the input embeddings
        seq_indx = torch.arange(l, device=input_seq.device)
        pos_emb = self.pos_emb(seq_indx).reshape(1, l, h).expand(bs, l, h)
        embs = input_embs + pos_emb

        # Add condition information if provided
        if is_block is not None and type_ is not None and colors is not None:
            # Process separately
            block_cond = self.block_proj(is_block)
            type_cond = self.type_proj(type_)
            color_cond = self.color_proj(colors)

            # Expand
            block_cond = self.block_expand(block_cond).unsqueeze(1)
            type_cond = self.type_expand(type_cond).unsqueeze(1)
            color_cond = self.color_expand(color_cond).unsqueeze(1)

            # Apply attention
            block_attn, _ = self.block_attention(embs, block_cond, block_cond)
            type_attn, _ = self.type_attention(embs, type_cond, type_cond)
            color_attn, _ = self.color_attention(embs, color_cond, color_cond)

            # Combine
            embs = embs + self.block_weight * block_attn + self.type_weight * type_attn + self.color_weight * color_attn
            embs = self.cond_norm(embs)

        return embs

    def encode(self, input_seq, is_block=None, type_=None, colors=None):
        # Embed the input sequence with conditions
        embs = self.embed(input_seq, is_block, type_, colors)

        # Encode the sequence
        embs_out = self.encoder(embs)
        return embs_out

    def forward(self, input_seq, is_block=None, type_=None, colors=None):
        # Encode the input sequence with conditions
        encoded_seq = self.encode(input_seq, is_block, type_, colors)

        return self.fc_out(encoded_seq)
    


