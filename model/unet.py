import torch
import torch.nn as nn

def get_time_embedding(time_steps, t_emb_dim):
    """Generates a sinusoidal time embedding for a given set of time steps, inspired from Transformer.
    This function converts a 1D tensor of time steps into a higher-dimensional
    embedding using sinusoidal functions. Each timestep will have different embedding. 
    The embedding is constructed such that it alternates between sine and cosine values, following the positional
    encoding formula commonly used in transformer models.
    Args:
        time_steps (torch.Tensor): A 1D tensor of shape (batch_size,) containing
            the time steps for which embeddings are to be generated.
        temb_dim (int): The dimensionality of the output embedding. Must be an
            even number.
    Returns:
        torch.Tensor: A tensor of shape (batch_size, temb_dim) containing the
        sinusoidal time embeddings for the input time steps.
    Raises:
        AssertionError: If `temb_dim` is not divisible by 2."""
    
    assert t_emb_dim % 2 == 0, "time embedding dimension must be divisible by 2"
    
    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((2 * torch.arange(start=0, end=t_emb_dim // 2, dtype=torch.float32, device=time_steps.device) / (t_emb_dim)))
    t_emb = torch.zeros(time_steps.size(0), t_emb_dim, device=time_steps.device)
    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    inner = time_steps[:, None].float() / factor
    t_emb[:, 0::2] = torch.sin(inner)
    t_emb[:, 1::2] = torch.cos(inner)
    return t_emb


class DownBlock(nn.Module):
    """
    Down block include:
    1. n_layers: (Normalization + SiLU + Conv) + Time projection + (Normalization + SiLU + Conv)
    2. n_layers: Normalization + Self-Attention
    3. Down sample
    """
    def __init__(self, in_channels, out_channels, t_emb_dim, down_sample=True, num_heads=4, num_layers=1, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample
        self.res_conv_block_1 = nn.ModuleList([
            nn.Sequential(
                # Normalize for each channel of each instance in a batch
                nn.GroupNorm(num_groups=(in_channels if i == 0 else out_channels), num_channels=(in_channels if i == 0 else out_channels)),
                nn.SiLU(),
                # (batch, c, h, w) --> (batch, c, h, w)
                nn.Conv2d(in_channels=(in_channels if i == 0 else out_channels), out_channels=out_channels,
                          kernel_size=3, stride=1, padding=1),
                nn.Dropout(dropout)
            ) for i in range(num_layers)
        ])
        self.time_projection = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                # (batch, t_emb_dim) --> (batch, out_channels)
                nn.Linear(t_emb_dim, out_channels),
            ) for _ in range(num_layers)
        ])
        self.res_conv_block_2 = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups=out_channels, num_channels=out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                          kernel_size=3, stride=1, padding=1),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        self.attention_norm = nn.ModuleList([
            nn.GroupNorm(num_groups=out_channels, num_channels=out_channels)
            for _ in range(num_layers)
        ])
        self.multihead_attention = nn.ModuleList([
            # Compute attention score for each element of (h=i, w=j) with feature_nums = c
            nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.input_projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.down_sample = nn.Conv2d(out_channels, out_channels, 2, 2) if self.down_sample else nn.Identity()

    def forward(self, x, t_emb):
        # (batch, c_in, h, w)
        out = x
        for i in range(self.num_layers):
            # (batch, c_in, h, w)
            input_res = out
            # (batch, c_in, h, w) --> (batch, c_out, h, w)
            out = self.res_conv_block_1[i](out)
            # (batch, c_out, h, w) + (batch, c_out) --> (batch, c_out, h, w)
            out = out + self.time_projection[i](t_emb)[:, :, None, None] # broadcast
            # (batch, c_out, h, w) --> (batch, c_out, h, w)
            out = self.res_conv_block_2[i](out)
            if i == 0:
                # (batch, c_out, h, w) --> (batch, c_out, h, w)
                out = out + self.input_projection(input_res)
            else:
                out = out + input_res

            batch, channels, h, w = out.shape
            input_attn = out
            # (batch, c_out, h, w) --> (batch, c_out, h, w)
            out = self.attention_norm[i](out)
            # (batch, c_out, h, w) --> (batch, c_out, h * w) --> (batch, h * w, c_out)
            out = out.reshape(batch, channels, h * w).transpose(1, 2)
            # (batch, h * w, c_out) --> (batch, h * w, c_out)
            out, _ = self.multihead_attention[i](out, out, out)
            # (batch, h * w, c_cout) --> (batch, c_out, h * w) --> (batch, c_out, h, w)
            out = out.transpose(1, 2).reshape(batch, channels, h, w)
            out = out + input_attn
        
        # (batch, c_out, h, w) --> (batch, c_out, h / 2, w / 2)
        out = self.down_sample(out)
        return out
    

class MidBlock(nn.Module):
    """
    Down block include:
    1. (Normalization + SiLU + Conv) + Time projection + (Normalization + SiLU + Conv)
    2. n_layers: Self-Attention + (Normalization + SiLU + Conv) + Time projection + (Normalization + SiLU + Conv)
    3. Down sample
    """
    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads=4, num_layers=1, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.t_emb_dim = t_emb_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.res_conv_block_1 = nn.ModuleList([
            nn.Sequential(
                # Normalize for each channel of each instance in a batch
                nn.GroupNorm(num_groups=(self.in_channels if i == 0 else self.out_channels), num_channels=(self.in_channels if i == 0 else self.out_channels)),
                nn.SiLU(),
                # (batch, c, h, w) --> (batch, c, h, w)
                nn.Conv2d(in_channels=(self.in_channels if i == 0 else self.out_channels), out_channels=self.out_channels,
                          kernel_size=3, stride=1, padding=1),
                nn.Dropout(self.dropout)
            ) for i in range(self.num_layers + 1)
        ])
        self.time_projection = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                # (batch, t_emb_dim) --> (batch, out_channels)
                nn.Linear(t_emb_dim, out_channels),
            ) for _ in range(num_layers + 1)
        ])
        self.res_conv_block_2 = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups=self.out_channels, num_channels=self.out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels,
                          kernel_size=3, stride=1, padding=1),
                nn.Dropout(self.dropout)
            ) for _ in range(num_layers + 1)
        ])
        self.attention_norm = nn.ModuleList([
            nn.GroupNorm(num_groups=out_channels, num_channels=out_channels)
            for _ in range(num_layers)
        ])
        self.multihead_attention =  nn.ModuleList([
            # Compute attention score for each element of (h=i, w=j) with feature_nums = c
            nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.input_projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x, t_emb):
        out = x
        # (batch, c_in, h, w)
        input_res = out
        # (batch, c_in, h, w) --> (batch, c_out, h, w)
        out = self.res_conv_block_1[0](out)
        # (batch, c_out, h, w) + (batch, c_out) --> (batch, c_out, h, w)
        out = out + self.time_projection[0](t_emb)[:, :, None, None]
        # (batch, c_out, h, w) --> (batch, c_out, h, w)
        out = self.res_conv_block_2[0](out)
        # (batch, c_out, h, w) + (batch, c_out, h, w) --> (batch, c_out, h, w)
        out = out + self.input_projection(input_res)

        for i in range(self.num_layers):
            batch, channels, h, w = out.shape
            input_attn = out
            # (batch, c_out, h, w) --> (batch, c_out, h, w)
            out = self.attention_norm[i](out)
            # (batch, c_out, h, w) --> (batch, c_out, h * w) --> (batch, h * w, c_out)
            out = out.reshape(batch, channels, h * w).transpose(1, 2)
            # (batch, h * w, c_out) --> (batch, h * w, c_out)
            out, _ = self.multihead_attention[i](out, out, out)
            # (batch, h * w, c_cout) --> (batch, c_out, h * w) --> (batch, c_out, h, w)
            out = out.transpose(1, 2).reshape(batch, channels, h, w)
            # (batch, c_out, h, w)
            out = out + input_attn

            input_res = out
            # (batch, c_out, h, w) --> (batch, c_out, h, w)
            out = self.res_conv_block_1[i + 1](out)
            # (batch, c_out, h, w) + (batch, c_out) --> (batch, c_out, h, w)
            out = out + self.time_projection[i + 1](t_emb)[:, :, None, None]
            # (batch, c_out, h, w) --> (batch, c_out, h, w)
            out = self.res_conv_block_2[i + 1](out)
            # (batch, c_out, h, w)
            out = out + input_res
        
        return out