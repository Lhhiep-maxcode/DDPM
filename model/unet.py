import torch
import torch.nn as nn

def get_time_embedding(time_steps, temb_dim):
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
    
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"
    
    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((2 * torch.arange(start=0, end=temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim)))
    t_emb = torch.zeros(time_steps.size(0), temb_dim, device=time_steps.device)
    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    inner = time_steps[:, None].float() / factor
    t_emb[:, 0::2] = torch.sin(inner)
    t_emb[:, 1::2] = torch.cos(inner)
    return t_emb