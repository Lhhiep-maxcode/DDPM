import torch

class LinearNoiseScheduler:
    def __init__(self, beta_start=0.0001, beta_end=0.02, timesteps=1000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=self.device)
        self.alphas = 1 - self.betas
        self.cumulative_alphas = torch.cumprod(self.alphas, 0)
        self.sqrt_cumulative_alphas = torch.sqrt(self.cumulative_alphas)
        self.sqrt_one_minus_cumulative_alphas = torch.sqrt(1 - self.cumulative_alphas)

    def forward(self, x0, noise, timestep):
        """
        Computes the noisy version of the input `x0` at a given timestep during the 
        diffusion process.

        Parameters:
            x0 (torch.Tensor): The original input tensor of shape (batch_size, c, h, w).
            noise (torch.Tensor): The noise tensor of the same shape as `x0` (batch_size, c, h, w).
            timestep (int or torch.Tensor): The current timestep in the diffusion 
                process. If an integer is provided, it will be converted to a tensor 
                with the same batch size as `x0`.

        Returns:
            torch.Tensor: The noisy version of `x0` at the given timestep, computed 
            as:
                x_t = x0 * sqrt(cum_alpha_t) + sqrt(1 - cum_alpha_t) * noise
        """
        batch_size = x0.size(0)
        if isinstance(timestep, int):
            timestep = torch.tensor([timestep] * batch_size, device=x0.device)

        sqrt_cumulative_alpha = self.sqrt_cumulative_alphas[timestep].view(batch_size, 1, 1, 1)
        one_minus_sqrt_cumulative_alpha = self.sqrt_one_minus_cumulative_alphas[timestep].view(batch_size, 1, 1, 1)
        
        return x0 * sqrt_cumulative_alpha + one_minus_sqrt_cumulative_alpha * noise
    
    def reverse(self, xt, noise_pred, timestep):
        """
        Reverses the diffusion process by predicting the previous state `x_{t-1}` 
        from the current state `x_t`, predicted noise, and the current timestep.
        Args:
            xt (torch.Tensor): The current state tensor of shape (batch, c, h, w).
            noise_pred (torch.Tensor): The predicted noise tensor of shape (batch, c, h, w).
            timestep (int or torch.Tensor): The current timestep in the diffusion process.
                If an integer is provided, it will be converted to a tensor 
                with the same batch size as `xt`.
        Returns:
            torch.Tensor: The predicted previous state `x_{t-1}` of shape (batch, c, h, w).
        Notes:
            - If the timestep is 0, the function directly returns the mean of the 
              reverse process without adding noise.
            - For timesteps greater than 0, the function adds noise sampled from 
              a Gaussian distribution scaled by the computed variance.
        """
        batch_size = xt.size(0)
        if isinstance(timestep, int):
            timestep = torch.tensor([timestep] * batch_size, device=xt.device)

        sqrt_alpha = torch.sqrt(self.alphas[timestep]).view(batch_size, 1, 1, 1)
        beta = self.betas[timestep].view(batch_size, 1, 1, 1)
        sqrt_one_minus_cumulative_alpha = self.sqrt_one_minus_cumulative_alphas[timestep].view(batch_size, 1, 1, 1)

        mean = (xt - beta * noise_pred / sqrt_one_minus_cumulative_alpha) / sqrt_alpha
        variance = beta * (1 - self.cumulative_alphas[timestep - 1].view(batch_size, 1, 1, 1)) / (1 - self.cumulative_alphas[timestep].view(batch_size, 1, 1, 1))
        std = torch.sqrt(variance)
        z = torch.randn(xt.shape, device=xt.device)
        return mean + std * z * (timestep > 1).int().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)