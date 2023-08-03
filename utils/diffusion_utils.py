import math
from typing import Tuple

import numpy as np
import torch


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)



class Diffusion:
    def __init__(self, noise_steps=50, beta_start=1e-4, beta_end=0.02, 
                 device="cuda", time=25,
                 n_joints=22):

        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.time = time
        self.joints = n_joints
        self.device = device
        self.beta = self.schedule_noise()
        self.alpha = (1. - self.beta)
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
    
    def prepare_noise_schedule(self) -> torch.Tensor:
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps, device=self.device)
    
    
    def schedule_noise(self):
        betas = betas_for_alpha_bar(
            self.noise_steps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
        
        return torch.tensor(betas, dtype=torch.float32, device=self.device)
            

    def noise_images(self, x:torch.Tensor, t:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = t.get_device()
        alpha_hat = self.alpha_hat.to(device)
        sqrt_alpha_hat = torch.sqrt(alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x, device=device)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
    

    def noise_graph(self, x:torch.Tensor, t:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = t.get_device()
        alpha_hat = self.alpha_hat.to(device)
        sqrt_alpha_hat = torch.sqrt(alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x, device=device)
        return sqrt_alpha_hat *x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
    
    
    def noise_latent(self, x:torch.Tensor, t:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = t.get_device()
        alpha_hat = self.alpha_hat.to(device)
        sqrt_alpha_hat = torch.sqrt(alpha_hat[t])[:,None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t])[:, None]
        Ɛ = torch.randn_like(x, device=device)
        return sqrt_alpha_hat *x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ 


    def sample_timesteps(self, n:int) -> torch.Tensor:
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
