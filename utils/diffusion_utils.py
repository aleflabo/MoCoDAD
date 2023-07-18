import torch
import numpy as np
import math
from tqdm import tqdm

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

class Diffusion:
    #def __init__(self, noise_steps=50, beta_start=1e-4, beta_end=0.02, img_size=64, device="cuda"):
    def __init__(self, noise_steps=50, beta_start=1e-4, beta_end=0.02, 
                 device="cuda", time=25,
                 n_joints=22, channels=2):

        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        self.channel = channels
        self.time = time
        self.joints = n_joints
               
        self.device = device

        #self.beta = self.prepare_noise_schedule().to(device)
        self.beta = self.my_schedule_noise()
        
        
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def my_schedule_noise(self):
        betas = betas_for_alpha_bar(
            self.noise_steps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
        
        return torch.tensor(betas, dtype=torch.float32)
            
        

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None] # was None, None, None
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None] # was None, None, None
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
    
    def noise_graph(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None].to(t.get_device()) # was None, None, None
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None].to(t.get_device()) # was None, None, None
        Ɛ = (torch.randn_like(x)).to(t.get_device())
        
        return sqrt_alpha_hat *x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
        
        
        
        

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
        

    #@torch.inference_mode()
    def sample(self, model, n, pst, cfg_scale=0):
        #logging.info(f"Sampling {n} new images....")
        #print(f"Sampling {n} new images....")
        #model.eval()
        #with torch.no_grad():
        x = torch.randn((n, self.channel, self.time, self.joints))
        x = x
        
        for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
            t = (torch.ones(n) * i).long()
            predicted_noise, _ = model(x, t, pst)
            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            #if self.version=='out_all':
            #    predicted_noise = predicted_noise[:,:,-self.time:]
            #predicted_noise = predicted_noise.permute(0,2,1,3)
            
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        #model.train()
        #x = (x.clamp(-1, 1) + 1) / 2
        #x = (x * 255).type(torch.uint8)
        return x

