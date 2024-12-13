import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import *


##########################################
### Gaussian Diffusion Implementation ###
#########################################

# Utility functions

def normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor


def zscore(tensor):
    mean = torch.mean(tensor, dim=0, keepdim=True)
    std = torch.std(tensor, dim=0, keepdim=True)
    z_score_normalized_tensor = (tensor - mean) / std
    return z_score_normalized_tensor


def check_for_nan(tensor_list):
    for i, tensor in enumerate(tensor_list):
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            return True
    return False


class GaussianDiffusion(nn.Module):
    def __init__(self, betas, model):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.betas = torch.tensor(betas, dtype=torch.float32).to(self.device)
        self.alphas = (1.0 - self.betas).to(self.device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], dtype=torch.float32).to(self.device), self.alphas_cumprod[:-1]]).to(self.device)
        
        self.sqrt_alphas = torch.sqrt(self.alphas).to(self.device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(self.device)
        
        self.model = model.to(self.device)
        self.l1 = nn.L1Loss().to(self.device)
        self.mse = nn.MSELoss().to(self.device)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            self.sqrt_alphas_cumprod[t] * x_start +
            self.sqrt_one_minus_alphas_cumprod[t] * noise
        )

    def p_losses(self, gen_list, WSI, t, lambda_regen=1.0, gc=16, noise=None):
        if noise is None:
            noise = [torch.randn_like(gen) for gen in gen_list]

        noisy = [self.q_sample(gen, t, n).to(self.device) for gen, n in zip(gen_list, noise)]
        
        predicted_noise = self.model(x_path=WSI, x_omic=noisy, timestep=t)

        pred_loss = [(self.l1(predicted_noise[i], noise[i]) + self.mse(predicted_noise[i], noise[i])) / 2 for i in range(len(noise))]

        loss = (sum(pred_loss) / len(pred_loss)) / gc
        
        return loss

    def sample_clamp(self, WSI, shape, batch_size):
        x = [torch.randn(size).to(self.device) for size in shape]

        for t in reversed(range(len(self.betas))):
            te = torch.full((batch_size,), t, device=self.device).long()
            predicted_noise = self.model(x_path=WSI, x_omic=x, timestep=te)
            x = self.p_sample_clamp(x, t, predicted_noise)
            x = [torch.clamp(xx, min=0) for xx in x]
            if check_for_nan(x):
                print('t=', t)
                break
        return x

    def p_sample_clamp(self, x, t, predicted_noise):
        beta_t = self.betas[t].to(self.device)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].to(self.device)
        sqrt_alpha_t = self.sqrt_alphas[t].to(self.device)
        one_min_alpha_t = 1.0 - self.alphas_cumprod[t]
        one_min_alpha_t_minus_one = 1.0 - self.alphas_cumprod[t-1]
        beta_t_var = (one_min_alpha_t_minus_one/one_min_alpha_t) * beta_t
        
        mean = [torch.clamp((1 / sqrt_alpha_t) * (xx - (beta_t * p / sqrt_one_minus_alpha_t)), min=0) for xx, p in zip(x, predicted_noise)]

        if t > 0:
            return [m + beta_t_var * torch.randn_like(xx) for m, xx in zip(mean, x)]
        else:
            return mean


