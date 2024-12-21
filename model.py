import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

def cosine_beta_schedule(timesteps, s=0.008):
    x = torch.linspace(0, timesteps, timesteps+1)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5)**2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = torch.clamp(1 - (alphas_cumprod[1:] / alphas_cumprod[:-1]), 1e-8, 0.999)
    return betas

def linear_beta_schedule(timesteps):
    return torch.linspace(1e-4, 0.02, timesteps)


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def generate_multi_resolution_noise(x, num_scales=3, weights=None):
        B, C, H, W = x.shape
        if weights is None:
            weights = [1.0 / num_scales] * num_scales  # Equal weights by default
        
        noise = torch.zeros_like(x)
        for i in range(num_scales):
            scale_factor = 2 ** i  # Adjust scale by powers of 2
            scaled_noise = torch.randn(B, C, H // scale_factor, W // scale_factor, device=x.device)
            scaled_noise = F.interpolate(scaled_noise, size=(H, W), mode='bilinear', align_corners=False)
            noise += weights[i] * scaled_noise
        return noise

def anneal_noise(noise, t, timesteps):
    strength = 1.0 - (t / timesteps).view(-1, 1, 1, 1)
    return strength * noise

def save_checkpoint(model, optimizer, epoch, loss, filepath="checkpoint.pth"):
    checkpoint = {
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'epoch' : epoch,
        'loss' : loss,
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath, model, optimizer):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded: epoch {epoch}, loss: {loss:.4f}")
    return epoch, loss


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size, img_size):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.Sequential(nn.LayerNorm(dim), nn.LayerNorm(dim))
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)),
            nn.ReLU(),  # Changed to ReLU
            nn.Linear(int(dim*mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_attn = self.norm1(x)
        x_attn, _ = self.attn(x_attn, x_attn, x_attn, need_weights=False)
        x = x + x_attn
        x_mlp = self.norm2(x)
        x_mlp = self.mlp(x_mlp)
        x = x + x_mlp
        return x

class DiffusionTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, img_size=(64,64), patch_size=8, embed_dim=256, depth=4, num_heads=8):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size, img_size)
        # Initialize pos_embed to zero instead of random
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size[0]//patch_size)*(img_size[1]//patch_size), embed_dim))
        
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(0.1)
        )
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # Remove Tanh from output
        self.fc_out = nn.Linear(embed_dim, patch_size * patch_size * out_channels)

    def forward(self, x, t_embed):
        B, C, H, W = x.shape
        x_patches = self.patch_embed(x)
        # Add position embeddings
        x_patches = x_patches + self.pos_embed
        # Add time embeddings
        t_out = self.time_mlp(t_embed).unsqueeze(1)
        x_patches = x_patches + t_out

        for blk in self.blocks:
            x_patches = blk(x_patches)

        x_patches = self.norm(x_patches)
        x_out = self.fc_out(x_patches)

        x_out = rearrange(
            x_out, 
            'b (h w) (p1 p2) -> b 1 (h p1) (w p2)', 
            h=H // self.patch_embed.patch_size, 
            w=W // self.patch_embed.patch_size, 
            p1=self.patch_embed.patch_size, 
            p2=self.patch_embed.patch_size
        )

        return x_out

class DiffusionModel:
    def __init__(self, model, timesteps=200, lr=1e-4, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.timesteps = timesteps
        self.betas = linear_beta_schedule(timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)
        
    def q_sample(self, y_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(y_0)
            #noise = generate_multi_resolution_noise(y_0, num_scales=3)
            #noise = anneal_noise(noise, t, self.timesteps)
        alpha_cumprod_t = self.alphas_cumprod[t].to(y_0.device)
        alpha_cumprod_t = alpha_cumprod_t.view(-1,1,1,1)
        return torch.sqrt(alpha_cumprod_t + 1e-8)*y_0 + torch.sqrt(1-alpha_cumprod_t)*noise, noise

    def p_losses(self, x, y_0):
        b = x.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=self.device).long()
        y_noisy, noise = self.q_sample(y_0, t)

        x_input = torch.cat([x, y_noisy], dim=1)
        t_emb = timestep_embedding(t, self.model.patch_embed.proj.out_channels).to(self.device)

        pred_noise = self.model(x_input, t_emb)

        mse_loss = F.mse_loss(pred_noise, noise)

        # Calculate L1 loss
        l1_loss = F.l1_loss(pred_noise, noise)

        total_loss = mse_loss + 0.1 * l1_loss

        return total_loss

    def train_step(self, x, y):
        self.optimizer.zero_grad()
        loss = self.p_losses(x, y)
        loss.backward()
        # Optional gradient clipping:
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()
    
    def adjust_learning_rate(self, metric):
        self.scheduler.step(metric)

    @torch.no_grad()
    def sample(self, x_cond):
        B, C, H, W = x_cond.shape
        y = torch.randn((B, 1, H, W), device=self.device)
        
        for i in reversed(range(self.timesteps)):
            t = torch.tensor([i]*B, device=self.device).long()
            t_emb = timestep_embedding(t, self.model.patch_embed.proj.out_channels).to(self.device)
            x_input = torch.cat([x_cond, y], dim=1)
            pred_noise = self.model(x_input, t_emb)
            alpha_t = self.alphas[i].to(self.device)
            alpha_cumprod_t = self.alphas_cumprod[i].to(self.device).view(-1,1,1,1)
            
            if i > 0:
                beta_t = self.betas[i].to(self.device)
                noise = torch.randn_like(y)
            else:
                noise = 0
            y = (1/torch.sqrt(alpha_t))*(y - ((1 - alpha_t)/torch.sqrt(1 - alpha_cumprod_t + 1e-8))*pred_noise) + torch.sqrt(beta_t + 1e-8)*noise
        return y
