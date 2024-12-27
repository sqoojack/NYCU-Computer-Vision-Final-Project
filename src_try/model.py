
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from diffusers.models.transformers import DiTModelOutput
from diffusers.models.attention import Attention
from einops import rearrange    # used to rearrange the tensor's dimension
from dataclasses import dataclass

@dataclass
class DiTModelOutput:
    sample: torch.Tensor
    
class MyDiTConfig:
    addition_time_embed_dim: int = 256  # it's in original unet attribute

# to solve not have linear_1 bug
class AddEmbedding(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear_1(x)

""" shape of input tensor: (B, F, C, H, W), Batch_size, Frames, Channels, Height, Width """
""" define DiT """
class MyDiT(nn.Module):
    # All of the following parameters are custom parameters
    def __init__(self, image_size=64, patch_size=8,  #  patch: small block, patch_size=8: means that every patch is 8x8 pixels
                in_channels=4,  # has 4 channels in every frame
                hidden_dim=768, num_layers=12,
                num_heads=12, mlp_ratio=4.0,   # it means that dimension of MLP is four times the hidden_dim (MLP: multi layer perception)
                use_cross_attention=True,
                addition_time_embed_dim=256):      # in resource-constrained situations, you need to forbid it
                
        super().__init__()  # ensure that nn.Module can successfully initialize
        self.config = MyDiTConfig()
        
        self.add_embedding = AddEmbedding(self.config.addition_time_embed_dim, hidden_dim)
        
        self.image_size = image_size    # store the externally passed parameters as attributes of the class
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_patches = (image_size // patch_size) ** 2
        
        self.latent_fft_post_merge = False
        self.latent_fft_ratio = 0.5
        self.optimize_latent_iter = 5
        self.optimize_latent_lr = 0.21
        self.optimize_latent_time = list(range(30, 46))
        self.record_layer_sublayer = [(2, 1), (2, 2)]
        
        # used to devide input image into small patches and map each patch to a feature space with a dimension of hidden_dim
        self.patch_embed = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)    # stride(步幅) = patch_size: ensure each kernel will not overlap   
        
        self.time_embedding = nn.Sequential(    # is similar to U-Net's time embedding, used to let timesteps map into continue vector
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.time_proj = nn.Embedding(10000, hidden_dim)    # used to map discrete timesteps to a hidden_dim dimensional space, 
        self.add_time_proj = nn.Embedding(10000, hidden_dim)    # provide added_time_ids embedding capability, assume the maximum timestep is 10000
        
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))  # add position information for each patch
        
        self.layers = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio, use_cross_attention=use_cross_attention)     # create num_layers instance of DiTBlock
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)    # layer normalization
        
        # project the final hidden_dim back to the latent dimension (align to dimension of UNet's final output)
        self.head = nn.Linear(hidden_dim, in_channels * (patch_size ** 2))

        self.latent_shape = None    # used to reshape
        
    def forward(self, sample, timestep, encoder_hidden_states, added_time_ids, return_dict: bool = True):
        # sample.shape: [B, F, C, H, W]
        # convert timestep to tensor and embedding
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)
        b, f, c, h, w = sample.shape
        self.latent_shape = sample.shape
        
        x = sample.reshape(b*f, c, h, w)
        
        x = self.patch_embed(x)     # patch embedding
        x = x.flatten(2).transpose(1, 2)    # [B*F, hidden_dim, H_patch, W_patch] -> [B*F, hidden_dim, num_patches] -> [B*F, num_patches, hidden_dim]
        
        x = x + self.pos_embed
        
        t_embed = self.time_proj(timestep)  # [1, hidden_dim]
        t_embed = t_embed.to(x.dtype)
        t_embed = self.time_embedding(t_embed)  # [1, hidden_dim]
        
        add_t_embed = self.add_time_proj(added_time_ids.flatten())   # process add_time
        add_t_embed = add_t_embed.reshape((b, -1))    # [B, ...]
        add_t_embed = add_t_embed.to(x.dtype)
        add_t_embed = self.add_embedding(add_t_embed)   # [B, hidden_dim]
        
        # reshape add_t_embed to match the processing dimension of the Transformer model
        add_t_embed = add_t_embed.unsqueeze(1).expand(b, f, add_t_embed.shape[-1]).reshape(b*f, add_t_embed.shape[-1])   # unsqueeze(1): let its shape convert to [B, 1, hidden_dim]
        t_embed = t_embed + add_t_embed
        
        x = x + t_embed.unsqueeze(1)    # similar to add time emb to hidden state in UNet
        
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(f, dim=0)   # flatten it so that can match x.shape (repeat f times in each batch)
            
        for i, block in enumerate(self.layers):
            x = block(x, encoder_hidden_states=encoder_hidden_states, parent=self, layer_index=i)
        
        x = self.norm(x)
        
        x = self.head(x)    # let it move to original latent patch [B*F, C, H, W]
        p = self.patch_size
        hp = h // p
        wp = w // p
        
        x = x.transpose(1, 2).reshape(b*f, c, hp, wp)
        x = x.reshape(b, f, c, hp*p, wp*p)
        
        if not return_dict:
            return (x,)
        
        return DiTModelOutput(sample=x)
    
    def dtype(self):
        return torch.float32
        
class DiTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_ratio, use_cross_attention=True):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.multi_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.cross_attn_norm = nn.LayerNorm(hidden_dim)  # let normalize input data before attn_layer
            self.cross_attn = Attention(
                query_dim=hidden_dim,               
                cross_attention_dim=hidden_dim,     
                heads=num_heads,                     
                dim_head=hidden_dim // num_heads,   
                dropout=0.0,                         
                bias=False,                          
                only_cross_attention=True,            
                added_kv_proj_dim=hidden_dim
            )
        
        # MLP (Feed-Forward network)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim)
        )
        
    def forward(self, x, encoder_hidden_states=None, parent=None, layer_index=None):
        h = self.attn_norm(x)
        attn_out, _ = self.multi_attn(h, h, h, need_weights=False)
        x = x + attn_out
                    
        # cross-Attention
        if self.use_cross_attention and encoder_hidden_states is not None:  
            h = self.cross_attn_norm(x)
            x = x + self.cross_attn(h, encoder_hidden_states)[0]
            
        h = self.mlp_norm(x)
        x = x + self.mlp(h)
        
        return x