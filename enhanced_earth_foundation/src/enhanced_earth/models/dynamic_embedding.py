"""
Dynamic Modality Embedding

借鉴Clay Foundation Model的动态嵌入设计，
支持任意通道数和波段配置的自适应嵌入。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import math
from einops import rearrange, repeat


class SpectralEmbedding(nn.Module):
    """光谱嵌入层，处理不同波段的光谱信息"""
    
    def __init__(self, wave_dim: int = 128, max_channels: int = 500):
        super().__init__()
        self.wave_dim = wave_dim
        self.max_channels = max_channels
        
        # 波长到嵌入的映射
        self.wavelength_embedding = nn.Embedding(max_channels, wave_dim)
        
        # 光谱特征处理
        self.spectral_processor = nn.Sequential(
            nn.Linear(wave_dim, wave_dim * 2),
            nn.GELU(),
            nn.Linear(wave_dim * 2, wave_dim),
            nn.LayerNorm(wave_dim)
        )
    
    def forward(self, wavelengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wavelengths: (C,) 波长索引或实际波长值
        Returns:
            (C, wave_dim) 光谱嵌入
        """
        if wavelengths.dtype == torch.float:
            # 如果是实际波长值，转换为索引
            wavelengths = (wavelengths * 10).long().clamp(0, self.max_channels - 1)
        
        wave_emb = self.wavelength_embedding(wavelengths)
        return self.spectral_processor(wave_emb)


class WaveTransformer(nn.Module):
    """波段变换器，生成动态权重"""
    
    def __init__(
        self,
        wave_dim: int,
        output_dim: int,
        num_latent_tokens: int = 128,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2
    ):
        super().__init__()
        self.num_latent_tokens = num_latent_tokens
        self.wave_dim = wave_dim
        self.output_dim = output_dim
        
        # 可学习的潜在token
        self.latent_tokens = nn.Parameter(torch.randn(num_latent_tokens, wave_dim) * 0.02)
        self.bias_token = nn.Parameter(torch.randn(1, wave_dim) * 0.02)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=wave_dim,
            nhead=num_heads,
            dim_feedforward=wave_dim * 4,
            activation='gelu',
            dropout=0.1,
            norm_first=True,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 权重和偏置生成器
        self.weight_generator = nn.Linear(wave_dim, output_dim)
        self.bias_generator = nn.Linear(wave_dim, embed_dim)
    
    def forward(self, spectral_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            spectral_embeddings: (C, wave_dim) 光谱嵌入
        Returns:
            weights: (C, output_dim) 动态权重
            bias: (embed_dim,) 偏置
        """
        C = spectral_embeddings.shape[0]
        
        # 拼接潜在token、光谱嵌入和偏置token
        tokens = torch.cat([
            self.latent_tokens,  # (num_latent, wave_dim)
            spectral_embeddings,  # (C, wave_dim)
            self.bias_token  # (1, wave_dim)
        ], dim=0)  # (num_latent + C + 1, wave_dim)
        
        # Transformer处理
        processed = self.transformer(tokens.unsqueeze(0)).squeeze(0)
        
        # 提取权重和偏置
        weights = self.weight_generator(processed[self.num_latent_tokens:-1])  # (C, output_dim)
        bias = self.bias_generator(processed[-1])  # (embed_dim,)
        
        return weights, bias


class DynamicModalityEmbedding(nn.Module):
    """
    动态模态嵌入
    
    根据输入数据的通道配置动态生成嵌入权重，
    支持任意数量和类型的波段/通道。
    """
    
    def __init__(
        self,
        input_channels: int,
        d_model: int,
        patch_size: int = 16,
        modality_name: str = "optical",
        wave_dim: int = 128,
        num_latent_tokens: int = 128
    ):
        super().__init__()
        self.input_channels = input_channels
        self.d_model = d_model
        self.patch_size = patch_size
        self.modality_name = modality_name
        
        # 计算patch后的输出维度
        self.patch_dim = patch_size * patch_size * input_channels
        
        # 光谱嵌入 (对于光学/高光谱数据)
        if modality_name in ["optical", "hyperspectral"]:
            self.spectral_embedding = SpectralEmbedding(wave_dim)
            self.use_spectral = True
        else:
            self.use_spectral = False
        
        # 波段变换器
        self.wave_transformer = WaveTransformer(
            wave_dim=wave_dim if self.use_spectral else d_model,
            output_dim=self.patch_dim * d_model,
            num_latent_tokens=num_latent_tokens,
            embed_dim=d_model
        )
        
        # 模态特定的预处理
        self.modality_preprocessor = self._create_modality_preprocessor()
        
        # 位置嵌入
        self.pos_embedding = nn.Parameter(
            torch.randn(1, (224 // patch_size) ** 2, d_model) * 0.02
        )
    
    def _create_modality_preprocessor(self) -> nn.Module:
        """创建模态特定的预处理层"""
        if self.modality_name == "optical":
            return nn.Sequential(
                nn.Conv2d(self.input_channels, self.input_channels, 3, padding=1),
                nn.BatchNorm2d(self.input_channels),
                nn.GELU()
            )
        elif self.modality_name == "sar":
            return nn.Sequential(
                nn.Conv2d(self.input_channels, self.input_channels, 5, padding=2),
                nn.BatchNorm2d(self.input_channels),
                nn.ReLU()
            )
        elif self.modality_name == "hyperspectral":
            # 高光谱数据可能需要光谱维度的处理
            return nn.Sequential(
                nn.Conv2d(self.input_channels, min(64, self.input_channels), 1),
                nn.BatchNorm2d(min(64, self.input_channels)),
                nn.GELU(),
                nn.Conv2d(min(64, self.input_channels), self.input_channels, 1)
            )
        else:
            return nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, H, W, C) 输入数据
        Returns:
            (B, T, num_patches, d_model) 嵌入后的patch特征
        """
        B, T, H, W, C = x.shape
        
        # 模态预处理
        x_reshaped = rearrange(x, 'b t h w c -> (b t) c h w')
        x_preprocessed = self.modality_preprocessor(x_reshaped)
        x = rearrange(x_preprocessed, '(b t) c h w -> b t h w c', b=B, t=T)
        
        # 生成动态权重
        if self.use_spectral:
            # 为光学/高光谱数据生成光谱嵌入
            channel_indices = torch.arange(C, device=x.device)
            spectral_emb = self.spectral_embedding(channel_indices)  # (C, wave_dim)
            dynamic_weights, bias = self.wave_transformer(spectral_emb)
        else:
            # 对于其他模态，使用通道索引
            channel_emb = torch.eye(C, device=x.device) * math.sqrt(self.d_model)
            if C < self.d_model:
                padding = torch.zeros(C, self.d_model - C, device=x.device)
                channel_emb = torch.cat([channel_emb, padding], dim=1)
            elif C > self.d_model:
                channel_emb = channel_emb[:, :self.d_model]
            dynamic_weights, bias = self.wave_transformer(channel_emb)
        
        # 分patch
        patches = rearrange(x, 'b t (h p1) (w p2) c -> b t (h w) (p1 p2 c)',
                           p1=self.patch_size, p2=self.patch_size)
        
        # 动态线性变换
        # dynamic_weights: (C, patch_dim * d_model)
        # 重塑为 (patch_dim, d_model) 的权重矩阵
        weights_matrix = rearrange(dynamic_weights, 'c (p d) -> c p d', 
                                 p=self.patch_size * self.patch_size, d=self.d_model)
        
        # 对每个patch应用动态权重
        BT, num_patches, patch_features = patches.shape
        patches = rearrange(patches, 'bt np (p c) -> bt np c p', c=C, p=self.patch_size * self.patch_size)
        
        # 批量矩阵乘法: (BT, NP, C, P) @ (C, P, D) -> (BT, NP, C, D)
        embedded_patches = torch.einsum('btpc,cpd->btpd', patches, weights_matrix)
        embedded_patches = embedded_patches.sum(dim=2)  # 聚合通道维度: (BT, NP, D)
        
        # 添加偏置
        embedded_patches = embedded_patches + bias.unsqueeze(0).unsqueeze(0)
        
        # 添加位置嵌入
        if embedded_patches.shape[1] <= self.pos_embedding.shape[1]:
            embedded_patches = embedded_patches + self.pos_embedding[:, :embedded_patches.shape[1]]
        
        # 重塑回时序格式
        embedded_output = rearrange(embedded_patches, '(b t) np d -> b t np d', b=B, t=T)
        
        # 如果需要，可以重塑回空间格式
        patches_per_side = int(math.sqrt(embedded_output.shape[2]))
        if patches_per_side ** 2 == embedded_output.shape[2]:
            embedded_output = rearrange(embedded_output, 'b t (h w) d -> b t h w d', 
                                      h=patches_per_side, w=patches_per_side)
        
        return embedded_output