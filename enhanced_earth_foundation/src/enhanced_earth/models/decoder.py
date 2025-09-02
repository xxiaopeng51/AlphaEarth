"""
Von Mises-Fisher Decoder

基于von Mises-Fisher分布的解码器，用于重建多模态数据。
支持从64D球面嵌入重建到各种模态数据。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
import math
from einops import rearrange, repeat

from .backbone import EnhancedTransformer
from .dynamic_embedding import DynamicModalityEmbedding


class VonMisesFisherDistribution(nn.Module):
    """von Mises-Fisher分布实现"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # 浓度参数 (kappa)
        self.log_kappa = nn.Parameter(torch.zeros(1))
        
    def log_prob(self, x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        计算vMF分布的对数概率
        
        Args:
            x: (*, dim) 观测点 (单位向量)
            mu: (*, dim) 均值方向 (单位向量)
        Returns:
            (*, ) 对数概率
        """
        kappa = torch.exp(self.log_kappa)
        
        # 确保输入是单位向量
        x = F.normalize(x, p=2, dim=-1)
        mu = F.normalize(mu, p=2, dim=-1)
        
        # vMF对数概率密度
        dot_product = torch.sum(x * mu, dim=-1)
        
        # 归一化常数的对数 (近似)
        log_norm_const = (self.dim / 2 - 1) * torch.log(kappa) - \
                        (self.dim / 2) * math.log(2 * math.pi) - \
                        torch.logaddexp(torch.zeros_like(kappa), kappa)
        
        return log_norm_const + kappa * dot_product
    
    def sample(self, mu: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """从vMF分布采样"""
        # 简化实现：在均值方向附近添加噪声
        kappa = torch.exp(self.log_kappa)
        noise_scale = 1.0 / (kappa + 1e-8)
        
        samples = mu.unsqueeze(0).expand(num_samples, -1, -1)
        noise = torch.randn_like(samples) * noise_scale
        
        return F.normalize(samples + noise, p=2, dim=-1)


class ModalitySpecificDecoder(nn.Module):
    """模态特定解码器"""
    
    def __init__(
        self,
        embed_dim: int,
        output_channels: int,
        modality_name: str,
        d_model: int = 512,
        patch_size: int = 16
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_channels = output_channels
        self.modality_name = modality_name
        self.d_model = d_model
        self.patch_size = patch_size
        
        # 嵌入到特征的投影
        self.embed_to_feature = nn.Sequential(
            nn.Linear(embed_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 模态特定的解码Transformer
        self.decoder_transformer = EnhancedTransformer(
            d_model=d_model,
            num_heads=d_model // 64,
            num_layers=4,
            dropout=0.1
        )
        
        # 输出投影
        self.output_projection = nn.Linear(d_model, output_channels * patch_size * patch_size)
        
        # 模态特定的后处理
        self.postprocessor = self._create_modality_postprocessor()
        
        # vMF分布
        self.vmf_distribution = VonMisesFisherDistribution(embed_dim)
    
    def _create_modality_postprocessor(self) -> nn.Module:
        """创建模态特定的后处理层"""
        if self.modality_name == "optical":
            return nn.Sequential(
                nn.Conv2d(self.output_channels, self.output_channels, 3, padding=1),
                nn.BatchNorm2d(self.output_channels),
                nn.Sigmoid()  # 光学数据通常在[0,1]范围
            )
        elif self.modality_name == "sar":
            return nn.Sequential(
                nn.Conv2d(self.output_channels, self.output_channels, 3, padding=1),
                nn.BatchNorm2d(self.output_channels),
                nn.ReLU()  # SAR数据非负
            )
        elif self.modality_name == "hyperspectral":
            return nn.Sequential(
                nn.Conv2d(self.output_channels, self.output_channels, 1),  # 1x1卷积保持光谱信息
                nn.BatchNorm2d(self.output_channels),
                nn.Sigmoid()
            )
        else:
            return nn.Identity()
    
    def forward(
        self,
        embeddings: torch.Tensor,  # (B, H, W, embed_dim)
        target_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        从球面嵌入解码到模态数据
        
        Args:
            embeddings: (B, H, W, embed_dim) 球面嵌入
            target_size: (H_target, W_target) 目标空间尺寸
        Returns:
            (B, output_channels, H_target, W_target) 重建的模态数据
        """
        B, H, W, E = embeddings.shape
        
        # 投影到特征空间
        features = self.embed_to_feature(embeddings)  # (B, H, W, d_model)
        
        # 重塑为序列格式
        feat_seq = rearrange(features, 'b h w d -> b (h w) d')
        
        # 解码Transformer
        decoded_seq = self.decoder_transformer(feat_seq)  # (B, HW, d_model)
        
        # 输出投影
        output_seq = self.output_projection(decoded_seq)  # (B, HW, C*P*P)
        
        # 重塑为patch格式
        patches = rearrange(output_seq, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)',
                           h=H, w=W, c=self.output_channels, 
                           p1=self.patch_size, p2=self.patch_size)
        
        # 目标尺寸调整
        if target_size is not None:
            patches = F.interpolate(patches, size=target_size, mode='bilinear', align_corners=False)
        
        # 模态特定后处理
        output = self.postprocessor(patches)
        
        return output
    
    def compute_reconstruction_loss(
        self,
        embeddings: torch.Tensor,
        target: torch.Tensor,
        reduction: str = "mean"
    ) -> torch.Tensor:
        """计算重建损失"""
        reconstructed = self.forward(embeddings, target.shape[-2:])
        
        # 根据模态选择损失函数
        if self.modality_name in ["optical", "hyperspectral"]:
            # L1 + SSIM损失
            l1_loss = F.l1_loss(reconstructed, target, reduction=reduction)
            ssim_loss = 1 - self._compute_ssim(reconstructed, target)
            return l1_loss + 0.1 * ssim_loss
        elif self.modality_name == "sar":
            # L2损失 (SAR数据对噪声敏感)
            return F.mse_loss(reconstructed, target, reduction=reduction)
        else:
            # 默认L1损失
            return F.l1_loss(reconstructed, target, reduction=reduction)
    
    def _compute_ssim(self, x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
        """计算SSIM (简化版本)"""
        # 简化的SSIM实现
        mu_x = F.avg_pool2d(x, window_size, stride=1, padding=window_size//2)
        mu_y = F.avg_pool2d(y, window_size, stride=1, padding=window_size//2)
        
        sigma_x = F.avg_pool2d(x * x, window_size, stride=1, padding=window_size//2) - mu_x * mu_x
        sigma_y = F.avg_pool2d(y * y, window_size, stride=1, padding=window_size//2) - mu_y * mu_y
        sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=window_size//2) - mu_x * mu_y
        
        c1, c2 = 0.01**2, 0.03**2
        
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
               ((mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2))
        
        return ssim.mean()


class VonMisesFisherDecoder(nn.Module):
    """
    von Mises-Fisher解码器主类
    
    从64D球面嵌入重建多模态数据
    """
    
    def __init__(
        self,
        embed_dim: int = 64,
        decode_modalities: Dict[str, int] = None,
        d_model: int = 512,
        patch_size: int = 16
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.decode_modalities = decode_modalities or {"optical": 13}
        self.d_model = d_model
        
        # 为每个模态创建解码器
        self.modality_decoders = nn.ModuleDict()
        for modality, channels in self.decode_modalities.items():
            self.modality_decoders[modality] = ModalitySpecificDecoder(
                embed_dim=embed_dim,
                output_channels=channels,
                modality_name=modality,
                d_model=d_model,
                patch_size=patch_size
            )
        
        # 共享的特征增强
        self.shared_enhancer = nn.Sequential(
            nn.Linear(embed_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
    
    def forward(
        self,
        embeddings: torch.Tensor,  # (B, H, W, embed_dim)
        target_data: Optional[Dict[str, torch.Tensor]] = None,
        target_modalities: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        解码球面嵌入到多模态数据
        
        Args:
            embeddings: (B, H, W, embed_dim) 球面嵌入
            target_data: 目标数据 (用于确定输出尺寸)
            target_modalities: 需要解码的模态列表
        Returns:
            Dict[modality_name, reconstructed_data]
        """
        if target_modalities is None:
            target_modalities = list(self.decode_modalities.keys())
        
        # 共享特征增强
        enhanced_embeddings = self.shared_enhancer(embeddings)  # (B, H, W, d_model)
        
        reconstructions = {}
        
        for modality in target_modalities:
            if modality in self.modality_decoders:
                # 确定目标尺寸
                target_size = None
                if target_data is not None and modality in target_data:
                    target_size = target_data[modality].shape[-2:]
                
                # 解码
                reconstructed = self.modality_decoders[modality](
                    embeddings, target_size
                )
                reconstructions[modality] = reconstructed
        
        return reconstructions
    
    def compute_total_reconstruction_loss(
        self,
        embeddings: torch.Tensor,
        target_data: Dict[str, torch.Tensor],
        loss_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """计算总重建损失"""
        if loss_weights is None:
            loss_weights = {name: 1.0 for name in target_data.keys()}
        
        losses = {}
        total_loss = 0.0
        
        for modality, target in target_data.items():
            if modality in self.modality_decoders:
                modality_loss = self.modality_decoders[modality].compute_reconstruction_loss(
                    embeddings, target
                )
                losses[f"{modality}_loss"] = modality_loss
                total_loss += loss_weights.get(modality, 1.0) * modality_loss
        
        losses["total_reconstruction_loss"] = total_loss
        return losses