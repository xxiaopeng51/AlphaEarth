"""
Enhanced Transformer Backbone

改进的Transformer骨干网络，融合了多种先进技术：
1. Flash Attention (可选)
2. RMSNorm替代LayerNorm
3. SwiGLU激活函数
4. 旋转位置编码 (RoPE)
5. 梯度检查点
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
from einops import rearrange

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class SwiGLU(nn.Module):
    """SwiGLU激活函数 (Swish-Gated Linear Unit)"""
    
    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8/3)  # 常用比例
        
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class RotaryPositionalEmbedding(nn.Module):
    """旋转位置编码 (RoPE)"""
    
    def __init__(self, dim: int, max_seq_len: int = 8192, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # 预计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # 缓存cos/sin值
        self._cached_seq_len = 0
        self._cached_cos = None
        self._cached_sin = None
    
    def _update_cache(self, seq_len: int, device: torch.device):
        """更新缓存的cos/sin值"""
        if seq_len > self._cached_seq_len:
            self._cached_seq_len = seq_len
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cached_cos = emb.cos()
            self._cached_sin = emb.sin()
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入张量
            seq_len: 序列长度
        Returns:
            cos, sin: 旋转编码的cos和sin部分
        """
        self._update_cache(seq_len, x.device)
        return self._cached_cos[:seq_len], self._cached_sin[:seq_len]


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """应用旋转位置编码"""
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class EnhancedAttention(nn.Module):
    """增强的注意力机制，支持Flash Attention和RoPE"""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_flash_attn: bool = True,
        use_rope: bool = True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim or dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_flash_attn = use_flash_attn and FLASH_ATTN_AVAILABLE
        self.use_rope = use_rope
        
        inner_dim = self.head_dim * num_heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.dropout = dropout
        
        if use_rope:
            self.rope = RotaryPositionalEmbedding(self.head_dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) 输入序列
            attention_mask: (B, L) 注意力掩码
        Returns:
            (B, L, D) 输出序列
        """
        B, L, D = x.shape
        
        # 生成Q, K, V
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b l (three h d) -> three b h l d', 
                           three=3, h=self.num_heads, d=self.head_dim)
        
        # 应用RoPE (如果启用)
        if self.use_rope:
            cos, sin = self.rope(x, L)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Flash Attention (如果可用)
        if self.use_flash_attn:
            # Flash attention需要(B, L, H, D)格式
            q = rearrange(q, 'b h l d -> b l h d')
            k = rearrange(k, 'b h l d -> b l h d') 
            v = rearrange(v, 'b h l d -> b l h d')
            
            out = flash_attn_func(q, k, v, dropout_p=self.dropout if self.training else 0.0)
            out = rearrange(out, 'b l h d -> b l (h d)')
        else:
            # 标准注意力
            attn = (q @ k.transpose(-2, -1)) * self.scale
            
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, L)
                attn = attn.masked_fill(mask == 0, float('-inf'))
            
            attn = F.softmax(attn, dim=-1)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            
            out = attn @ v
            out = rearrange(out, 'b h l d -> b l (h d)')
        
        return self.to_out(out)


class EnhancedTransformerBlock(nn.Module):
    """增强的Transformer块"""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_flash_attn: bool = True,
        use_rope: bool = True,
        use_rmsnorm: bool = True
    ):
        super().__init__()
        
        # 选择归一化方法
        norm_layer = RMSNorm if use_rmsnorm else nn.LayerNorm
        
        self.norm1 = norm_layer(dim)
        self.attention = EnhancedAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            use_flash_attn=use_flash_attn,
            use_rope=use_rope
        )
        
        self.norm2 = norm_layer(dim)
        self.mlp = SwiGLU(dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-norm attention
        x = x + self.dropout(self.attention(self.norm1(x), attention_mask))
        
        # Pre-norm MLP
        x = x + self.dropout(self.mlp(self.norm2(x)))
        
        return x


class EnhancedTransformer(nn.Module):
    """
    增强的Transformer主体
    
    集成了现代Transformer的多种优化技术
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.0,
        use_flash_attn: bool = True,
        use_rope: bool = True,
        use_rmsnorm: bool = True,
        use_gradient_checkpointing: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Transformer层堆叠
        self.layers = nn.ModuleList([
            EnhancedTransformerBlock(
                dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                use_flash_attn=use_flash_attn,
                use_rope=use_rope,
                use_rmsnorm=use_rmsnorm
            ) for _ in range(num_layers)
        ])
        
        # 最终归一化
        norm_layer = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.final_norm = norm_layer(d_model)
    
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) 输入序列
            attention_mask: (B, L) 注意力掩码
        Returns:
            (B, L, D) 输出序列
        """
        for layer in self.layers:
            if self.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, attention_mask)
            else:
                x = layer(x, attention_mask)
        
        return self.final_norm(x)