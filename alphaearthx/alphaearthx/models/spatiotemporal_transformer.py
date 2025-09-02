import torch
import torch.nn as nn
from einops import rearrange


class PatchEmbedding(nn.Module):
	def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 4):
		super().__init__()
		self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: (batch, channels, height, width)
		return rearrange(self.proj(x), "b c h w -> b (h w) c")


class TemporalSelfAttention(nn.Module):
	def __init__(self, embed_dim: int, num_heads: int):
		super().__init__()
		self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
		self.ln = nn.LayerNorm(embed_dim)
		self.mlp = nn.Sequential(
			nn.Linear(embed_dim, embed_dim * 4),
			nn.GELU(),
			nn.Linear(embed_dim * 4, embed_dim),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: (batch, time, tokens, dim)
		b, t, n, d = x.shape
		x_flat = rearrange(x, "b t n d -> (b n) t d")
		attn_out, _ = self.attn(x_flat, x_flat, x_flat, need_weights=False)
		x_flat = x_flat + attn_out
		x_flat = x_flat + self.mlp(self.ln(x_flat))
		return rearrange(x_flat, "(b n) t d -> b t n d", b=b, n=n)


class SpatioTemporalBackbone(nn.Module):
	def __init__(self, in_channels: int = 3, embed_dim: int = 256, patch_size: int = 4, depth: int = 4, num_heads: int = 8):
		super().__init__()
		self.patch = PatchEmbedding(in_channels, embed_dim, patch_size)
		self.pos = nn.Parameter(torch.randn(1, 1, 1024, embed_dim) * 0.02)
		self.temporal_blocks = nn.ModuleList([
			TemporalSelfAttention(embed_dim, num_heads) for _ in range(depth)
		])

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: (batch, time, channels, height, width)
		b, t, c, h, w = x.shape
		tokens = []
		for i in range(t):
			tokens.append(self.patch(x[:, i]))  # (b, n, d)
		x = torch.stack(tokens, dim=1)  # (b, t, n, d)
		n = x.size(2)
		pos = self.pos[:, :, :n, :]
		x = x + pos
		for blk in self.temporal_blocks:
			x = blk(x)
		return x  # (b, t, n, d)