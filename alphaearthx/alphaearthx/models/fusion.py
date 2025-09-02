import torch
import torch.nn as nn
from einops import rearrange


class ModalityFusion(nn.Module):
	def __init__(self, embed_dim: int = 256, num_heads: int = 8, depth: int = 2):
		super().__init__()
		self.ln = nn.LayerNorm(embed_dim)
		self.blocks = nn.ModuleList([
			nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
			for _ in range(depth)
		])

	def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
		# features: list of (batch, tokens, dim) from different modalities
		z = torch.cat(features, dim=1)
		z = self.ln(z)
		for blk in self.blocks:
			z = blk(z)
		return z