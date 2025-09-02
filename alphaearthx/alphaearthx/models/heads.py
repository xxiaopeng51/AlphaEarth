import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
	def __init__(self, in_dim: int, out_dim: int = 256):
		super().__init__()
		self.net = nn.Sequential(
			nn.LayerNorm(in_dim),
			nn.Linear(in_dim, in_dim),
			nn.GELU(),
			nn.Linear(in_dim, out_dim),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)