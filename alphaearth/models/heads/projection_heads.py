from typing import Optional

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
	def __init__(self, in_dim: int, out_dim: int, hidden: Optional[int] = None):
		super().__init__()
		if hidden is None:
			self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim))
		else:
			self.proj = nn.Sequential(
				nn.LayerNorm(in_dim),
				nn.Linear(in_dim, hidden),
				nn.GELU(),
				nn.Linear(hidden, out_dim),
			)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.proj(x)


class LearnableTemperature(nn.Module):
	def __init__(self, init_logit: float = 2.4):
		super().__init__()
		self.logit = nn.Parameter(torch.tensor(init_logit))

	def forward(self) -> torch.Tensor:
		return torch.exp(self.logit)

