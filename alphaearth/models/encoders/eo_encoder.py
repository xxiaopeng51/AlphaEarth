import math
from typing import Dict, Optional

import torch
import torch.nn as nn


class SpectralPatchify(nn.Module):
	def __init__(self, in_channels: int, patch_size: int, embed_dim: int):
		super().__init__()
		self.patch_size = patch_size
		self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
		self.spectral_mlp = nn.Sequential(
			nn.Conv2d(embed_dim, embed_dim, 1),
			nn.GELU(),
			nn.Conv2d(embed_dim, embed_dim, 1),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: B,C,H,W
		tokens = self.proj(x)
		tokens = self.spectral_mlp(tokens)
		# B, E, H', W' -> B, N, E
		return tokens.flatten(2).transpose(1, 2)


class PositionalEncoding(nn.Module):
	def __init__(self, dim: int, max_tokens: int = 8192):
		super().__init__()
		self.pos = nn.Parameter(torch.randn(1, max_tokens, dim) * 0.02)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return x + self.pos[:, : x.size(1), :]


class EOEncoder(nn.Module):
	def __init__(
		self,
		in_channels: int,
		patch_size: int = 16,
		dim: int = 768,
		depth: int = 12,
		heads: int = 12,
	):
		super().__init__()
		self.patch = SpectralPatchify(in_channels, patch_size, dim)
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=dim, nhead=heads, batch_first=True, norm_first=True
		)
		self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
		self.pos = PositionalEncoding(dim)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		tokens = self.patch(x)
		tokens = self.pos(tokens)
		return self.encoder(tokens)


class MultiSensorEOEncoder(nn.Module):
	"""
	Wrap multiple EO encoders with per-sensor projections to a shared dimension.
	Missing modalities can be passed as None and will be ignored.
	"""

	def __init__(self, dims: Dict[str, int], patch_size: int = 16, model_dim: int = 768, depth: int = 12, heads: int = 12):
		super().__init__()
		self.model_dim = model_dim
		self.encoders = nn.ModuleDict(
			{
				name: EOEncoder(in_channels=ch, patch_size=patch_size, dim=model_dim, depth=depth, heads=heads)
				for name, ch in dims.items()
			}
		)
		self.modality_tokens = nn.ParameterDict({name: nn.Parameter(torch.randn(1, 1, model_dim) * 0.02) for name in dims.keys()})

	def forward(self, inputs: Dict[str, Optional[torch.Tensor]]) -> Dict[str, torch.Tensor]:
		outputs: Dict[str, torch.Tensor] = {}
		for name, encoder in self.encoders.items():
			x = inputs.get(name)
			if x is None:
				continue
			tokens = encoder(x)
			mod_tok = self.modality_tokens[name].expand(tokens.size(0), -1, -1)
			outputs[name] = torch.cat([mod_tok, tokens], dim=1)
		return outputs

