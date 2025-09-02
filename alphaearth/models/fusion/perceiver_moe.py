from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionBlock(nn.Module):
	def __init__(self, latent_dim: int, input_dim: int, heads: int = 8):
		super().__init__()
		self.norm_latent = nn.LayerNorm(latent_dim)
		self.norm_input = nn.LayerNorm(input_dim)
		self.attn = nn.MultiheadAttention(
			embed_dim=latent_dim, kdim=input_dim, vdim=input_dim, num_heads=heads, batch_first=True
		)
		self.ffn = nn.Sequential(
			nn.LayerNorm(latent_dim),
			nn.Linear(latent_dim, 4 * latent_dim),
			nn.GELU(),
			nn.Linear(4 * latent_dim, latent_dim),
		)

	def forward(self, latents: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
		l = self.norm_latent(latents)
		i = self.norm_input(inputs)
		latents2, _ = self.attn(l, i, i)
		latents = latents + latents2
		latents = latents + self.ffn(latents)
		return latents


class SimpleMoE(nn.Module):
	def __init__(self, dim: int, num_experts: int = 8):
		super().__init__()
		self.gate = nn.Linear(dim, num_experts)
		self.experts = nn.ModuleList(
			[nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)) for _ in range(num_experts)]
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		scores = F.softmax(self.gate(x), dim=-1)
		out = 0.0
		for i, expert in enumerate(self.experts):
			out = out + scores[..., i : i + 1] * expert(x)
		return out


class PerceiverMoE(nn.Module):
	def __init__(self, latent_dim: int = 1024, latent_tokens: int = 512, perceiver_layers: int = 8, moe_experts: int = 8):
		super().__init__()
		self.latents = nn.Parameter(torch.randn(1, latent_tokens, latent_dim) * 0.02)
		self.layers = nn.ModuleList([CrossAttentionBlock(latent_dim, latent_dim) for _ in range(perceiver_layers)])
		self.moe = SimpleMoE(latent_dim, num_experts=moe_experts)

	def forward(self, inputs_list: List[torch.Tensor]) -> torch.Tensor:
		batch_size = inputs_list[0].size(0)
		latents = self.latents.expand(batch_size, -1, -1)
		for layer in self.layers:
			for inputs in inputs_list:
				latents = layer(latents, inputs)
		return self.moe(latents).mean(dim=1)

