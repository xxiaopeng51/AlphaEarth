import torch
import torch.nn as nn
import torch.nn.functional as F


class SwitchMLP(nn.Module):
	def __init__(self, model_dim: int, hidden_dim: int, num_experts: int = 4):
		super().__init__()
		self.num_experts = num_experts
		self.gate = nn.Linear(model_dim, num_experts)
		self.experts = nn.ModuleList([
			nn.Sequential(
				nn.Linear(model_dim, hidden_dim),
				nn.GELU(),
				nn.Linear(hidden_dim, model_dim),
			)
			for _ in range(num_experts)
		])

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: (batch, tokens, dim)
		b, n, d = x.shape
		g_logits = self.gate(x)  # (b, n, e)
		indices = g_logits.argmax(dim=-1)  # (b, n)
		# Route tokens to experts
		outputs = torch.zeros_like(x)
		for expert_id, expert in enumerate(self.experts):
			mask = indices == expert_id
			if not mask.any():
				continue
			sel = x[mask]
			out = expert(sel)
			outputs[mask] = out
		return outputs