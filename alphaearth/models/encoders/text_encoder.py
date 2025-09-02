from typing import Optional

import torch
import torch.nn as nn


class TextEncoder(nn.Module):
	def __init__(self, backbone: nn.Module, proj_dim: int):
		super().__init__()
		self.backbone = backbone
		hidden = getattr(backbone.config, "hidden_size", None) or getattr(backbone.config, "d_model", 768)
		self.proj = nn.Linear(hidden, proj_dim)

	@torch.no_grad()
	def tokenize(self, tokenizer, texts):
		return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

	def forward(self, input_ids, attention_mask=None):
		outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
		cls = outputs.last_hidden_state[:, 0]
		return self.proj(cls)


class MetaEncoder(nn.Module):
	def __init__(self, in_dim: int, proj_dim: int, hidden: int = 256):
		super().__init__()
		self.net = nn.Sequential(
			nn.LayerNorm(in_dim),
			nn.Linear(in_dim, hidden),
			nn.GELU(),
			nn.Linear(hidden, proj_dim),
		)

	def forward(self, meta: torch.Tensor):
		return self.net(meta)

