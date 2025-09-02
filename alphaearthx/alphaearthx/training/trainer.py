from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any


class Trainer:
	def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
		self.model = model.to(device)
		self.optimizer = optimizer
		self.device = device

	def fit(self, dataloader: DataLoader, max_steps: int = 10) -> None:
		self.model.train()
		step = 0
		for batch in dataloader:
			if step >= max_steps:
				break
			video = batch["video"].to(self.device)  # (b, t, c, h, w)
			# Minimal forward and dummy loss
			out = self.model(video)  # (b, t, n, d)
			loss = out.mean()
			self.optimizer.zero_grad(set_to_none=True)
			loss.backward()
			self.optimizer.step()
			step += 1

	def state_dict(self) -> Dict[str, Any]:
		return {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}