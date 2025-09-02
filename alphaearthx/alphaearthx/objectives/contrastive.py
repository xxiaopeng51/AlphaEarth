import torch
import torch.nn.functional as F


def simple_nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
	# z_i, z_j: (batch, dim)
	z_i = F.normalize(z_i, dim=-1)
	z_j = F.normalize(z_j, dim=-1)
	logits = (z_i @ z_j.t()) / temperature
	target = torch.arange(z_i.size(0), device=z_i.device)
	loss_i = F.cross_entropy(logits, target)
	loss_j = F.cross_entropy(logits.t(), target)
	return (loss_i + loss_j) * 0.5