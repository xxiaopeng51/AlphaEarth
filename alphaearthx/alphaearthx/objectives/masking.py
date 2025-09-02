import torch


def random_spatial_mask(tokens: torch.Tensor, mask_ratio: float = 0.3):
	# tokens: (batch, tokens, dim)
	b, n, d = tokens.shape
	mask = torch.rand(b, n, device=tokens.device) < mask_ratio
	masked = tokens.clone()
	masked[mask] = 0.0
	return masked, mask