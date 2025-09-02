import torch
import torch.nn.functional as F


def clip_contrastive(img_emb: torch.Tensor, txt_emb: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
	img_emb = F.normalize(img_emb, dim=-1)
	txt_emb = F.normalize(txt_emb, dim=-1)
	logits = img_emb @ txt_emb.t() / temperature
	labels = torch.arange(img_emb.size(0), device=img_emb.device)
	return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) * 0.5


def masked_reconstruction(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
	return F.l1_loss(pred[mask], target[mask])

