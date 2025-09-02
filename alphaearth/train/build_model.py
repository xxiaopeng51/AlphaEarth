from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from models import MultiSensorEOEncoder, PerceiverMoE
from models.heads.projection_heads import ProjectionHead, LearnableTemperature
from objectives.losses import clip_contrastive


@dataclass
class ModelConfig:
	eo_channels: Dict[str, int]
	model_dim: int = 768
	text_model: str = "distilbert-base-uncased"
	proj_dim: int = 768
	latent_dim: int = 1024
	latent_tokens: int = 512
	perceiver_layers: int = 8
	moe_experts: int = 8


class AlphaEarthModel(nn.Module):
	def __init__(self, cfg: ModelConfig):
		super().__init__()
		self.cfg = cfg
		self.eo = MultiSensorEOEncoder(cfg.eo_channels, patch_size=16, model_dim=cfg.model_dim)
		self.text_backbone = AutoModel.from_pretrained(cfg.text_model)
		self.text_tokenizer = AutoTokenizer.from_pretrained(cfg.text_model)
		self.text_proj = ProjectionHead(self.text_backbone.config.hidden_size, cfg.proj_dim)
		self.meta_proj = ProjectionHead(16, cfg.proj_dim)  # placeholder meta dim
		self.fusion = PerceiverMoE(cfg.latent_dim, cfg.latent_tokens, cfg.perceiver_layers, cfg.moe_experts)
		self.temp = LearnableTemperature()

	@torch.no_grad()
	def tokenize(self, texts):
		return self.text_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

	def forward(self, batch: Dict[str, object]) -> Dict[str, torch.Tensor]:
		inputs: Dict[str, Optional[torch.Tensor]] = {
			"s2": batch.get("s2"),
			"s1": batch.get("s1"),
			"landsat": batch.get("landsat"),
		}
		encoded = self.eo(inputs)
		modal_tokens = [v for v in encoded.values() if v is not None]
		img_emb = self.fusion(modal_tokens)

		# Text
		captions = batch.get("caption", None)
		if captions is not None and isinstance(captions, list) and len(captions) > 0:
			toks = self.tokenize(captions)
			toks = {k: v.to(img_emb.device) for k, v in toks.items()}
			text_hidden = self.text_backbone(**toks).last_hidden_state[:, 0]
			text_emb = self.text_proj(text_hidden)
		else:
			text_emb = None

		return {"img_emb": img_emb, "text_emb": text_emb, "temperature": self.temp()}

	def compute_losses(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
		losses: Dict[str, torch.Tensor] = {}
		if outputs["text_emb"] is not None:
			losses["contrastive"] = clip_contrastive(outputs["img_emb"], outputs["text_emb"], temperature=outputs["temperature"]) 
		else:
			losses["contrastive"] = torch.tensor(0.0, device=outputs["img_emb"].device)
		losses["total"] = sum(losses.values())
		return losses

