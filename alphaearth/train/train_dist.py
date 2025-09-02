import os
import argparse
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from datasets.webdataset_datamodule import make_dataset, collate_batch
from models import MultiSensorEOEncoder, PerceiverMoE
from objectives.losses import clip_contrastive
from utils.config import load_yaml, get_value


@dataclass
class TrainConfig:
	shards: str
	global_batch_size: int = 256
	grad_accum_steps: int = 8
	precision: str = "bf16"
	model_dim: int = 768
	latent_dim: int = 1024
	latent_tokens: int = 512


def setup_ddp():
	if not dist.is_initialized():
		dist.init_process_group(backend="nccl")
	torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())


def main(cfg: TrainConfig, log_wandb: bool = False, project: str = "alphaearth"):
	setup_ddp()
	device = torch.device("cuda")

	ds = make_dataset(cfg.shards)
	world = dist.get_world_size()
	batch_per_rank = max(1, cfg.global_batch_size // world // cfg.grad_accum_steps)
	dl = torch.utils.data.DataLoader(ds, batch_size=batch_per_rank, num_workers=4, collate_fn=collate_batch)

	eo = MultiSensorEOEncoder({"s2": 13, "s1": 2, "landsat": 11}, patch_size=16, model_dim=cfg.model_dim)
	fusion = PerceiverMoE(latent_dim=cfg.latent_dim, latent_tokens=cfg.latent_tokens)

	wrap_policy = size_based_auto_wrap_policy(min_num_params=10_000_000)
	eo = FSDP(eo.cuda(), auto_wrap_policy=wrap_policy)
	fusion = FSDP(fusion.cuda(), auto_wrap_policy=wrap_policy)

	optim = torch.optim.AdamW(list(eo.parameters()) + list(fusion.parameters()), lr=2e-4, weight_decay=0.05)
	scaler = torch.cuda.amp.GradScaler(enabled=(cfg.precision == "fp16"))

	eo.train(); fusion.train()

	if log_wandb and dist.get_rank() == 0:
		try:
			import wandb
			wandb.init(project=project, config={"batch_per_rank": batch_per_rank, "precision": cfg.precision})
		except Exception:
			wandb = None

	for step, sample in enumerate(dl):
		# Demo loop: build a minimal forward using s2 only when available
		s2 = sample.get("s2")
		if not isinstance(s2, torch.Tensor):
			continue
		s2 = s2.float().to(device)
		inputs = {"s2": s2}
		with torch.cuda.amp.autocast(dtype=torch.bfloat16 if cfg.precision == "bf16" else torch.float16, enabled=True):
			encoded = eo(inputs)
			xs = list(encoded.values())
			img_emb = fusion(xs)
			# Fake text embedding for smoke test
			txt_emb = img_emb.detach()  # same tensor to keep loss ~log(B)
			loss = clip_contrastive(img_emb, txt_emb)

		optim.zero_grad(set_to_none=True)
		scaler.scale(loss).backward()
		scaler.step(optim)
		scaler.update()

		if step % 50 == 0 and dist.get_rank() == 0:
			print(f"step {step} loss {loss.item():.4f}")
			if log_wandb:
				try:
					wandb.log({"loss": loss.item(), "step": step})
				except Exception:
					pass

		if step > 100:
			break

	# Save state dict (rank 0)
	if dist.get_rank() == 0:
		os.makedirs("/workspace/alphaearth/checkpoints", exist_ok=True)
		torch.save({"eo": eo.state_dict(), "fusion": fusion.state_dict()}, "/workspace/alphaearth/checkpoints/smoke.pt")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", default="/workspace/alphaearth/configs/base.yaml")
	parser.add_argument("--wandb", action="store_true")
	args = parser.parse_args()
	cfg_yaml = load_yaml(args.config)
	cfg = TrainConfig(
		shards=get_value(cfg_yaml, "data.shards", "/data/alphaearth/shards/{000000..000000}.tar"),
		global_batch_size=int(get_value(cfg_yaml, "global_batch_size", 256)),
		grad_accum_steps=int(get_value(cfg_yaml, "grad_accum_steps", 8)),
		precision=str(get_value(cfg_yaml, "precision", "bf16")),
		model_dim=int(get_value(cfg_yaml, "model.eo.dim", 768)),
		latent_dim=int(get_value(cfg_yaml, "model.fusion.latent_dim", 1024)),
		latent_tokens=int(get_value(cfg_yaml, "model.fusion.latent_tokens", 512)),
	)
	main(cfg, log_wandb=args.wandb)

