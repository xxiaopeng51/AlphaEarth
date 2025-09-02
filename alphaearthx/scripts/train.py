import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader

from alphaearthx.data import DatasetRegistry
from alphaearthx.models import SpatioTemporalBackbone
from alphaearthx.training import Trainer


@hydra.main(config_path="../alphaearthx/configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
	print(OmegaConf.to_yaml(cfg))
	# Dataset
	ds = DatasetRegistry.build(
		cfg.data.name,
		num_samples=cfg.data.num_samples,
		time_steps=cfg.data.time_steps,
		channels=cfg.data.channels,
		height=cfg.data.height,
		width=cfg.data.width,
	)
	dl = DataLoader(ds, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers)
	# Model
	model = hydra.utils.instantiate(cfg.model.backbone)
	# Optimizer
	optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
	# Trainer
	trainer = Trainer(model=model, optimizer=optimizer)
	trainer.fit(dl, max_steps=cfg.trainer.max_steps)


if __name__ == "__main__":
	main()