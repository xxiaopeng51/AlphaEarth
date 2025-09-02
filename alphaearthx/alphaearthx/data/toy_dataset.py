import torch
from torch.utils.data import Dataset
from .registry import DatasetRegistry


class ToySpatioTemporalDataset(Dataset):
	def __init__(self, num_samples: int = 128, time_steps: int = 4, channels: int = 3, height: int = 64, width: int = 64):
		self.num_samples = num_samples
		self.time_steps = time_steps
		self.channels = channels
		self.height = height
		self.width = width

	def __len__(self) -> int:
		return self.num_samples

	def __getitem__(self, idx: int):
		# Return a dict to mimic multi-modal inputs later
		video = torch.randn(self.time_steps, self.channels, self.height, self.width)
		return {"video": video}


def _builder(**kwargs):
	return ToySpatioTemporalDataset(**kwargs)


DatasetRegistry.register("toy_spatiotemporal", _builder)