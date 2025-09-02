import io
import json
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import webdataset as wds


def decode_npy(bytestr: bytes) -> torch.Tensor:
	array = np.load(io.BytesIO(bytestr))
	return torch.from_numpy(array)


def default_keys() -> Sequence[str]:
	return (
		"s2.npy",
		"s1.npy",
		"landsat.npy",
		"dem.npy",
		"viirs.npy",
		"meta.json",
		"caption.json",
	)


def make_dataset(
	shards_pattern: str,
	keys: Optional[Sequence[str]] = None,
	seed: int = 42,
	shuffle: int = 1000,
	resampled: bool = True,
) -> wds.WebDataset:
	"""
	Create a WebDataset pipeline with decoding for npy/json and tupled outputs.
	"""
	if keys is None:
		keys = default_keys()

	ds = (
		wds.WebDataset(shards_pattern, resampled=resampled)
		.shuffle(shuffle, initial=shuffle, rng=seed)
		.decode({"npy": decode_npy, "json": json.loads})
		.to_tuple(*keys)
	)
	return ds


def collate_batch(batch: List[tuple]) -> Dict[str, torch.Tensor]:
	"""
	Collate a list of tuples into a dict of modality tensors with presence masks.
	Assumes consistent key ordering across the dataset.
	"""
	# Convert into dict-of-lists
	cols: List[List[object]] = list(map(list, zip(*batch)))
	result: Dict[str, object] = {}
	# Heuristics for names, align with default_keys ordering
	names = [
		"s2",
		"s1",
		"landsat",
		"dem",
		"viirs",
		"meta",
		"caption",
	]
	for name, column in zip(names, cols):
		if name in {"meta", "caption"}:
			result[name] = list(column)
		else:
			# Pad variable shapes by simple stacking if shapes match; otherwise keep list
			try:
				result[name] = torch.stack([x for x in column])
			except Exception:
				result[name] = column
	# Presence mask (1 if tensor exists and not empty)
	for name in ["s2", "s1", "landsat", "dem", "viirs"]:
		value = result.get(name, None)
		if isinstance(value, list):
			mask = torch.tensor([0 if v is None else 1 for v in value], dtype=torch.uint8)
		else:
			mask = torch.ones(value.shape[0], dtype=torch.uint8)
		result[f"{name}_mask"] = mask
	return result

