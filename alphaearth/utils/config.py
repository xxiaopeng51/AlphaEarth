from dataclasses import dataclass
from typing import Any, Dict

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
	with open(path, "r") as f:
		return yaml.safe_load(f)


def get_value(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
	cur: Any = cfg
	for part in path.split("."):
		if not isinstance(cur, dict) or part not in cur:
			return default
		cur = cur[part]
	return cur

