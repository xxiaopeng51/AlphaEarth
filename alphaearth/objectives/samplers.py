from dataclasses import dataclass
from typing import Dict


@dataclass
class LossWeights:
	contrastive_weight: float = 1.0
	mae_weight: float = 1.0
	temporal_weight: float = 0.3

	def weight(self, losses: Dict[str, float]) -> Dict[str, float]:
		return {
			"contrastive": self.contrastive_weight,
			"mae": self.mae_weight if "mae" in losses else 0.0,
			"temporal": self.temporal_weight if "temporal" in losses else 0.0,
		}

