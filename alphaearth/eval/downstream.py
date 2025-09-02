"""
Lightweight downstream evaluation hooks.
Fill in dataset loaders as needed; this provides the API shape.
"""
from typing import Dict, Tuple

import numpy as np
import torch


def linear_probe(train_feats: np.ndarray, train_labels: np.ndarray, test_feats: np.ndarray) -> np.ndarray:
	# closed-form ridge regression as a quick probe
	lmbda = 1e-2
	X = train_feats
	Y = train_labels
	W = np.linalg.solve(X.T @ X + lmbda * np.eye(X.shape[1]), X.T @ Y)
	return test_feats @ W


def evaluate_classification(pred_logits: np.ndarray, test_labels: np.ndarray) -> Dict[str, float]:
	pred = pred_logits.argmax(axis=1)
	acc = (pred == test_labels).mean().item() if hasattr(acc := (pred == test_labels).mean(), "item") else float(acc)
	return {"acc": float(acc)}

