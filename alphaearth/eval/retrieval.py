import faiss
import numpy as np


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
	index = faiss.IndexFlatIP(embeddings.shape[1])
	faiss.normalize_L2(embeddings)
	index.add(embeddings)
	return index


def search(index: faiss.IndexFlatIP, queries: np.ndarray, topk: int = 10):
	faiss.normalize_L2(queries)
	D, I = index.search(queries, topk)
	return D, I

