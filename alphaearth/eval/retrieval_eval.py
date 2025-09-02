import argparse
import numpy as np
import torch

from datasets.webdataset_datamodule import make_dataset, collate_batch
from train.build_model import AlphaEarthModel, ModelConfig
from eval.retrieval import build_index, search


def recall_at_k(I: np.ndarray, k: int) -> float:
	N = I.shape[0]
	correct = sum(1 for i in range(N) if i in I[i, :k])
	return correct / float(N)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--shards", required=True)
	parser.add_argument("--limit", type=int, default=2048)
	args = parser.parse_args()

	ds = make_dataset(args.shards)
	dl = torch.utils.data.DataLoader(ds, batch_size=8, num_workers=4, collate_fn=collate_batch)

	model = AlphaEarthModel(ModelConfig(eo_channels={"s2": 13, "s1": 2, "landsat": 11})).cuda().eval()
	img_embeds = []
	txt_embeds = []
	count = 0
	with torch.no_grad():
		for batch in dl:
			s2 = batch.get("s2")
			cap = batch.get("caption")
			if not isinstance(s2, torch.Tensor) or not isinstance(cap, list):
				continue
			out = model({"s2": s2.cuda(), "caption": cap})
			img_embeds.append(out["img_emb"].detach().cpu().numpy())
			txt_embeds.append(out["text_emb"].detach().cpu().numpy())
			count += s2.shape[0]
			if count >= args.limit:
				break

	img = np.concatenate(img_embeds, axis=0)
	txt = np.concatenate(txt_embeds, axis=0)
	index = build_index(img.copy())
	D, I = search(index, txt.copy(), topk=10)
	print({"R@1": recall_at_k(I, 1), "R@5": recall_at_k(I, 5)})


if __name__ == "__main__":
	main()

