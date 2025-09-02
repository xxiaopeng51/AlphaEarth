from typing import List

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from train.build_model import AlphaEarthModel, ModelConfig


app = FastAPI()
model = None


class CaptionRequest(BaseModel):
	texts: List[str]


class ImageEmbeddingRequest(BaseModel):
	s2: List[List[List[float]]]  # CxHxW


@app.on_event("startup")
def load_model():
	global model
	cfg = ModelConfig(eo_channels={"s2": 13, "s1": 2, "landsat": 11})
	model = AlphaEarthModel(cfg).eval().cuda()


@app.post("/embed_text")
def embed_text(req: CaptionRequest):
	with torch.no_grad():
		toks = model.tokenize(req.texts)
		toks = {k: v.cuda() for k, v in toks.items()}
		h = model.text_backbone(**toks).last_hidden_state[:, 0]
		e = model.text_proj(h)
		return {"embeddings": e.detach().cpu().tolist()}


@app.post("/embed_image")
def embed_image(req: ImageEmbeddingRequest):
	with torch.no_grad():
		arr = torch.tensor(req.s2, dtype=torch.float32).unsqueeze(0).cuda()
		out = model({"s2": arr})
		return {"embedding": out["img_emb"].detach().cpu().tolist()[0]}

