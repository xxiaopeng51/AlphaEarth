AlphaEarth Foundations (multimodal, global-scale EO foundation model)

Quickstart
1) Prepare a CSV manifest with columns: key,s2_path,s1_path,landsat_path,dem_path,viirs_path,meta_json,caption_json
2) Build shards:
   python -m scripts.build_shards --manifest /data/manifest.csv --out /data/alphaearth/shards/%06d.tar --maxcount 5000
3) Run a smoke test training loop (single GPU works for the demo):
   python -m train.train_dist

Repo layout
- configs/base.yaml: default hyperparameters
- data/builders/tile_index.py: tile utils and global index
- datasets/webdataset_datamodule.py: WebDataset pipeline
- models/encoders: EO, text, meta encoders
- models/fusion/perceiver_moe.py: Perceiver + MoE fusion
- objectives/losses.py: contrastive & masked reconstruction losses
- eval/retrieval.py: building/searching an embedding index
- scripts/build_shards.py: shard builder from CSV manifest
- train/train_dist.py: distributed training smoke test

