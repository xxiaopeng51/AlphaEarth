# AlphaEarthX (skeleton)

AlphaEarthX is a research scaffold inspired by AlphaEarth Foundations, Clay Foundation Model, SatCLIP, and Prithvi. It targets a global-scale, multi-modal, spatiotemporal foundation model with scalable training objectives.

Status: repository skeleton with configs, model/data/objectives/training stubs.

## Quickstart (toy run)

```bash
python -m pip install -r requirements.txt
python scripts/train.py trainer.max_steps=5
```

## Directory layout

- `alphaearthx/models`: spatiotemporal backbone, modality adapters, fusion, heads
- `alphaearthx/data`: dataset registry, tiling/grid utils, random toy dataset
- `alphaearthx/objectives`: contrastive and masked modeling stubs
- `alphaearthx/training`: trainer loop, distributed utils, helpers
- `alphaearthx/configs`: Hydra configs
- `scripts/train.py`: Hydra entry point

## Notes
- This is a minimal scaffold to iterate quickly. Swap the toy dataset for real EO sources and add objectives, modalities, and scaling features incrementally.
