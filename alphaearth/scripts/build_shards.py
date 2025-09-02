import argparse
import os
import tarfile
from typing import Dict, Optional

import pandas as pd
import webdataset as wds


def add_file(writer: wds.ShardWriter, sample_key: str, field: str, path: Optional[str]):
	if path is None or (isinstance(path, float) and pd.isna(path)):
		return
	if not os.path.exists(path):
		return
	with open(path, "rb") as f:
		writer.write({"__key__": sample_key, f"{field}{os.path.splitext(path)[1]}": f.read()})


def build_from_manifest(manifest_csv: str, out_pattern: str, maxcount: int = 10000):
	"""
	Manifest columns (any can be missing):
	key, s2_path, s1_path, landsat_path, dem_path, viirs_path, meta_json, caption_json
	"""
	df = pd.read_csv(manifest_csv)
	with wds.ShardWriter(out_pattern, maxcount=maxcount) as sink:
		for _, row in df.iterrows():
			key = row.get("key")
			if not isinstance(key, str) or len(key) == 0:
				continue
			add_file(sink, key, "s2", row.get("s2_path"))
			add_file(sink, key, "s1", row.get("s1_path"))
			add_file(sink, key, "landsat", row.get("landsat_path"))
			add_file(sink, key, "dem", row.get("dem_path"))
			add_file(sink, key, "viirs", row.get("viirs_path"))
			add_file(sink, key, "meta", row.get("meta_json"))
			add_file(sink, key, "caption", row.get("caption_json"))


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--manifest", required=True, help="CSV manifest path")
	parser.add_argument("--out", required=True, help="Output shard pattern, e.g. /data/shards/%06d.tar")
	parser.add_argument("--maxcount", type=int, default=10000)
	args = parser.parse_args()
	os.makedirs(os.path.dirname(args.out), exist_ok=True)
	build_from_manifest(args.manifest, args.out, args.maxcount)


if __name__ == "__main__":
	main()

