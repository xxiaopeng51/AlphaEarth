import argparse
import json
import numpy as np
from scipy.optimize import curve_fit


def power_law(N, a, b):
	return a * (N ** (-b))


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--log", required=True, help="JSONL with fields: steps, tokens, loss")
	args = parser.parse_args()

	Ns = []
	Ls = []
	with open(args.log) as f:
		for line in f:
			rec = json.loads(line)
			Ns.append(rec.get("tokens", rec.get("steps", 0)))
			Ls.append(rec["loss"]) 

	Ns = np.array(Ns, dtype=float)
	Ls = np.array(Ls, dtype=float)
	popt, _ = curve_fit(power_law, Ns, Ls, maxfev=10000)
	print(json.dumps({"a": float(popt[0]), "b": float(popt[1])}))


if __name__ == "__main__":
	main()

