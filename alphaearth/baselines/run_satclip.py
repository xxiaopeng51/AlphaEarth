import argparse
import os
import subprocess


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--repo", default="https://github.com/microsoft/satclip")
	parser.add_argument("--workdir", default="/workspace/baselines/satclip")
	args = parser.parse_args()
	os.makedirs(args.workdir, exist_ok=True)
	if not os.path.exists(os.path.join(args.workdir, ".git")):
		subprocess.run(["git", "clone", args.repo, args.workdir], check=True)
	print("Cloned SatCLIP. Please follow repo instructions to prepare data and run.")


if __name__ == "__main__":
	main()

