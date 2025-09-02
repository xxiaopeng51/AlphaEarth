import argparse
import os
import subprocess


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--repo", default="https://github.com/Clay-foundation/model")
	parser.add_argument("--workdir", default="/workspace/baselines/clay")
	args = parser.parse_args()
	os.makedirs(args.workdir, exist_ok=True)
	if not os.path.exists(os.path.join(args.workdir, ".git")):
		subprocess.run(["git", "clone", args.repo, args.workdir], check=True)
	print("Cloned Clay model. Follow repo docs to prepare and run baselines.")


if __name__ == "__main__":
	main()

