import argparse
import subprocess
from tqdm import tqdm
import os

def run_script_with_progress(script_path):
    with open(script_path) as f:
        lines = [line.strip() for line in f if line.strip()]

    exec_lines = [line for line in lines if not line.startswith("cd ")]

    pbar = tqdm(exec_lines, desc=f"Running {script_path}")

    cwd = None
    for line in lines:
        if line.startswith("cd "):
            path = line[3:].strip()
            if path == "..":
                cwd = os.path.dirname(cwd) if cwd else ".."
            else:
                cwd = path if cwd is None else os.path.join(cwd, path)
        else:
            subprocess.run(line, shell=True, cwd=cwd,
               stdout=subprocess.DEVNULL,
               stderr=subprocess.DEVNULL)
            pbar.update(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("script", help="Path to the .sh script to execute")
    args = parser.parse_args()
    run_script_with_progress(args.script)
    #run_script_with_progress("scripts/run_compute1.sh")