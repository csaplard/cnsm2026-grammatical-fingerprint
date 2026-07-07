"""One-command reproduction of every result, figure, and table.

Usage:
    python reproduce.py --email YOUR@EMAIL   # email is for the Dataverse guestbook

Steps (each skipped if its output already exists; delete outputs to rerun):
  1. src/acquire.py   -- download + verify + reduce the Milano dataset
  2. src/smoke_test.py-- synthetic pipeline validation
  3. src/sweep.py     -- training-set-only hyperparameter sweep
  4. src/evaluate.py  -- single frozen test evaluation (Test-A / Test-B)
  5. src/h3_drift.py  -- H3 drift lead-time analysis
  6. src/figures.py   -- all paper figures and tables

Global seed: 42 (see src/common.py).
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
RESULTS = ROOT / "results"


def run(script, *args):
    print(f"=== {script} {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, str(SRC / script), *args],
                   check=True, cwd=SRC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True,
                    help="email for the Dataverse guestbook (dataset requirement)")
    args = ap.parse_args()

    daily = ROOT / "data" / "daily"
    if len(list(daily.glob("*.npz"))) < 62:
        run("acquire.py", "--email", args.email)
    run("smoke_test.py")
    if not (RESULTS / "sweep.json").exists():
        run("sweep.py")
    if not (RESULTS / "test_evaluation.json").exists():
        run("evaluate.py")
    if not (RESULTS / "h3_drift.json").exists():
        run("h3_drift.py")
    run("figures.py")
    print("=== reproduction complete", flush=True)


if __name__ == "__main__":
    main()
