"""Package this entry as code/run.py for the explicit on-host batch workflow."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from bridge_evidence.contest.contest_batch_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
