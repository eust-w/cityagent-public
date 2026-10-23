#!/usr/bin/env python3
"""Prepare a content identity without loading, copying or modifying weights."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from bridge_evidence.contest.backends import ModelManifest
from bridge_evidence.contest.model_identity import prepare_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('use a new output path; preparation receipts are immutable')
    digest = prepare_identity(ModelManifest.load(args.model_manifest), args.output)
    print(json.dumps({'receipt': str(args.output), 'sha256': digest,
                      'fast_recovery_requires': 'kernel read-only model mounts'}))


if __name__ == '__main__':
    main()
