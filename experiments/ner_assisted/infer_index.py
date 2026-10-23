"""Training-only experiment: run the existing UIE checkpoint, emit bound index.

No model services are started or reconfigured. CUDA device is explicit in the
launch environment. Source paragraphs are never synthesized from predictions.
"""
from __future__ import annotations
import argparse
import bisect
import hashlib
import json
from pathlib import Path
import time


def main():
    started = time.monotonic()
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    import torch
    from torch import nn
    from transformers import ErnieModel, ErnieConfig, BertTokenizerFast
    from safetensors.torch import load_file
    from bridge_evidence.contest.document_ner_materials import build_index, PROMPTS
    from bridge_evidence.contest.run_state import atomic_json

    class Model(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.encoder = ErnieModel(config)
            self.linear_start = nn.Linear(config.hidden_size, 1)
            self.linear_end = nn.Linear(config.hidden_size, 1)

        def forward(self, **kw):
            h = self.encoder(**kw).last_hidden_state
            return self.linear_start(h).squeeze(-1), self.linear_end(h).squeeze(-1)

    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    cfg = ErnieConfig.from_pretrained(args.checkpoint, local_files_only=True)
    cfg._attn_implementation = "eager"
    model = Model(cfg)
    model.load_state_dict(load_file(str(args.checkpoint / "model.safetensors")), strict=True)
    model.to(args.device).eval()
    tokenizer = BertTokenizerFast.from_pretrained(args.checkpoint, local_files_only=True)
    manifest = json.loads(args.manifest.read_bytes())
    units, starts, pieces, offset = [], [], [], 0
    for u in manifest["units"]:
        if not u["text"]:
            continue
        units.append(u)
        starts.append(offset)
        pieces.append(u["text"])
        offset += len(u["text"]) + 1
    raw = "\n".join(pieces)
    windows = [(p, left, raw[left:left+360]) for left in range(0, len(raw), 296) for p in PROMPTS]
    loaded = time.monotonic()
    spans, boundary_spans = set(), 0
    with torch.inference_mode():
        for left in range(0, len(windows), 24):
            rows = windows[left:left+24]
            enc = tokenizer([p for p, _, _ in rows], [t for _, _, t in rows], padding=True,
                truncation="only_second", max_length=512, return_offsets_mapping=True, return_tensors="pt")
            offsets = enc.pop("offset_mapping").tolist()
            sequences = [enc.sequence_ids(i) for i in range(len(rows))]
            a, b = model(**{k: v.to(args.device) for k, v in enc.items()})
            a, b = a.sigmoid().cpu().tolist(), b.sigmoid().cpu().tolist()
            for row_a, row_b, positions, seq, (kind, shift, _) in zip(a, b, offsets, sequences, rows):
                valid = [i for i, s in enumerate(seq) if s == 1 and positions[i][1] > positions[i][0]]
                candidate_starts = [i for i in valid if row_a[i] > .5]
                used = set()
                for end in [i for i in valid if row_b[i] > .5]:
                    ss = [s for s in candidate_starts if s <= end and s not in used]
                    if not ss:
                        continue
                    begin = max(ss)
                    used.add(begin)
                    s, e = shift + positions[begin][0], shift + positions[end][1]
                    pos = bisect.bisect_right(starts, s) - 1
                    if pos < 0 or e > starts[pos] + len(units[pos]["text"]):
                        boundary_spans += 1
                        continue
                    spans.add((units[pos]["id"], kind, s - starts[pos], e - starts[pos]))
    inferred = time.monotonic()
    by_id = {u["id"]: u for u in units}
    mentions = [{"evidence_id": eid, "type": kind, "start": a, "end": b, "text": by_id[eid]["text"][a:b]}
                for eid, kind, a, b in sorted(spans)]
    timing = {"backend": "torch_float32", "device": args.device,
        "model_sha256": hashlib.sha256((args.checkpoint / "model.safetensors").read_bytes()).hexdigest(),
        "load_seconds": loaded - started, "inference_seconds": inferred - loaded,
        "total_seconds": time.monotonic() - started, "windows": len(windows),
        "unbound_cross_unit_mentions_discarded": boundary_spans,
        "gpu_max_allocated_bytes": torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None}
    result = build_index(manifest, mentions, inference=timing)
    atomic_json(args.output, result)
    print(json.dumps({"output": str(args.output), "mentions": len(mentions), "statistics": result["statistics"], "timing": timing}))


if __name__ == "__main__":
    main()
