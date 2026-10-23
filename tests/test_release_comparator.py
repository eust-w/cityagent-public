from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "compare_release_to_v12.py"
SPEC = importlib.util.spec_from_file_location("compare_release_to_v12", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def disease(anchor: str, disease_type: str, description: str) -> dict:
    return {
        "type": disease_type,
        "description": description,
        "evidence": [anchor],
    }


def test_release_gate_allows_only_evidence_audited_inactive_row_prune() -> None:
    name = "sample.docx"
    baseline_rows = [
        ["序号", "病害类型"],
        ["1", "修补"],
        ["2", "破损"],
    ]
    candidate_rows = [
        ["序号", "病害类型"],
        ["1", "破损"],
    ]
    baseline_states = {"sample.json": {"diseases": [
        disease("T2R1", "修补", "梁底修补现状"),
        disease("T2R2", "破损", "梁底破损"),
    ]}}
    candidate_states = {"sample.json": {"diseases": [
        disease("T2R2", "破损", "梁底破损"),
    ]}}

    allowed, detail = MODULE._allowed_inactive_disease_prune(
        name=name,
        table_index=2,
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
        baseline_states=baseline_states,
        candidate_states=candidate_states,
    )

    assert allowed
    assert detail["anchors"] == ["T2R1"]
    # Two removed cells plus one renumbered sequence cell.
    assert MODULE._aligned_row_change_cost(baseline_rows, candidate_rows) == 3


def test_release_gate_rejects_removal_of_active_disease() -> None:
    baseline_rows = [["序号", "病害类型"], ["1", "裂缝"]]
    candidate_rows = [["序号", "病害类型"]]
    allowed, detail = MODULE._allowed_inactive_disease_prune(
        name="sample.docx",
        table_index=2,
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
        baseline_states={"sample.json": {"diseases": [
            disease("T2R1", "裂缝", "梁底纵向裂缝"),
        ]}},
        candidate_states={"sample.json": {"diseases": []}},
    )

    assert not allowed
    assert detail is None
