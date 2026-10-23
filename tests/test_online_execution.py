from __future__ import annotations

# Hand-constructed fixtures use synthetic bridge and organization names.

from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import pytest

from bridge_evidence.agents.client import ClientMetrics, GatewayClient
from bridge_evidence.agents.diseases import DiseaseNormalizer
from bridge_evidence.config import Settings
from bridge_evidence.models.agent import (
    AgentCommand,
    AgentObservation,
    AgentRunRecord,
    ToolDescriptor,
    ToolManifest,
    ToolResult,
    TrajectoryStep,
)
from bridge_evidence.models.schema import Disease, DocumentState, EvidenceRef, FieldValue, Recommendation
from bridge_evidence.models.validation import ConstraintEngine
from bridge_evidence.orchestrator import BridgeEvidenceOrchestrator
from bridge_evidence.parsers.docx import ParagraphBlock, ParsedDocument, TableBlock
from bridge_evidence.runtime.planners import PlannedCommand, QwenPlanner
from bridge_evidence.runtime.graph import CoreAgentLoop
from bridge_evidence.runtime.registry import EvidenceToolRegistry


def test_online_modes_fail_closed_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BRIDGE_EVIDENCE_SEMANTIC_API_KEY", raising=False)
    monkeypatch.delenv("BRIDGE_EVIDENCE_API_KEY", raising=False)

    assert Settings(llm_mode="off").llm_enabled is False
    with pytest.raises(RuntimeError, match="refusing to silently downgrade"):
        _ = Settings(llm_mode="auto").llm_enabled
    with pytest.raises(RuntimeError, match="refusing to silently downgrade"):
        _ = Settings(llm_mode="on").llm_enabled

    monkeypatch.setenv("BRIDGE_EVIDENCE_API_KEY", "test-only-placeholder")
    with pytest.raises(RuntimeError, match="BRIDGE_EVIDENCE_SEMANTIC_API_KEY"):
        _ = Settings(llm_mode="on").llm_enabled
    monkeypatch.setenv("BRIDGE_EVIDENCE_SEMANTIC_API_KEY", "test-only-semantic-placeholder")
    assert Settings(llm_mode="auto").llm_enabled is True
    assert Settings(llm_mode="on").llm_enabled is True


def test_formal_mode_fails_closed_without_usable_history(tmp_path: Path) -> None:
    settings = Settings(
        llm_mode="off",
        require_history=True,
        cache_dir=tmp_path / "cache",
        work_dir=tmp_path / "work",
    )
    with pytest.raises(ValueError, match="未提供 --history-label-dir"):
        BridgeEvidenceOrchestrator(settings)

    empty_history = tmp_path / "history"
    empty_history.mkdir()
    settings.history_label_dir = empty_history
    with pytest.raises(ValueError, match="没有可用的历年标签"):
        BridgeEvidenceOrchestrator(settings)


def test_gateway_metrics_separate_network_tasks_and_cache_hits() -> None:
    client = object.__new__(GatewayClient)
    client.metrics = ClientMetrics()
    client._metrics_lock = Lock()
    completion = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3))

    client._record_completion(
        completion, kind="chat", task="react-plan", model="deepseek/deepseek-v4-pro"
    )
    client._record_completion(
        completion, kind="embedding", task="summary-evidence-embedding", model="Qwen3-VL-Embedding-8B"
    )
    client._record_request(
        kind="reranker", task="summary-evidence-rerank", model="Qwen3-VL-Reranker-8B"
    )
    client._record_cache_hit("summary-review", "deepseek/deepseek-v4-pro")

    assert client.metrics.calls == 3
    assert client.metrics.chat_calls == 1
    assert client.metrics.embedding_calls == 1
    assert client.metrics.reranker_calls == 1
    assert client.metrics.planner_calls == 1
    assert client.metrics.cache_hits == 1
    assert client.metrics.task_calls == {
        "react-plan": 1,
        "summary-evidence-embedding": 1,
        "summary-evidence-rerank": 1,
    }
    assert client.metrics.task_cache_hits == {"summary-review": 1}
    assert client.metrics.model_calls == {
        "deepseek/deepseek-v4-pro": 1,
        "Qwen3-VL-Embedding-8B": 1,
        "Qwen3-VL-Reranker-8B": 1,
    }
    assert client.metrics.model_cache_hits == {"deepseek/deepseek-v4-pro": 1}


def test_kimi_typed_completion_accepts_fenced_json() -> None:
    class _Cache(dict):
        def set(self, key, value):
            self[key] = value

    completions = iter([
        SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"unexpected":true}'))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4),
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='```json\n{"action":"finalize"}\n```'))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4),
        ),
    ])
    create = lambda **_: next(completions)
    client = object.__new__(GatewayClient)
    client.settings = Settings(
        llm_mode="off",
        task_models={"react-plan": "kimi-k2.6"},
    )
    client.cache = _Cache()
    client.raw = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    client.typed = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create_with_completion=lambda **_: (_ for _ in ()).throw(
                    AssertionError("Kimi must bypass Instructor fence parsing")
                )
            )
        )
    )
    client.metrics = ClientMetrics()
    client._metrics_lock = Lock()

    result = client.complete_model(
        PlannedCommand,
        system="system",
        user_payload={"task": "finish"},
        temperature=0.0,
        task="react-plan",
    )

    assert result.action == "finalize"
    assert client.metrics.task_calls == {"react-plan": 2}
    assert client.metrics.task_models == {"react-plan": "kimi-k2.6"}
    assert client.metrics.retries == 1


def test_qwen_typed_completion_disables_thinking_and_bounds_output() -> None:
    class _Cache(dict):
        def set(self, key, value):
            self[key] = value

    captured: dict[str, object] = {}

    def create_with_completion(**kwargs: object):
        captured.update(kwargs)
        return (
            PlannedCommand(action="finalize"),
            SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4)),
        )

    client = object.__new__(GatewayClient)
    client.settings = Settings(
        llm_mode="off",
        task_models={"detail-review": "qwen3.6-27b"},
    )
    client.cache = _Cache()
    client.typed = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create_with_completion=create_with_completion)
        )
    )
    client.metrics = ClientMetrics()
    client._metrics_lock = Lock()

    result = client.complete_model(
        PlannedCommand,
        system="system",
        user_payload={"task": "review"},
        temperature=0.0,
        task="detail-review",
    )

    assert result.action == "finalize"
    assert captured["max_tokens"] == 4096
    assert captured["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_online_loop_can_use_deterministic_planner_with_model_tools() -> None:
    loop = CoreAgentLoop(
        Settings(llm_mode="off", planner_mode="deterministic"),
        ConstraintEngine(),
        client=object(),
    )

    assert loop.planner.name == "deterministic"
    assert loop.client is not None


class _PlannerClient:
    def __init__(self, action: str):
        self.action = action
        self.calls = 0

    def complete_model(self, response_model, **_: object):
        self.calls += 1
        return response_model(action=self.action, arguments={}, rationale="test")


def test_qwen_planner_is_called_for_mandatory_actions() -> None:
    client = _PlannerClient("review_summary")
    planner = QwenPlanner(client, Settings(llm_mode="off"), "system")
    observation = AgentObservation(required_actions=[{"action": "review_summary", "arguments": {}, "reason": "required"}])
    manifest = ToolManifest(tools=[ToolDescriptor(name="review_summary", description="review")])

    command = planner.plan(run_id="run:test", seq=1, observation=observation, manifest=manifest, trajectory=[])

    assert client.calls == 1
    assert command.action == "review_summary"


def test_qwen_planner_policy_gate_cannot_bypass_mandatory_actions() -> None:
    client = _PlannerClient("finalize")
    planner = QwenPlanner(client, Settings(llm_mode="off"), "system")
    observation = AgentObservation(required_actions=[{"action": "review_summary", "arguments": {}, "reason": "required"}])
    manifest = ToolManifest(tools=[
        ToolDescriptor(name="review_summary", description="review"),
        ToolDescriptor(name="finalize", description="end"),
    ])

    command = planner.plan(
        run_id="run:test",
        seq=1,
        observation=observation,
        manifest=manifest,
        trajectory=[],
    )

    assert command.action == "review_summary"
    assert "策略门纠正" in command.rationale


class _SemanticClient:
    def __init__(self):
        self.tasks: list[str] = []

    def embed_texts(self, texts: list[str], *, task: str):
        self.tasks.append(task)
        return [[1.0, float(index % 2)] for index, _ in enumerate(texts)]

    def rerank_documents(self, *, query: str, documents: list[str], top_n: int, task: str):
        self.tasks.append(task)
        return [{"index": 0, "score": 0.99, "document": documents[0]}]

    def complete_model(self, response_model, *, task: str, **_: object):
        self.tasks.append(task)
        return response_model(patches=[], warnings=[])


class _CapturingSemanticClient(_SemanticClient):
    def __init__(self) -> None:
        super().__init__()
        self.summary_payload: dict | None = None

    def complete_model(self, response_model, *, task: str, **kwargs: object):
        self.tasks.append(task)
        if task == "summary-review":
            self.summary_payload = kwargs["user_payload"]  # type: ignore[assignment]
        return response_model(patches=[], warnings=[])


class _ReflectingSummaryClient(_SemanticClient):
    def complete_model(self, response_model, *, task: str, **_: object):
        self.tasks.append(task)
        if task == "summary-review":
            return response_model(
                patches=[{
                    "field": "scores.substructure.score",
                    "value": "97.44（左幅）/97.44（右幅）",
                    "evidence_ids": ["P0"],
                    "reason": "primary chose a contradictory control value",
                }],
                warnings=[],
            )
        if task == "summary-conflict-review":
            return response_model(
                patches=[{
                    "field": "scores.substructure.score",
                    "value": "98.4（左幅）/98.3（右幅）",
                    "evidence_ids": ["T0R1", "T1R1"],
                    "reason": "reflection used final side-specific rating tables",
                }],
                warnings=[],
            )
        return response_model(patches=[], warnings=[])


def test_score_candidate_tool_exposes_bci_and_bsi_without_deciding() -> None:
    text = "桥面系 BCIm=85.48，BSIm=51.61，桥面系评定为D级。"
    state = DocumentState(document_id="sha256:score-conflict", source_path="score-conflict.docx")
    parsed = ParsedDocument(
        path=Path("score-conflict.docx"),
        paragraphs=[ParagraphBlock(index=0, text=text, style="正文")],
        tables=[],
    )
    registry = EvidenceToolRegistry(state, parsed, ConstraintEngine())

    result = registry.execute(AgentCommand(
        run_id="run:score-conflict",
        seq=1,
        action="list_score_candidates",
        arguments={"scope": "deck"},
    ))

    assert result.success, result.error
    assert result.content["candidate_metric_kinds"] == ["BCIm", "BSIm"]
    assert result.content["distinct_values"] == [51.61, 85.48]
    assert result.content["semantic_conflict"] is True
    by_metric = {item["metric"]: item for item in result.content["candidates"]}
    assert by_metric["BCIm"]["grade_binding"] is False
    assert by_metric["BCIm"]["grade"] is None
    assert by_metric["BSIm"]["grade_binding"] is True
    assert by_metric["BSIm"]["grade"] == "D"


def test_score_candidate_tool_keeps_left_and_right_clauses_distinct() -> None:
    text = (
        "左幅上部结构BCIs=81.36，BSIs=81.36，评定为B级；"
        "右幅上部结构BCIs=90.0，BSIs=90.0，评定为A级。"
    )
    state = DocumentState(document_id="sha256:sides", source_path="sides.docx")
    parsed = ParsedDocument(
        path=Path("sides.docx"),
        paragraphs=[ParagraphBlock(index=0, text=text, style="正文")],
        tables=[],
    )
    registry = EvidenceToolRegistry(state, parsed, ConstraintEngine())

    result = registry.execute(AgentCommand(
        run_id="run:sides",
        seq=1,
        action="list_score_candidates",
        arguments={"scope": "superstructure"},
    ))

    bci_rows = [item for item in result.content["candidates"] if item["metric"] == "BCIs"]
    assert [(item["side"], item["score"], item["grade"], item["grade_binding"]) for item in bci_rows] == [
        ("左幅", 81.36, None, False),
        ("右幅", 90.0, None, False),
    ]
    bsi_rows = [item for item in result.content["candidates"] if item["metric"] == "BSIs"]
    assert [(item["side"], item["grade"], item["grade_binding"]) for item in bsi_rows] == [
        ("左幅", "B", True),
        ("右幅", "A", True),
    ]


def test_score_candidate_tool_reads_final_rating_table_without_bci_token() -> None:
    paragraph = "左幅下部结构BSIx=min(BCIxj)=97.44，评定为A级；右幅下部结构BSIx=min(BCIxj)=97.44，评定为B级。"
    state = DocumentState(document_id="sha256:rating-table", source_path="rating-table.docx")
    parsed = ParsedDocument(
        path=Path("rating-table.docx"),
        paragraphs=[ParagraphBlock(index=0, text=paragraph, style="正文")],
        tables=[
            TableBlock(index=0, rows=[["部位", "评分"], ["下部结构", "98.4", "0.45"]], context=["左幅技术状况评定表"]),
            TableBlock(index=1, rows=[["部位", "评分"], ["下部结构", "98.3", "0.45"]], context=["右幅技术状况评定表"]),
        ],
    )
    registry = EvidenceToolRegistry(state, parsed, ConstraintEngine())

    result = registry.execute(AgentCommand(
        run_id="run:rating-table",
        seq=1,
        action="list_score_candidates",
        arguments={"scope": "substructure"},
    ))

    rows = result.content["candidates"]
    bsi_rows = [item for item in rows if item["metric"] == "BSIx" and item["score"] == 97.44]
    assert bsi_rows
    assert all(item["metric_role"] == "control_index" for item in bsi_rows)
    assert all(item["self_contradictory"] is True for item in bsi_rows)
    assert {(item["side"], item["score"]) for item in rows if item["metric"] == "BCIx"} == {
        ("左幅", 98.4),
        ("右幅", 98.3),
    }
    assert all(
        item["source_role"] == "final_component_rating_table"
        for item in rows
        if item["metric"] == "BCIx"
    )


def test_score_candidate_tool_binds_parallel_published_metrics_by_position() -> None:
    text = (
        "BSIm、BSIs、BSIx分别为76.36(C)、80(B)、100(A)。"
        "合成平面立交1号桥本次检测完成。"
    )
    state = DocumentState(document_id="sha256:parallel-metrics", source_path="parallel.docx")
    parsed = ParsedDocument(
        path=Path("parallel.docx"),
        paragraphs=[ParagraphBlock(index=0, text=text, style="正文")],
        tables=[],
    )
    registry = EvidenceToolRegistry(state, parsed, ConstraintEngine())

    result = registry.execute(AgentCommand(
        run_id="run:parallel-metrics",
        seq=1,
        action="list_score_candidates",
        arguments={"scope": "substructure"},
    ))

    rows = [item for item in result.content["candidates"] if item["metric"] == "BSIx"]
    assert [(item["score"], item["grade"], item["grade_binding"]) for item in rows] == [
        (100.0, "A", True),
    ]


def test_score_candidate_tool_binds_parallel_grades_with_level_suffix() -> None:
    text = "BSIm、BSIs、BSIx分别为64.38（D级）、62.72（D级）、88（B级）。"
    state = DocumentState(document_id="sha256:parallel-grade", source_path="parallel-grade.docx")
    parsed = ParsedDocument(
        path=Path("parallel-grade.docx"),
        paragraphs=[ParagraphBlock(index=0, text=text, style="正文")],
        tables=[],
    )
    registry = EvidenceToolRegistry(state, parsed, ConstraintEngine())

    expected = {
        "deck": (64.38, "D"),
        "superstructure": (62.72, "D"),
        "substructure": (88.0, "B"),
    }
    for seq, (scope, pair) in enumerate(expected.items(), 1):
        result = registry.execute(AgentCommand(
            run_id="run:parallel-grade",
            seq=seq,
            action="list_score_candidates",
            arguments={"scope": scope},
        ))
        rows = [item for item in result.content["candidates"] if item["metric"].upper().startswith("BSI")]
        assert [(item["score"], item["grade"], item["grade_binding"]) for item in rows] == [
            (pair[0], pair[1], True),
        ]


def test_control_grade_ignores_bsi_row_that_mirrors_all_component_bci_values() -> None:
    state = DocumentState(document_id="sha256:mirrored-bsi", source_path="mirrored-bsi.docx")
    state.scores.deck.grade = FieldValue(value="B级", confidence=0.95, source_priority=97)
    state.scores.superstructure.grade = FieldValue(value="B级", confidence=0.95, source_priority=97)
    state.scores.substructure.grade = FieldValue(value="A级", confidence=0.95, source_priority=97)
    parsed = ParsedDocument(path=Path("mirrored-bsi.docx"), paragraphs=[], tables=[])
    registry = EvidenceToolRegistry(state, parsed, ConstraintEngine())

    def final_component(evidence_id: str, score: float) -> dict:
        return {
            "evidence_id": evidence_id,
            "score": score,
            "side": None,
            "source_role": "final_component_rating_table",
            "self_contradictory": False,
        }

    def control(evidence_id: str, score: float, grade: str) -> dict:
        return {
            "evidence_id": evidence_id,
            "score": score,
            "grade": grade,
            "grade_binding": True,
            "side": None,
            "metric_role": "control_index",
            "source_role": "explicit_assessment_sentence",
            "self_contradictory": False,
        }

    catalog = {
        "deck": [
            final_component("Tfinal-deck", 80.50),
            control("Pbad", 80.50, "B"),
            control("Pdeck", 60.00, "D"),
        ],
        "superstructure": [
            final_component("Tfinal-super", 80.17),
            control("Pbad", 80.17, "B"),
            control("Psuper", 66.68, "C"),
        ],
        "substructure": [
            final_component("Tfinal-sub", 98.88),
            control("Pbad", 98.88, "A"),
            control("Psub", 97.75, "A"),
        ],
    }

    mirrored = registry._bsi_mirrors_component_summary_ids(catalog)
    patches = registry._unambiguous_control_grade_patches(
        catalog,
        excluded_evidence_ids=mirrored,
    )

    assert mirrored == {"Pbad"}
    assert {(item["field"], item["value"]) for item in patches} == {
        ("scores.deck.grade", "D级"),
        ("scores.superstructure.grade", "C级"),
    }


def test_independent_control_consensus_beats_duplicated_opening_abstract() -> None:
    opening = "BSIm为100（A级），BSIs为73.83（C级），BSIx为97.75（A级）。"
    state = DocumentState(document_id="sha256:control-consensus", source_path="consensus.docx")
    state.scores.deck.grade = FieldValue(value="A级", confidence=0.99, source_priority=90)
    for evidence_id, text, kind in (
        ("Popen", opening, "paragraph"),
        ("T0R1", f"检测结论 | {opening}", "table"),
        ("Pcalc", "经计算BSIm=60.00，桥面系评定为D级。", "paragraph"),
        ("Plate", "最终结论：BSIm为60.00（D级）。", "paragraph"),
    ):
        state.add_evidence(EvidenceRef(
            id=evidence_id,
            kind=kind,
            text=text,
            source_path="consensus.docx",
            priority=90,
        ))
    parsed = ParsedDocument(
        path=Path("consensus.docx"),
        paragraphs=[],
        tables=[TableBlock(index=0, rows=[["项目", "内容"], ["检测结论", opening]])],
    )
    registry = EvidenceToolRegistry(state, parsed, ConstraintEngine())

    def control(evidence_id: str, score: float, grade: str, text: str) -> dict:
        return {
            "evidence_id": evidence_id,
            "score": score,
            "grade": grade,
            "grade_binding": True,
            "side": None,
            "metric_role": "control_index",
            "source_role": "explicit_assessment_sentence",
            "self_contradictory": False,
            "text": text,
        }

    catalog = {"deck": [
        control("Popen", 100.0, "A", opening),
        control("T0R1", 100.0, "A", f"检测结论 | {opening}"),
        control("Pcalc", 60.0, "D", "经计算BSIm=60.00，桥面系评定为D级。"),
        control("Plate", 60.0, "D", "最终结论：BSIm为60.00（D级）。"),
    ]}

    patches = registry._unambiguous_control_grade_patches(
        catalog,
        excluded_evidence_ids=set(),
    )
    published = registry._published_summary_metric_patches(catalog)

    assert [(item["field"], item["value"]) for item in patches] == [
        ("scores.deck.grade", "D级"),
    ]
    assert published == []


def test_control_grade_can_agree_when_control_scores_differ() -> None:
    state = DocumentState(document_id="sha256:grade-only-consensus", source_path="grade-only.docx")
    state.scores.deck.grade = FieldValue(value="A级")
    registry = EvidenceToolRegistry(
        state,
        ParsedDocument(path=Path("grade-only.docx"), paragraphs=[], tables=[]),
        ConstraintEngine(),
    )
    catalog = {"deck": [
        {
            "evidence_id": "Pabstract",
            "score": 85.0,
            "grade": "B",
            "grade_binding": True,
            "side": None,
            "metric_role": "control_index",
            "source_role": "explicit_assessment_sentence",
            "self_contradictory": False,
        },
        {
            "evidence_id": "Pcalculation",
            "score": 80.0,
            "grade": "B",
            "grade_binding": True,
            "side": None,
            "metric_role": "control_index",
            "source_role": "explicit_assessment_sentence",
            "self_contradictory": False,
        },
    ]}

    patches = registry._unambiguous_control_grade_patches(
        catalog,
        excluded_evidence_ids=set(),
    )

    assert [(item["field"], item["value"]) for item in patches] == [
        ("scores.deck.grade", "B级"),
    ]


def test_published_summary_patch_only_repairs_gaps_or_invalid_scores() -> None:
    stable = FieldValue(value=89.0, confidence=0.99, source_priority=96)
    missing = FieldValue(value="无", confidence=0.2, source_priority=0)
    parser_failure = FieldValue(value=1.0, confidence=0.99, source_priority=99)

    assert not EvidenceToolRegistry._published_summary_patch_repairs_gap(stable, 100.0)
    assert EvidenceToolRegistry._published_summary_patch_repairs_gap(missing, 100.0)
    assert EvidenceToolRegistry._published_summary_patch_repairs_gap(parser_failure, 100.0)


def test_published_summary_patch_keeps_bci_score_but_allows_control_grade() -> None:
    state = DocumentState(document_id="sha256:published-bsi", source_path="published-bsi.docx")
    state.scores.superstructure.grade = FieldValue(value="B级", confidence=0.94, source_priority=91)
    registry = EvidenceToolRegistry(
        state,
        ParsedDocument(
            path=Path("published-bsi.docx"),
            paragraphs=[],
            tables=[TableBlock(
                index=0,
                rows=[
                    ["BCI", "[90,100]", "[80,90)"],
                    ["等级", "A", "B"],
                ],
            )],
        ),
        ConstraintEngine(),
    )
    score = FieldValue(value=86.1, confidence=0.99, source_priority=97)
    grade = FieldValue(value="B级", confidence=0.94, source_priority=91)

    assert not registry._published_summary_patch_allowed(
        "scores.superstructure.score", score, 62.72,
    )
    assert registry._published_summary_patch_allowed(
        "scores.superstructure.grade", grade, "D级",
    )


def test_published_summary_patch_never_replaces_stable_bci_due_to_control_grade() -> None:
    state = DocumentState(document_id="sha256:published-bsi-control", source_path="published-bsi-control.docx")
    state.scores.deck.grade = FieldValue(value="D级", confidence=0.99, source_priority=99)
    registry = EvidenceToolRegistry(
        state,
        ParsedDocument(
            path=Path("published-bsi-control.docx"),
            paragraphs=[],
            tables=[TableBlock(
                index=0,
                rows=[
                    ["BCI", "[90,100]", "[80,90)", "[66,80)"],
                    ["等级", "A", "B", "D"],
                ],
            )],
        ),
        ConstraintEngine(),
    )

    assert not registry._published_summary_patch_allowed(
        "scores.deck.score",
        FieldValue(value=96.27, confidence=0.99, source_priority=97),
        "62.72（左幅）/62.72（右幅）",
    )


def test_score_candidate_tool_binds_only_component_grade_column() -> None:
    state = DocumentState(document_id="sha256:component-grade", source_path="component-grade.docx")
    parsed = ParsedDocument(
        path=Path("component-grade.docx"),
        paragraphs=[],
        tables=[
            TableBlock(
                index=0,
                rows=[
                    ["部位名称", "技术状况指数", "技术状况", "权重", "BCI", "桥梁整体技术状况等级"],
                    ["桥面系", "85.0", "B", "0.15", "86.0", "B"],
                ],
                context=["技术状况评定表"],
            ),
            TableBlock(
                index=1,
                rows=[
                    ["部位名称", "技术状况指数", "权重", "BCI", "桥梁整体技术状况等级"],
                    ["桥面系", "89.0", "0.15", "83.0", "B"],
                ],
                context=["技术状况评定表"],
            ),
        ],
    )
    registry = EvidenceToolRegistry(state, parsed, ConstraintEngine())

    result = registry.execute(AgentCommand(
        run_id="run:component-grade",
        seq=1,
        action="list_score_candidates",
        arguments={"scope": "deck"},
    ))

    final_rows = [
        item for item in result.content["candidates"]
        if item["source_role"] == "final_component_rating_table"
    ]
    assert [(item["score"], item["grade"], item["grade_binding"]) for item in final_rows] == [
        (85.0, "B", True),
        (89.0, None, False),
    ]


def test_summary_reviewer_receives_score_conflicts_as_decision_context() -> None:
    text = "桥面系 BCIm=85.48，BSIm=51.61，桥面系评定为D级。"
    state = DocumentState(document_id="sha256:review-conflict", source_path="review-conflict.docx")
    parsed = ParsedDocument(
        path=Path("review-conflict.docx"),
        paragraphs=[ParagraphBlock(index=0, text=text, style="正文")],
        tables=[],
    )
    client = _CapturingSemanticClient()
    registry = EvidenceToolRegistry(
        state,
        parsed,
        ConstraintEngine(),
        client=client,
        settings=Settings(llm_mode="off"),
    )

    result = registry.execute(AgentCommand(run_id="run:review-conflict", seq=1, action="review_summary"))

    assert result.success, result.error
    assert result.content["ambiguous_scopes"] == ["deck"]
    assert client.summary_payload is not None
    context = client.summary_payload["decision_context"]
    assert context["ambiguous_scopes"] == ["deck"]
    assert {item["metric"] for item in context["score_candidates"]["deck"]} == {"BCIm", "BSIm"}


def test_summary_reviewer_receives_report_local_rating_scale() -> None:
    text = "左幅下部结构BSIx=97.44，评定为A级；右幅下部结构BSIx=97.44，评定为B级。"
    state = DocumentState(document_id="sha256:scale-context", source_path="scale-context.docx")
    parsed = ParsedDocument(
        path=Path("scale-context.docx"),
        paragraphs=[ParagraphBlock(index=0, text=text, style="正文")],
        tables=[
            TableBlock(
                index=0,
                rows=[
                    ["BCI", "[90，100]", "[80，90）", "[66，80）", "[50，66）", "[0，50）"],
                    ["评估等级", "A", "B", "C", "D", "E"],
                ],
                context=["桥梁完好状况评估标准"],
            ),
        ],
    )
    client = _CapturingSemanticClient()
    registry = EvidenceToolRegistry(
        state,
        parsed,
        ConstraintEngine(),
        client=client,
        settings=Settings(llm_mode="off"),
    )

    result = registry.execute(AgentCommand(run_id="run:scale-context", seq=1, action="review_summary"))

    assert result.success, result.error
    assert client.summary_payload is not None
    context = client.summary_payload["decision_context"]
    assert context["report_local_rating_scales"] == [{
        "table_id": 0,
        "evidence_ids": ["T0R0", "T0R1"],
        "text": "BCI | [90，100] | [80，90） | [66，80） | [50，66） | [0，50） || 评估等级 | A | B | C | D | E",
    }]
    rows = context["score_candidates"]["substructure"]
    assert all(item["self_contradictory"] is True for item in rows)


def test_summary_review_reflects_after_contradictory_patch_is_rejected() -> None:
    paragraph = "左幅下部结构BSIx=97.44，评定为A级；右幅下部结构BSIx=97.44，评定为B级。"
    state = DocumentState(document_id="sha256:reflection", source_path="reflection.docx")
    state.scores.substructure.score = FieldValue(value=98.4, source_priority=93, confidence=0.99)
    parsed = ParsedDocument(
        path=Path("reflection.docx"),
        paragraphs=[ParagraphBlock(index=0, text=paragraph, style="正文")],
        tables=[
            TableBlock(index=0, rows=[["部位", "BCI"], ["下部结构", "98.4"]], context=["左幅技术状况评定表"]),
            TableBlock(index=1, rows=[["部位", "BCI"], ["下部结构", "98.3"]], context=["右幅技术状况评定表"]),
        ],
    )
    client = _ReflectingSummaryClient()
    registry = EvidenceToolRegistry(
        state,
        parsed,
        ConstraintEngine(),
        client=client,
        settings=Settings(llm_mode="off"),
    )

    result = registry.execute(AgentCommand(run_id="run:reflection", seq=1, action="review_summary"))

    assert result.success, result.error
    assert result.content["reflection_used"] is True
    assert result.content["applied"] == ["scores.substructure.score"]
    assert state.scores.substructure.score.value == "98.4（左幅）/98.3（右幅）"
    assert "summary-conflict-review" in client.tasks


def test_semantic_conflict_allows_only_near_priority_model_arbitration() -> None:
    target = FieldValue(value=96.27, source_priority=93, confidence=0.99)

    assert not EvidenceToolRegistry._fact_override_allowed(target, 92, semantic_conflict=True)
    assert not EvidenceToolRegistry._fact_override_allowed(target, 92)
    assert not EvidenceToolRegistry._fact_override_allowed(target, 70, semantic_conflict=True)


def test_fact_patch_preserves_semantic_conflict_context_at_write_gate() -> None:
    state = DocumentState(document_id="sha256:write-gate", source_path="write-gate.docx")
    state.scores.deck.score = FieldValue(value=96.27, source_priority=93, confidence=0.99)
    state.add_evidence(EvidenceRef(
        id="P1",
        kind="paragraph",
        text="左幅桥面系BSIm=62.72，评定为D级；右幅桥面系BSIm=62.72，评定为D级。",
        source_path="write-gate.docx",
        priority=92,
        paragraph_index=1,
    ))
    registry = EvidenceToolRegistry(
        state,
        ParsedDocument(path=Path("write-gate.docx"), paragraphs=[], tables=[]),
        ConstraintEngine(),
    )

    with pytest.raises(ValueError, match="现有高置信事实证据不弱于候选"):
        registry._apply_fact_patches(
            [{
                "field": "scores.deck.score",
                "value": "62.72（左幅）/62.72（右幅）",
                "evidence_ids": ["P1"],
            }],
            semantic_conflict_fields={"scores.deck.score"},
        )

    assert state.scores.deck.score.value == 96.27


def test_contradictory_evidence_cannot_be_the_only_patch_support() -> None:
    state = DocumentState(document_id="sha256:contradiction-gate", source_path="contradiction-gate.docx")
    state.add_evidence(EvidenceRef(
        id="P1",
        kind="paragraph",
        text="左幅BSIx=97.44，A级；右幅BSIx=97.44，B级。",
        source_path="contradiction-gate.docx",
        priority=92,
        paragraph_index=1,
    ))
    state.add_evidence(EvidenceRef(
        id="T2R3",
        kind="table",
        text="右幅技术状况评定表 | 下部结构 | 98.3",
        source_path="contradiction-gate.docx",
        priority=90,
        table_index=2,
        row_index=3,
    ))
    registry = EvidenceToolRegistry(
        state,
        ParsedDocument(path=Path("contradiction-gate.docx"), paragraphs=[], tables=[]),
        ConstraintEngine(),
    )

    assert not registry._has_noncontradictory_support(97.44, ["P1"], {"P1"})
    assert not registry._has_noncontradictory_support(97.44, ["P1", "T2R3"], {"P1"})
    assert registry._has_noncontradictory_support(98.3, ["P1", "T2R3"], {"P1"})


def test_grade_write_gate_does_not_treat_b_in_bci_as_b_grade() -> None:
    assert not EvidenceToolRegistry._value_supported("B级", "部位 | BCI | 技术状况指数 85.0")
    assert EvidenceToolRegistry._value_supported("B级", "上部结构评定为B级")


def test_structural_side_completion_preserves_selected_final_table_metric() -> None:
    state = DocumentState(document_id="sha256:shape", source_path="shape.docx")
    state.scores.substructure.score = FieldValue(value=98.4)
    state.scores.deck.score = FieldValue(value=96.27)
    registry = EvidenceToolRegistry(
        state,
        ParsedDocument(path=Path("shape.docx"), paragraphs=[], tables=[]),
        ConstraintEngine(),
    )
    final_sub = [
        {"source_role": "final_component_rating_table", "metric_role": "component_index", "score": 98.4, "side": "左幅", "self_contradictory": False, "evidence_id": "T0R1"},
        {"source_role": "final_component_rating_table", "metric_role": "component_index", "score": 98.3, "side": "右幅", "self_contradictory": False, "evidence_id": "T1R1"},
        {"source_role": "explicit_assessment_sentence", "metric_role": "control_index", "score": 97.44, "side": "左幅", "self_contradictory": True, "evidence_id": "P0"},
    ]
    final_deck = [
        {"source_role": "final_component_rating_table", "metric_role": "component_index", "score": 96.27, "side": "左幅", "self_contradictory": False, "evidence_id": "T2R1"},
        {"source_role": "final_component_rating_table", "metric_role": "component_index", "score": 96.27, "side": "右幅", "self_contradictory": False, "evidence_id": "T3R1"},
        {"source_role": "explicit_assessment_sentence", "metric_role": "control_index", "score": 62.72, "side": "左幅", "self_contradictory": False, "evidence_id": "P2"},
    ]

    patches = registry._structural_side_completion_patches(
        {"substructure": final_sub, "deck": final_deck},
        ["scores.substructure.score", "scores.deck.score"],
    )

    assert patches == [
        {
            "field": "scores.substructure.score",
            "value": "98.4（左幅）/98.3（右幅）",
            "evidence_ids": ["T0R1", "T1R1"],
        },
        {
            "field": "scores.deck.score",
            "value": "96.27（左幅）/96.27（右幅）",
            "evidence_ids": ["T2R1", "T3R1"],
        },
    ]


def test_structural_completion_does_not_change_metric_role_to_control() -> None:
    state = DocumentState(document_id="sha256:control-pair", source_path="control-pair.docx")
    state.scores.deck.score = FieldValue(value="96.27（左幅）/96.27（右幅）")
    state.scores.deck.grade = FieldValue(value="D级")
    registry = EvidenceToolRegistry(
        state,
        ParsedDocument(path=Path("control-pair.docx"), paragraphs=[], tables=[]),
        ConstraintEngine(),
    )
    candidates = [
        {"metric_role": "control_index", "score": 62.72, "grade": "D", "grade_binding": True, "side": "左幅", "self_contradictory": False, "evidence_id": "P1"},
        {"metric_role": "control_index", "score": 62.72, "grade": "D", "grade_binding": True, "side": "右幅", "self_contradictory": False, "evidence_id": "P1"},
    ]

    patches = registry._structural_side_completion_patches(
        {"deck": candidates},
        ["scores.deck.score"],
    )

    assert patches == []


def test_metric_consistency_uses_report_scale_and_preserves_one_metric_role() -> None:
    state = DocumentState(document_id="sha256:metric-consistency", source_path="metric-consistency.docx")
    parsed = ParsedDocument(
        path=Path("metric-consistency.docx"),
        paragraphs=[],
        tables=[TableBlock(
            index=0,
            rows=[
                ["BCI", "[90，100]", "[80，90）", "[66，80）", "[50，66）", "[0，50）"],
                ["评估等级", "A", "B", "C", "D", "E"],
            ],
        )],
    )
    registry = EvidenceToolRegistry(state, parsed, ConstraintEngine())
    catalog = {
        "deck": [
            {"source_role": "final_component_rating_table", "metric_role": "component_index", "score": 91.95, "grade": "A", "side": "右幅", "self_contradictory": False, "evidence_id": "T1"},
            {"source_role": "final_component_rating_table", "metric_role": "component_index", "score": 88.65, "grade": "B", "side": "左幅", "self_contradictory": False, "evidence_id": "T2"},
            {"metric_role": "control_index", "score": 85.0, "grade": "B", "grade_binding": True, "side": "右幅", "self_contradictory": False, "evidence_id": "P1"},
            {"metric_role": "control_index", "score": 85.0, "grade": "B", "grade_binding": True, "side": "左幅", "self_contradictory": False, "evidence_id": "P1"},
        ],
        "superstructure": [
            {"source_role": "final_component_rating_table", "metric_role": "component_index", "score": 77.97, "grade": "C", "side": "右幅", "self_contradictory": False, "evidence_id": "T1"},
            {"source_role": "final_component_rating_table", "metric_role": "component_index", "score": 88.82, "grade": "B", "side": "左幅", "self_contradictory": False, "evidence_id": "T2"},
            {"metric_role": "control_index", "score": 77.78, "grade": "C", "grade_binding": True, "side": "右幅", "self_contradictory": False, "evidence_id": "P2"},
            {"metric_role": "control_index", "score": 81.36, "grade": "B", "grade_binding": True, "side": "左幅", "self_contradictory": False, "evidence_id": "P2"},
        ],
        "substructure": [
            {"source_role": "final_component_rating_table", "metric_role": "component_index", "score": 97.28, "grade": None, "side": None, "self_contradictory": False, "evidence_id": "T3"},
            {"metric_role": "control_index", "score": 92.44, "grade": "A", "grade_binding": True, "side": None, "self_contradictory": False, "evidence_id": "P3"},
        ],
    }

    patches = registry._metric_semantic_consistency_patches(catalog)
    by_field = {item["field"]: item["value"] for item in patches}

    assert by_field["scores.deck.score"] == "85.0（左幅）/85.0（右幅）"
    assert by_field["scores.deck.grade"] == "B级（左幅）/B级（右幅）"
    assert by_field["scores.superstructure.score"] == "88.82（左幅）/77.97（右幅）"
    assert by_field["scores.superstructure.grade"] == "B级（左幅）/C级（右幅）"
    assert by_field["scores.substructure.score"] == 97.28
    assert by_field["scores.substructure.grade"] == "A级"


def test_selected_component_grade_follows_report_local_scale_only_for_selected_metric() -> None:
    state = DocumentState(document_id="sha256:selected-component-grade", source_path="selected-component-grade.docx")
    state.scores.deck.score = FieldValue(value=89.79)
    state.scores.deck.grade = FieldValue(value="C级")
    parsed = ParsedDocument(
        path=Path("selected-component-grade.docx"),
        paragraphs=[],
        tables=[
            TableBlock(
                index=0,
                rows=[
                    ["部位名称", "技术状况指数", "权重", "BCI", "桥梁整体技术状况等级"],
                    ["桥面系", "89.79", "0.15", "83.9", "B"],
                ],
                context=["技术状况评定表"],
            ),
            TableBlock(
                index=1,
                rows=[
                    ["BCI", "[90，100]", "[80，90）", "[66，80）", "[50，66）", "[0，50）"],
                    ["评估等级", "A", "B", "C", "D", "E"],
                ],
            ),
        ],
    )
    registry = EvidenceToolRegistry(state, parsed, ConstraintEngine())
    registry._ensure_table_evidence(0, 1, "桥面系 | 89.79 | 0.15 | 83.9 | B", priority=90)
    catalog = {"deck": [{
        "source_role": "final_component_rating_table",
        "score": 89.79,
        "grade": None,
        "side": None,
        "self_contradictory": False,
        "evidence_id": "T0R1",
    }]}

    patches = registry._selected_component_grade_patches(catalog)

    assert [(item["field"], item["value"]) for item in patches] == [
        ("scores.deck.grade", "B级"),
    ]
    state.scores.deck.score.value = 76.36
    assert registry._selected_component_grade_patches(catalog) == []


def test_repeated_final_tables_block_lone_intermediate_override() -> None:
    state = DocumentState(document_id="sha256:repeated-final", source_path="repeated-final.docx")
    registry = EvidenceToolRegistry(
        state,
        ParsedDocument(path=Path("repeated-final.docx"), paragraphs=[], tables=[]),
        ConstraintEngine(),
    )
    catalog = {"superstructure": [
        {
            "source_role": "final_component_rating_table",
            "score": 79.0,
            "grade": "C",
            "side": None,
            "self_contradictory": False,
            "evidence_id": "T1R2",
        },
        {
            "source_role": "final_component_rating_table",
            "score": 79.0,
            "grade": "C",
            "side": None,
            "self_contradictory": False,
            "evidence_id": "T36R2",
        },
        {
            "source_role": "explicit_assessment_sentence",
            "score": 84.04,
            "grade": "B",
            "side": None,
            "self_contradictory": False,
            "evidence_id": "P2718",
        },
    ]}

    score_consensus = registry._repeated_final_table_consensus(
        "scores.superstructure.score", catalog,
    )
    grade_consensus = registry._repeated_final_table_consensus(
        "scores.superstructure.grade", catalog,
    )

    assert score_consensus == {"value": 79.0, "evidence_ids": ["T1R2", "T36R2"]}
    assert grade_consensus == {"value": "C级", "evidence_ids": ["T1R2", "T36R2"]}
    assert not registry._same_summary_value(84.04, score_consensus["value"])
    assert not registry._same_summary_value("B级", grade_consensus["value"])


def test_review_write_gate_isolates_one_unsupported_patch() -> None:
    state = DocumentState(document_id="sha256:isolated-review-patch", source_path="isolated-review-patch.docx")
    state.add_evidence(EvidenceRef(
        id="T0R0",
        kind="table",
        text="工程名称 | 示例桥",
        source_path="isolated-review-patch.docx",
        priority=95,
    ))
    state.add_evidence(EvidenceRef(
        id="T1R1",
        kind="table",
        text="下部结构 | 83.0 | B级",
        source_path="isolated-review-patch.docx",
        priority=90,
    ))
    registry = EvidenceToolRegistry(
        state,
        ParsedDocument(path=Path("isolated-review-patch.docx"), paragraphs=[], tables=[]),
        ConstraintEngine(),
    )
    skipped: list[dict] = []

    applied, used_evidence = registry._apply_review_fact_patches(
        [
            {"field": "bridge.name", "value": "示例桥", "evidence_ids": ["T0R0"]},
            {"field": "scores.substructure.score", "value": 0.0, "evidence_ids": ["T1R1"]},
        ],
        semantic_conflict_fields={"scores.substructure.score"},
        skipped=skipped,
    )

    assert applied == ["bridge.name"]
    assert used_evidence == ["T0R0"]
    assert state.bridge.name.value == "示例桥"
    assert state.scores.substructure.score.value is None
    assert skipped == [{
        "field": "scores.substructure.score",
        "phase": "write_gate",
        "reason": "scores.substructure.score 的值不受引用证据支持: 0.0",
    }]


def test_published_abstract_bsi_is_narrow_exception_to_component_default() -> None:
    conclusion = (
        "检测结论 | 左幅桥面系BSIm=62.72，评定为D级；"
        "右幅桥面系BSIm=62.72，评定为D级，整桥BCI=91.3，评定为A级。"
    )
    state = DocumentState(document_id="sha256:abstract-bsi", source_path="abstract-bsi.docx")
    parsed = ParsedDocument(
        path=Path("abstract-bsi.docx"),
        paragraphs=[],
        tables=[TableBlock(index=0, rows=[
            ["项目", "内容"],
            ["检测结论", conclusion],
        ])],
    )
    registry = EvidenceToolRegistry(state, parsed, ConstraintEngine())
    registry._ensure_table_evidence(0, 1, conclusion, priority=99)
    catalog = {"deck": [
        {"evidence_id": "T0R1", "metric_role": "control_index", "score": 62.72, "grade": "D", "grade_binding": True, "side": "左幅", "self_contradictory": False},
        {"evidence_id": "T0R1", "metric_role": "control_index", "score": 62.72, "grade": "D", "grade_binding": True, "side": "右幅", "self_contradictory": False},
    ]}

    patches = registry._published_summary_metric_patches(catalog)
    by_field = {item["field"]: item["value"] for item in patches}

    assert by_field == {
        "scores.deck.score": "62.72（左幅）/62.72（右幅）",
        "scores.deck.grade": "D级",
    }


def test_overall_side_tables_complete_selected_scalar_for_equal_or_divergent_grades() -> None:
    def rating_table(index: int, side: str, score: str, grade: str) -> TableBlock:
        return TableBlock(
            index=index,
            rows=[
                ["部位名称", "技术状况指数", "技术状况", "权重", "BCI", "桥梁整体技术状况等级"],
                ["桥面系", "88", "B", "0.15", score, grade],
                ["上部结构", "80", "B", "0.4", score, grade],
            ],
            context=[f"{side}技术状况评定表"],
        )

    state = DocumentState(document_id="sha256:overall-sides", source_path="overall-sides.docx")
    state.scores.overall.score = FieldValue(value=86.56)
    registry = EvidenceToolRegistry(
        state,
        ParsedDocument(
            path=Path("overall-sides.docx"),
            paragraphs=[],
            tables=[rating_table(0, "左幅", "93.83", "A"), rating_table(1, "右幅", "86.56", "B")],
        ),
        ConstraintEngine(),
    )

    patches = registry._divergent_overall_side_patches()
    assert {item["field"]: item["value"] for item in patches} == {
        "scores.overall.score": "93.83（左幅）/86.56（右幅）",
        "scores.overall.grade": "A级（左幅）/B级（右幅）",
    }

    equal_state = DocumentState(document_id="sha256:overall-equal", source_path="overall-equal.docx")
    equal_state.scores.overall.score = FieldValue(value=82.77)
    equal_registry = EvidenceToolRegistry(
        equal_state,
        ParsedDocument(
            path=Path("overall-equal.docx"),
            paragraphs=[],
            tables=[rating_table(0, "左幅", "82.77", "B"), rating_table(1, "右幅", "82.71", "B")],
        ),
        ConstraintEngine(),
    )
    assert {item["field"]: item["value"] for item in equal_registry._divergent_overall_side_patches()} == {
        "scores.overall.score": "82.77（左幅）/82.71（右幅）",
        "scores.overall.grade": "B级（左幅）/B级（右幅）",
    }

    unrelated_state = DocumentState(document_id="sha256:overall-unrelated", source_path="overall-unrelated.docx")
    unrelated_state.scores.overall.score = FieldValue(value=91.3)
    unrelated_registry = EvidenceToolRegistry(
        unrelated_state,
        ParsedDocument(
            path=Path("overall-unrelated.docx"),
            paragraphs=[],
            tables=[rating_table(0, "左幅", "85.0", "B"), rating_table(1, "右幅", "84.2", "B")],
        ),
        ConstraintEngine(),
    )
    assert unrelated_registry._divergent_overall_side_patches() == []


def test_summary_review_uses_embedding_reranker_and_qwen() -> None:
    state = DocumentState(document_id="sha256:test", source_path="test.docx")
    state.add_evidence(EvidenceRef(
        id="P0",
        kind="paragraph",
        text="测试桥最终评定：总体 BCI=90.1，技术状况等级为A级。",
        source_path="test.docx",
        priority=92,
        paragraph_index=0,
    ))
    parsed = ParsedDocument(
        path=Path("test.docx"),
        paragraphs=[ParagraphBlock(index=0, text="测试桥最终评定：总体 BCI=90.1，技术状况等级为A级。", style="正文")],
        tables=[],
    )
    client = _SemanticClient()
    registry = EvidenceToolRegistry(
        state,
        parsed,
        ConstraintEngine(),
        client=client,
        settings=Settings(llm_mode="off"),
    )

    result = registry.execute(AgentCommand(run_id="run:test", seq=1, action="review_summary"))

    assert result.success, result.error
    assert client.tasks == [
        "summary-evidence-embedding",
        "summary-evidence-rerank",
        "summary-review",
    ]
    assert result.content["embedded_evidence"] >= 1
    assert result.content["reranked_evidence"] == 1


class _UnsupportedSummaryPatchClient(_SemanticClient):
    def complete_model(self, response_model, *, task: str, **_: object):
        self.tasks.append(task)
        return response_model(
            patches=[{
                "field": "scores.previous.score",
                "value": "无",
                "evidence_ids": ["P0"],
                "reason": "model proposed unsupported absence",
            }],
            warnings=[],
        )


def test_summary_review_skips_one_unsupported_patch_without_failing_action() -> None:
    state = DocumentState(document_id="sha256:skip", source_path="skip.docx")
    state.add_evidence(EvidenceRef(
        id="P0",
        kind="paragraph",
        text="本次总体 BCI=88.0，技术状况等级为B级。",
        source_path="skip.docx",
        priority=92,
        paragraph_index=0,
    ))
    parsed = ParsedDocument(
        path=Path("skip.docx"),
        paragraphs=[ParagraphBlock(
            index=0,
            text="本次总体 BCI=88.0，技术状况等级为B级。",
            style="正文",
        )],
        tables=[],
    )
    registry = EvidenceToolRegistry(
        state,
        parsed,
        ConstraintEngine(),
        client=_UnsupportedSummaryPatchClient(),
        settings=Settings(llm_mode="off"),
    )

    result = registry.execute(
        AgentCommand(run_id="run:skip", seq=1, action="review_summary")
    )

    assert result.success, result.error
    assert result.content["applied"] == []
    assert result.content["skipped"] == [{
        "field": "scores.previous.score",
        "phase": "primary",
        "reason": "候选值不受引用证据支持",
    }]


class _DetailClient:
    def __init__(self):
        self.calls = 0
        self.tasks: list[str] = []

    def embed_texts(self, texts: list[str], *, task: str):
        self.tasks.append(task)
        return [[1.0, float(index % 2)] for index, _ in enumerate(texts)]

    def rerank_documents(self, *, query: str, documents: list[str], top_n: int, task: str):
        self.tasks.append(task)
        return [{"index": 0, "score": 0.99, "document": documents[0]}]

    def complete_model(self, response_model, *, task: str, **_: object):
        self.calls += 1
        self.tasks.append(task)
        return response_model(
            disease_patches=[{
                "index": 0,
                "evidence_ids": ["D0"],
                "location": "左幅1#梁跨中",
                "disease_type": "裂缝",
                "description": "左幅1#梁跨中存在裂缝，长1.2m",
            }],
            recommendation_patches=[{
                "index": 0,
                "evidence_ids": ["R0"],
                "category": "尽快维修",
                "component": "左幅1#梁",
            }],
            warnings=[],
        )


def test_detail_review_only_applies_evidence_supported_patches() -> None:
    state = DocumentState(document_id="sha256:detail", source_path="detail.docx")
    state.add_evidence(EvidenceRef(
        id="D0",
        kind="table",
        text="左幅1#梁跨中存在裂缝，长1.2m",
        source_path="detail.docx",
        priority=90,
        table_index=0,
        row_index=1,
    ))
    state.add_evidence(EvidenceRef(
        id="R0",
        kind="paragraph",
        text="建议及时修复左幅1#梁裂缝。",
        source_path="detail.docx",
        priority=90,
        paragraph_index=2,
    ))
    state.diseases = [Disease(
        component="其他",
        location="1#梁",
        type="开裂",
        description="1#梁开裂",
        evidence=["D0"],
    )]
    state.recommendations = [Recommendation(
        sequence=1,
        category="预防性养护",
        component="主梁",
        content="建议及时修复左幅1#梁裂缝。",
        evidence=["R0"],
    )]
    client = _DetailClient()
    registry = EvidenceToolRegistry(
        state,
        ParsedDocument(path=Path("detail.docx"), paragraphs=[], tables=[]),
        ConstraintEngine(),
        client=client,
        settings=Settings(llm_mode="off"),
    )

    result = registry.execute(AgentCommand(run_id="run:detail", seq=1, action="review_details"))

    assert result.success, result.error
    assert client.calls == 1
    assert client.tasks == ["detail-evidence-embedding", "detail-evidence-rerank", "detail-review"]
    assert state.diseases[0].location == "左幅1#梁跨中"
    assert state.diseases[0].type == "裂缝"
    assert state.diseases[0].description == "左幅1#梁跨中存在裂缝，长1.2m"
    assert state.recommendations[0].category == "尽快维修"
    assert state.recommendations[0].component == "左幅1#梁"


class _TimeoutDetailClient(_DetailClient):
    def complete_model(self, response_model, *, task: str, **_: object):
        self.calls += 1
        self.tasks.append(task)
        raise TimeoutError("competition detail review timed out")


class _UnavailableSemanticDetailClient(_DetailClient):
    def embed_texts(self, texts: list[str], *, task: str):
        self.tasks.append(task)
        raise RuntimeError("semantic gateway 503")


def test_detail_review_falls_back_to_row_evidence_when_semantic_gateway_fails() -> None:
    state = DocumentState(document_id="sha256:semantic-fallback", source_path="fallback.docx")
    state.add_evidence(EvidenceRef(
        id="D0",
        kind="table",
        text="左幅1#梁跨中存在裂缝，长1.2m",
        source_path="fallback.docx",
        priority=90,
        table_index=0,
        row_index=1,
    ))
    state.diseases = [Disease(
        component="其他",
        location="1#梁",
        type="开裂",
        description="1#梁开裂",
        evidence=["D0"],
    )]
    client = _UnavailableSemanticDetailClient()
    registry = EvidenceToolRegistry(
        state,
        ParsedDocument(
            path=Path("fallback.docx"),
            paragraphs=[ParagraphBlock(
                index=0,
                text="左幅1#梁跨中存在裂缝，长1.2m",
                style="正文",
            )],
            tables=[],
        ),
        ConstraintEngine(),
        client=client,
        settings=Settings(llm_mode="off"),
    )

    result = registry.execute(
        AgentCommand(run_id="run:fallback", seq=1, action="review_details")
    )

    assert result.success, result.error
    assert result.content["semantic_fallback"] == "RuntimeError"
    assert result.content["model_batches"] == 1
    assert state.diseases[0].location == "左幅1#梁跨中"


def test_detail_review_timeout_preserves_deterministic_rows() -> None:
    state = DocumentState(document_id="sha256:timeout", source_path="timeout.docx")
    state.add_evidence(EvidenceRef(
        id="D0",
        kind="table",
        text="1#梁跨中存在裂缝，长1.2m",
        source_path="timeout.docx",
        priority=90,
        table_index=0,
        row_index=1,
    ))
    state.diseases = [Disease(
        component="主梁",
        location="1#梁跨中",
        type="裂缝",
        description="1#梁跨中存在裂缝，长1.2m",
        evidence=["D0"],
    )]
    client = _TimeoutDetailClient()
    registry = EvidenceToolRegistry(
        state,
        ParsedDocument(
            path=Path("timeout.docx"),
            paragraphs=[ParagraphBlock(
                index=0,
                text="1#梁跨中存在裂缝，长1.2m",
                style="正文",
            )],
            tables=[],
        ),
        ConstraintEngine(),
        client=client,
        settings=Settings(llm_mode="off"),
    )

    result = registry.execute(
        AgentCommand(run_id="run:timeout", seq=1, action="review_details")
    )

    assert result.success, result.error
    assert result.content["model_batches"] == 0
    assert result.content["model_failures"] == [{
        "phase": "primary",
        "batch": 0,
        "error": "TimeoutError",
    }]
    assert state.diseases[0].description == "1#梁跨中存在裂缝，长1.2m"


def test_detail_review_samples_uncertain_rows_across_long_ledger() -> None:
    selected = EvidenceToolRegistry._bounded_review_indices(
        list(range(100)),
        lambda _: True,
        4,
    )

    assert selected == [0, 25, 50, 75]


def test_detail_review_can_select_every_uncertain_row_without_cap() -> None:
    selected = EvidenceToolRegistry._bounded_review_indices(
        list(range(100)),
        lambda item: item % 2 == 0,
        None,
    )

    assert selected == list(range(0, 100, 2))


def test_detail_review_samples_ranked_index_pool_across_long_ledger() -> None:
    selected = EvidenceToolRegistry._sample_index_pool(list(range(100)), 4)

    assert selected == [0, 25, 50, 75]


class _NoveltyCorrectionClient:
    def embed_texts(self, texts: list[str], *, task: str):
        return [[1.0, float(index % 2)] for index, _ in enumerate(texts)]

    def rerank_documents(self, *, query: str, documents: list[str], top_n: int, task: str):
        return [{"index": 0, "score": 0.99, "document": documents[0]}]

    def complete_model(self, response_model, *, task: str, **_: object):
        return response_model(
            disease_patches=[{
                "index": 0,
                "evidence_ids": ["HISTORY:2017:abc:D0"],
                "is_new": False,
            }],
            recommendation_patches=[],
            warnings=[],
        )


def test_detail_review_can_correct_low_confidence_history_absence() -> None:
    exact_id = "HISTORY:2017:abc:D0"
    absence_id = "HISTORY:2017:abc:ABSENCE:0"
    state = DocumentState(document_id="sha256:novelty", source_path="novelty.docx")
    state.add_evidence(EvidenceRef(
        id=exact_id,
        kind="derived",
        text="上一年度0#台侧墙已发现渗水病害",
        source_path="history.docx",
        priority=76,
    ))
    state.add_evidence(EvidenceRef(
        id=absence_id,
        kind="derived",
        text="一对一匹配未找到对应项",
        source_path="history.docx",
        priority=68,
    ))
    state.diseases = [Disease(
        component="桥台",
        location="0#台侧墙",
        type="渗水",
        description="0#台侧墙渗水",
        is_new=True,
        novelty_source="history_absence",
        previous_state="未发现",
        development="无",
        evidence=[absence_id, exact_id],
    )]
    registry = EvidenceToolRegistry(
        state,
        ParsedDocument(path=Path("novelty.docx"), paragraphs=[], tables=[]),
        ConstraintEngine(),
        client=_NoveltyCorrectionClient(),
        settings=Settings(llm_mode="off"),
    )

    result = registry.execute(AgentCommand(run_id="run:novelty", seq=1, action="review_details"))

    assert result.success, result.error
    assert state.diseases[0].is_new is False
    assert state.diseases[0].novelty_source == "model"
    assert state.diseases[0].previous_state == "已发现"


class _OmittedTemporalRetryClient:
    def __init__(self) -> None:
        self.detail_calls = 0

    def embed_texts(self, texts: list[str], *, task: str):
        return [[1.0, float(index % 2)] for index, _ in enumerate(texts)]

    def rerank_documents(self, *, query: str, documents: list[str], top_n: int, task: str):
        return [{"index": 0, "score": 0.99, "document": documents[0]}]

    def complete_model(self, response_model, *, task: str, user_payload: dict, **_: object):
        self.detail_calls += 1
        if self.detail_calls == 1:
            return response_model(
                disease_patches=[{
                    "index": 0,
                    "evidence_ids": ["T2R1"],
                    "is_new": True,
                    "reason": "只引用当前病害行，写入门应拒绝并触发重试。",
                }],
                recommendation_patches=[],
                warnings=[],
            )
        assert user_payload["force_temporal_decision"] is True
        return response_model(
            disease_patches=[{
                "index": 0,
                "evidence_ids": ["T8R1"],
                "previous_state": "未发现",
                "reason": "上次未列该病害，本次新增部分病害中出现。",
            }],
            recommendation_patches=[],
            warnings=[],
        )


def test_detail_review_retries_only_omitted_temporal_rows() -> None:
    state = DocumentState(document_id="sha256:omitted", source_path="omitted.docx")
    state.add_evidence(EvidenceRef(
        id="T2R1",
        kind="table",
        text="1#跨主梁涂层剥落。",
        source_path="omitted.docx",
        priority=80,
        table_index=2,
        row_index=1,
    ))
    state.add_evidence(EvidenceRef(
        id="T8R1",
        kind="table",
        text="上一次检测结果=未发现涂层剥落；本次检测结果=新增部分主梁涂层剥落。",
        source_path="omitted.docx",
        priority=90,
        table_index=8,
        row_index=1,
    ))
    state.diseases = [Disease(
        component="主梁",
        location="1#跨",
        type="涂层剥落",
        description="1#跨主梁涂层剥落",
        evidence=["T2R1", "T8R1"],
    )]
    client = _OmittedTemporalRetryClient()
    registry = EvidenceToolRegistry(
        state,
        ParsedDocument(path=Path("omitted.docx"), paragraphs=[], tables=[]),
        ConstraintEngine(),
        client=client,
        settings=Settings(llm_mode="off"),
    )

    result = registry.execute(AgentCommand(run_id="run:omitted", seq=1, action="review_details"))

    assert result.success, result.error
    assert client.detail_calls == 2
    assert state.diseases[0].is_new is True
    assert state.diseases[0].previous_state == "未发现"
    assert state.diseases[0].development == "无"


class _DiseaseClient:
    def __init__(self) -> None:
        self.task = ""

    def complete_model(self, response_model, *, task: str, **_: object):
        self.task = task
        return response_model(items=[{"index": 0, "component": "不存在的构件", "type": "裂缝"}])


def test_disease_normalizer_uses_flash_task_and_rejects_unknown_component() -> None:
    diseases = [Disease(
        sequence="1",
        component="其他",
        location="1#梁",
        type="未明确",
        description="1#梁裂缝",
    )]
    client = _DiseaseClient()

    normalized = DiseaseNormalizer(client, Settings(llm_mode="off"), "system").normalize(diseases)

    assert client.task == "disease-normalization"
    assert normalized[0].component == "其他"
    assert normalized[0].type == "裂缝"


def test_fact_patch_cannot_replace_equal_priority_stable_fact() -> None:
    state = DocumentState(document_id="sha256:stable", source_path="stable.docx")
    state.scores.overall.grade = FieldValue(
        value="A级",
        evidence=["T0R1"],
        source_priority=95,
        confidence=0.99,
    )
    state.add_evidence(EvidenceRef(
        id="T0R1",
        kind="table",
        text="最终评定：总体技术状况等级 A级",
        source_path="stable.docx",
        priority=95,
        table_index=0,
        row_index=1,
    ))
    state.add_evidence(EvidenceRef(
        id="T1R1",
        kind="table",
        text="另一候选表：总体技术状况等级 B级",
        source_path="stable.docx",
        priority=95,
        table_index=1,
        row_index=1,
    ))
    registry = EvidenceToolRegistry(
        state,
        ParsedDocument(path=Path("stable.docx"), paragraphs=[], tables=[]),
        ConstraintEngine(),
    )

    result = registry.execute(AgentCommand(
        run_id="run:stable",
        seq=1,
        action="apply_fact_patch",
        arguments={
            "patches": [{
                "field": "scores.overall.grade",
                "value": "B级",
                "evidence_ids": ["T1R1"],
            }]
        },
    ))

    assert result.success is False
    assert "高置信事实证据不弱于候选" in (result.error or "")
    assert state.scores.overall.grade.value == "A级"


def test_visual_gap_does_not_treat_absent_component_scores_as_missing() -> None:
    state = DocumentState(document_id="sha256:visual-gate", source_path="visual-gate.docx")
    state.bridge.name = FieldValue(value="测试桥", confidence=0.99)
    state.bridge.report_date = FieldValue(value="2026年8月14日", confidence=0.99)
    state.scores.overall.score = FieldValue(value=87.35, confidence=0.99)
    state.scores.overall.grade = FieldValue(value="B级", confidence=0.99)
    parsed = ParsedDocument(
        path=Path("visual-gate.docx"),
        paragraphs=[ParagraphBlock(index=0, text="本报告仅提供全桥综合评定。", style="正文")],
        tables=[],
    )
    registry = EvidenceToolRegistry(state, parsed, ConstraintEngine())

    assert registry._critical_visual_gap() is False

    registry.parsed = ParsedDocument(
        path=Path("visual-gate.docx"),
        paragraphs=[ParagraphBlock(index=0, text="桥面系评分 BCIm 见附图。", style="正文")],
        tables=[],
    )
    assert registry._critical_visual_gap() is True


def test_decision_grounding_rejects_new_numbers_grades_and_diseases() -> None:
    support = "总体评分 87.35，等级 A级；主梁存在裂缝2处。"

    assert EvidenceToolRegistry._narrative_grounded("总体评分87.35，A级，裂缝2处。", support)
    assert not EvidenceToolRegistry._narrative_grounded("总体评分99.99。", support)
    assert not EvidenceToolRegistry._narrative_grounded("总体等级B级。", support)
    assert not EvidenceToolRegistry._narrative_grounded("存在支座锈蚀。", support)


def _online_state() -> DocumentState:
    state = DocumentState(document_id="sha256:audit", source_path="audit.docx")
    actions = ["review_summary", "review_details", "normalize_diseases", "write_decision", "finalize"]
    trajectory = []
    for seq, action in enumerate(actions, start=1):
        command = AgentCommand(run_id="run:audit", seq=seq, action=action)
        result = ToolResult(run_id="run:audit", seq=seq, tool=action, success=True)
        trajectory.append(TrajectoryStep(command=command, result=result, state_hash_before="a", state_hash_after="b"))
    state.agent_run = AgentRunRecord(
        run_id="run:audit",
        planner="qwen",
        status="completed",
        trajectory=trajectory,
        final_validation_passed=True,
    )
    state.metrics.task_calls = {
        "react-plan": 4,
        "summary-evidence-embedding": 1,
        "summary-evidence-rerank": 1,
        "detail-evidence-embedding": 1,
        "detail-evidence-rerank": 1,
        "summary-review": 1,
        "detail-review": 1,
        "decision": 1,
    }
    state.metrics.task_models = {
        "react-plan": "deepseek/deepseek-v4-pro",
        "summary-evidence-embedding": "Qwen3-VL-Embedding-8B",
        "summary-evidence-rerank": "Qwen3-VL-Reranker-8B",
        "detail-evidence-embedding": "Qwen3-VL-Embedding-8B",
        "detail-evidence-rerank": "Qwen3-VL-Reranker-8B",
        "summary-review": "deepseek/deepseek-v4-pro",
        "detail-review": "deepseek/deepseek-v4-pro",
        "decision": "deepseek/deepseek-v4-pro",
    }
    state.metrics.model_calls = {
        "deepseek/deepseek-v4-pro": 7,
        "Qwen3-VL-Embedding-8B": 2,
        "Qwen3-VL-Reranker-8B": 2,
    }
    return state


def test_online_audit_requires_each_model_and_tool_stage() -> None:
    state = _online_state()
    assert BridgeEvidenceOrchestrator._online_audit_errors(state) == []

    del state.metrics.task_calls["summary-evidence-embedding"]
    errors = BridgeEvidenceOrchestrator._online_audit_errors(state)
    assert "缺少真实网络模型任务调用 summary-evidence-embedding" in errors


def test_online_audit_accepts_deterministic_planner_without_react_calls() -> None:
    state = _online_state()
    state.agent_run.planner = "deterministic"
    state.metrics.task_calls.pop("react-plan")
    state.metrics.task_models.pop("react-plan")

    assert BridgeEvidenceOrchestrator._online_audit_errors(
        state,
        planner_mode="deterministic",
    ) == []


def test_summary_workflow_excludes_detail_tools_and_obligations() -> None:
    state = DocumentState(document_id="sha256:summary", source_path="summary.docx")
    parsed = ParsedDocument(
        path=Path("summary.docx"),
        paragraphs=[ParagraphBlock(index=0, text="桥面系 BCIm=85，BSIm=60", style="")],
        tables=[],
    )
    registry = EvidenceToolRegistry(
        state,
        parsed,
        ConstraintEngine(),
        client=object(),
        settings=Settings(llm_mode="off"),
        workflow_profile="summary",
    )

    tool_names = {tool.name for tool in registry.manifest.tools}
    pending_names = {item["action"] for item in registry.pending_actions()}
    assert "review_summary" in tool_names
    assert "review_summary" in pending_names
    assert "review_details" not in tool_names
    assert "normalize_diseases" not in pending_names
    assert "write_decision" not in pending_names


def test_details_workflow_excludes_summary_fact_and_decision_tools() -> None:
    state = DocumentState(document_id="sha256:details", source_path="details.docx")
    parsed = ParsedDocument(
        path=Path("details.docx"),
        paragraphs=[ParagraphBlock(
            index=0,
            text="桥面系 BCIm=85，BSIm=60；主梁存在裂缝。",
            style="正文",
        )],
        tables=[],
    )
    registry = EvidenceToolRegistry(
        state,
        parsed,
        ConstraintEngine(),
        client=object(),
        settings=Settings(llm_mode="off"),
        workflow_profile="details",
    )

    tool_names = {tool.name for tool in registry.manifest.tools}
    pending_names = {item["action"] for item in registry.pending_actions()}
    assert tool_names == {
        "get_validation_report",
        "review_details",
        "finalize",
        "fail",
    }
    assert pending_names == {"review_details"}


def test_summary_online_audit_requires_only_summary_network_stages() -> None:
    state = DocumentState(document_id="sha256:summary-audit", source_path="summary-audit.docx")
    trajectory = []
    for seq, action in enumerate(("review_summary", "finalize"), start=1):
        command = AgentCommand(run_id="run:summary-audit", seq=seq, action=action)
        result = ToolResult(run_id="run:summary-audit", seq=seq, tool=action, success=True)
        trajectory.append(TrajectoryStep(command=command, result=result, state_hash_before="a", state_hash_after="b"))
    state.agent_run = AgentRunRecord(
        run_id="run:summary-audit",
        planner="qwen",
        status="completed",
        trajectory=trajectory,
        final_validation_passed=True,
    )
    state.metrics.task_calls = {
        "react-plan": 2,
        "summary-evidence-embedding": 1,
        "summary-evidence-rerank": 1,
        "summary-review": 1,
    }

    assert BridgeEvidenceOrchestrator._online_audit_errors(
        state,
        workflow_profile="summary",
    ) == []


def test_details_online_audit_requires_only_detail_network_stages() -> None:
    state = DocumentState(document_id="sha256:details-audit", source_path="details-audit.docx")
    trajectory = []
    for seq, action in enumerate(("review_details", "finalize"), start=1):
        command = AgentCommand(run_id="run:details-audit", seq=seq, action=action)
        result = ToolResult(run_id="run:details-audit", seq=seq, tool=action, success=True)
        trajectory.append(TrajectoryStep(
            command=command,
            result=result,
            state_hash_before="a",
            state_hash_after="b",
        ))
    state.agent_run = AgentRunRecord(
        run_id="run:details-audit",
        planner="deterministic",
        status="completed",
        trajectory=trajectory,
        final_validation_passed=True,
    )
    state.metrics.task_calls = {
        "detail-evidence-embedding": 1,
        "detail-evidence-rerank": 1,
        "detail-review": 1,
    }

    assert BridgeEvidenceOrchestrator._online_audit_errors(
        state,
        workflow_profile="details",
        planner_mode="deterministic",
    ) == []


def test_details_online_audit_accepts_explicit_semantic_fallback() -> None:
    state = DocumentState(document_id="sha256:fallback-audit", source_path="fallback-audit.docx")
    trajectory = []
    for seq, action in enumerate(("review_details", "finalize"), start=1):
        command = AgentCommand(run_id="run:fallback-audit", seq=seq, action=action)
        content = {"semantic_fallback": "HTTPStatusError"} if action == "review_details" else {}
        result = ToolResult(
            run_id="run:fallback-audit",
            seq=seq,
            tool=action,
            success=True,
            content=content,
        )
        trajectory.append(TrajectoryStep(
            command=command,
            result=result,
            state_hash_before="a",
            state_hash_after="b",
        ))
    state.agent_run = AgentRunRecord(
        run_id="run:fallback-audit",
        planner="deterministic",
        status="completed",
        trajectory=trajectory,
        final_validation_passed=True,
    )
    state.metrics.task_calls = {"detail-review": 1}

    assert BridgeEvidenceOrchestrator._online_audit_errors(
        state,
        workflow_profile="details",
        planner_mode="deterministic",
    ) == []


def test_online_audit_requires_both_visual_calls_when_visual_tool_runs() -> None:
    state = _online_state()
    command = AgentCommand(run_id="run:audit", seq=6, action="inspect_visual_evidence")
    result = ToolResult(
        run_id="run:audit",
        seq=6,
        tool="inspect_visual_evidence",
        success=True,
    )
    state.agent_run.trajectory.insert(
        -1,
        TrajectoryStep(
            command=command,
            result=result,
            state_hash_before="a",
            state_hash_after="b",
        ),
    )
    state.metrics.task_calls["visual-evidence-primary"] = 1

    errors = BridgeEvidenceOrchestrator._online_audit_errors(state)

    assert "缺少真实网络模型任务调用 visual-evidence-critic" in errors


def test_expected_model_routes_include_visual_models_only_when_used() -> None:
    orchestrator = object.__new__(BridgeEvidenceOrchestrator)
    orchestrator.settings = Settings(llm_mode="off", vision_model="minimax-m3", vision_critic_model="kimi-k3")
    state = _online_state()
    assert "visual-evidence-primary" not in orchestrator._expected_task_models(state)

    command = AgentCommand(run_id="run:audit", seq=6, action="inspect_visual_evidence")
    result = ToolResult(
        run_id="run:audit",
        seq=6,
        tool="inspect_visual_evidence",
        success=True,
    )
    state.agent_run.trajectory.insert(
        -1,
        TrajectoryStep(
            command=command,
            result=result,
            state_hash_before="a",
            state_hash_after="b",
        ),
    )

    expected = orchestrator._expected_task_models(state)

    assert expected["visual-evidence-primary"] == "minimax-m3"
    assert expected["visual-evidence-critic"] == "kimi-k3"


def test_normalization_model_route_is_required_only_when_targets_exist() -> None:
    orchestrator = object.__new__(BridgeEvidenceOrchestrator)
    orchestrator.settings = Settings(llm_mode="off", task_models={"disease-normalization": "deepseek/deepseek-v4-flash"})
    state = _online_state()
    step = next(
        item for item in state.agent_run.trajectory
        if item.command.action == "normalize_diseases"
    )
    step.result.content = {"targets_before": 2, "targets_after": 0}

    expected = orchestrator._expected_task_models(state)
    errors = BridgeEvidenceOrchestrator._online_audit_errors(state, expected)

    assert expected["disease-normalization"] == "deepseek/deepseek-v4-flash"
    assert "缺少真实网络模型任务调用 disease-normalization" in errors
