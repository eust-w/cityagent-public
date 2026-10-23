"""Ordered source snapshots, token-budgeted ownership and fail-closed reduction."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..domain_rules import ACTIVE_DEFECT_RE
from .inference_core import ContextBudgetError, build_prompt
from .model_json import parse_json_object
from .run_state import json_digest

LEGACY_FRAGMENT_INSTRUCTION = """你是桥梁定检报告抽取专家。输入包含有序来源、负责区 owned 和只读上下文 context。
只提取 owned 中有证据的基本信息、维修建议、病害明细。不得丢弃重复的真实行，不得把历史病害当作本次病害。
输出完整 JSON：{"covered":[所有 owned 的 key],"基本信息":{字段:值},"basic_sources":{字段:source_id},
"维修建议":[{"source_id":"来源","source_key":"owned key","instance_id":"该来源内唯一序号","fields":{现有建议字段},"quotes":{字段:"精确原文"}}],
"病害明细":[{"source_id":"来源","source_key":"owned key","instance_id":"该来源内唯一序号","fields":{现有病害字段},"quotes":{字段:"精确原文"},
"identity":{"bridge":"桥","span":"幅别","component":"构件","position":"位置","year":2018},
"temporal":"新增或已有或证据不足","history_sources":[],"history_identity":{同identity字段但年份为历史年}}]}。
无信息可返回空对象或空列表。所有行必须含来源。引用文字必须与来源一致；不确定历史时 temporal=证据不足。
多种评分冲突时保留其对象、年份、评分体系，不得把不同对象的评分混合。只输出 JSON。"""

FRAGMENT_INSTRUCTION = """抽取owned三表，context仅证据；保留重复行，区分本次/历史。
只输出JSON：{"covered":[所有owned的key],"基本信息":{},"basic_sources":{},"basic_quotes":{},"basic_scopes":{},"维修建议":[],"病害明细":[]}。
基本信息字段：桥梁名称、报告日期、总体评分、总体等级、上部结构评分、上部结构等级、下部结构评分、下部结构等级、桥面系评分、桥面系等级、上一次总体评分、上一次总体等级、上一次定检结果、病害发展趋势与具体说明、总体结论、主要风险点、建议统计。
basic_sources填owned的source_id或key；basic_quotes填支持该值的精确原文。basic_scopes默认{}；scope值须在对应basic_quotes中，否则{}；勿猜全桥/年份。
基本信息每个非空值必须被对应basic_quotes原文直接支持；原文等级B就填B，不得补成B级。缺失填null，勿用无/未提及/不详代替缺失，勿虚构引用。建议统计须有原文统计；不得自行统计、分类或推导结论，没有原文则null。
行格式：{"source_id":"来源","source_key":"owned key","instance_id":"来源内唯一序号","fields":{},"quotes":{字段:"精确原文"}}。
每行必须是上述嵌套对象，业务字段只放fields，不能直接放在行顶层；source_id与source_key来自当前owned，quotes与fields字段同名，值是该来源切片的精确原文。保留有证据的行，勿通过删行修复格式。
维修建议fields：序号、建议类别、建议内容、病害部位；勿用“建议”代替“建议内容”。病害fields：序号、病害部位、病害类型、病害描述、是否新增、上一次定检状态、发展程度。
病害默认temporal="证据不足"。主张新增/已有须含identity与history_identity（bridge/span/component/position/year）、history_sources及current_evidence/history_evidence/relationship_evidence列表（每项{source_key,quote}），原文须证明跨年同一病害，不能仅同位置。
多对象评分用basic_observations:[{field,value,scope,source_id,source_key,quote}]，scope限引文支持的year/span/rating_system。缺失填null/[]。
visual-transcription是图片文字转写，并非原生文档文字；visual-observation是模型对画面的描述，并非原文。后者只能支持病害类型和病害描述，不能支持基本信息、维修建议、部位身份、量测或跨年结论；保持其来源域。
以下仅示格式，勿复制示例值。假设owned只有source_id=p0、key=p0:0:23、text=桥梁名称：示例桥；等级B；梁有裂缝，建议复查。
合法输出示例：
{"covered":["p0:0:23"],"基本信息":{"桥梁名称":"示例桥","总体等级":"B","建议统计":null},"basic_sources":{"桥梁名称":"p0","总体等级":"p0"},"basic_quotes":{"桥梁名称":"示例桥","总体等级":"B"},"basic_scopes":{},"basic_observations":[],"维修建议":[{"source_id":"p0","source_key":"p0:0:23","instance_id":"a1","fields":{"建议内容":"建议复查","建议类别":null},"quotes":{"建议内容":"建议复查"}}],"病害明细":[{"source_id":"p0","source_key":"p0:0:23","instance_id":"d1","fields":{"病害部位":"梁","病害描述":"梁有裂缝"},"quotes":{"病害部位":"梁","病害描述":"梁有裂缝"},"temporal":"证据不足"}]}"""


# Opt-in prompt-only contract. Preserve the existing prompt bytes and concise
# suffix so historical baselines remain reproducible.
EXACT_FRAGMENT_INSTRUCTION = FRAGMENT_INSTRUCTION + (
    "\n语义字段简洁；可引用的描述放 quotes，避免重复解释。"
    "\n引文逐字复制对应可见来源切片：basic_quotes、quotes及其他quote必须是原文连续子串，"
    "字符及UTF-8字节完全一致；空格、换行、标点、数字不得增删或改写，"
    "尤其不得在字母与中文之间或数字与单位/年份之间插入空格。JSON必要转义不改变解码后的原文。"
    "\nbasic_quotes和行quotes来自对应owned切片；current_evidence来自当前行source_key的owned切片；"
    "history_evidence和relationship_evidence可来自owned或context，保持实际source_key，"
    "不得把context来源当作本次抽取行。"
    "\n缺失值填null；basic_quotes和quotes仅含实际逐字引用的非空字段，"
    "未提供、未引用或值为null的字段直接省略对应quote键，不能填写null、空串或占位引文；"
    "序号等非引用字段无需虚构quote。非空基本信息仍必须提供对应basic_sources和精确basic_quotes。"
)


EXACT_FRAGMENT_V2_INSTRUCTION = EXACT_FRAGMENT_INSTRUCTION + (
    "\nowned原文明确支持的基本信息必须提取，填写原文直接支持的值、对应basic_sources和逐字basic_quotes。"
    "null仅用于原文确实缺失或证据含糊、无法确定的字段；不得为规避精确引文要求而将已有明确证据的字段留空。"
    "先核对owned中每项基本信息的证据，再决定该字段是否为null；没有原文支持时仍不得猜测或补造。"
)


# Opt-in lean row contract: quotable values appear once (in quotes); the parser
# fills fields from resolved quotes, derives source_id from source_key, and
# merge renumbers rows, so 序号/instance_id/null slots cost output tokens only.
LEAN_FRAGMENT_INSTRUCTION = """抽取owned三表，context仅证据；保留重复行，区分本次/历史。
只输出一个紧凑JSON对象（不换行、不缩进、不加```）：{"基本信息":{},"维修建议":[],"病害明细":[]}。
基本信息必须列出全部15个字段，owned原文有本次报告的值就逐字填写，没有填null：桥梁名称、报告日期、总体评分、总体等级、上部结构评分、上部结构等级、下部结构评分、下部结构等级、桥面系评分、桥面系等级、上一次总体评分、上一次总体等级、病害发展趋势与具体说明、总体结论、主要风险点。报告日期取本次报告日期，不是上次检测日期；评分和等级取本次检测结果，上次检测的总体评分/等级填上一次总体评分/上一次总体等级。
维修建议每条一个扁平对象：{"source_key":"owned key","建议类别":"立即处置或尽快维修或预防性养护","建议内容":"原文","病害部位":"原文"}。
病害明细每条一个扁平对象：{"source_key":"owned key","病害部位":"原文","病害类型":"原文","病害描述":"原文"}。只抽逐处病害记录（含构件位置、数量或尺寸），不抽病害数量汇总（如“裂缝3处”）、评分扣分表和照片说明；原文写明是否新增或上次状态时可加“是否新增”“上一次定检状态”。
值逐字复制原文，不改写、不增删空格和标点；行对象内不嵌套对象、不写null、不写序号。
以下仅示格式，勿复制示例值。假设owned只有key=p0:0:40，原文：桥梁名称：示例桥；技术状况等级B；1#梁跨中裂缝L=0.5m，建议复查。
合法输出示例：
{"基本信息":{"桥梁名称":"示例桥","报告日期":null,"总体评分":null,"总体等级":"B","上部结构评分":null,"上部结构等级":null,"下部结构评分":null,"下部结构等级":null,"桥面系评分":null,"桥面系等级":null,"上一次总体评分":null,"上一次总体等级":null,"病害发展趋势与具体说明":null,"总体结论":null,"主要风险点":null},"维修建议":[{"source_key":"p0:0:40","建议类别":"尽快维修","建议内容":"建议复查","病害部位":"1#梁"}],"病害明细":[{"source_key":"p0:0:40","病害部位":"1#梁跨中","病害类型":"裂缝","病害描述":"1#梁跨中裂缝L=0.5m"}]}"""

ROW_FIELDS = {
    "维修建议": ("序号", "建议类别", "建议内容", "病害部位"),
    "病害明细": ("序号", "病害部位", "病害类型", "病害描述", "是否新增", "上一次定检状态", "发展程度"),
}


@dataclass(frozen=True)
class SourceRegion:
    source_id: str
    order: int
    text: str
    kind: str = "paragraph"
    location: dict = field(default_factory=dict)
    cells: tuple[dict, ...] = ()
    header: tuple[str, ...] = ()
    years: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceSlice:
    source_id: str
    start: int
    end: int
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def key(self):
        return f"{self.source_id}:{self.start}:{self.end}"


@dataclass
class DocumentSnapshot:
    document_sha256: str
    regions: list[SourceRegion]
    parser: str = "ordered-word-xml-v1"
    unverified_content: list[dict] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    @classmethod
    def from_text(cls, text: str):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("nonempty source text required")
        regions = [
            SourceRegion(
                f"p{i}", i, line, years=tuple(re.findall(r"(?:19|20)\d{2}", line))
            )
            for i, line in enumerate(text.splitlines(keepends=True))
        ]
        return cls(json_digest(text), regions, "bound-text-v1")

    @classmethod
    def from_docx(cls, path: Path):
        # Both extraction workflows share one physical OOXML inventory. Nested
        # rows are independent sources; inherited cells remain context only.
        from .evidence_manifest import build_manifest

        manifest = build_manifest(path)
        units = manifest["units"]
        unit_index = {unit["id"]: unit for unit in units}
        tables = {table["id"]: table for table in manifest["tables"]}
        regions = []
        for unit in units:
            table = tables.get(unit.get("table_id"), {})
            headers = [unit_index[key] for key in table.get("header_evidence_ids", [])]
            cells = tuple(
                {
                    **cell,
                    "column": cell["column_index"],
                    "grid_span": cell["column_span"],
                    "v_merge": cell["vertical_merge"],
                }
                for cell in unit.get("cells", [])
            )
            header = tuple(
                cell["text"] for entry in headers for cell in entry.get("cells", [])
            )
            source_id = unit.get("legacy_source_id", unit["id"])
            regions.append(
                SourceRegion(
                    source_id,
                    len(regions),
                    unit["text"],
                    unit["kind"],
                    {
                        "source_path": unit["source_path"],
                        "manifest_id": unit["id"],
                        "story": unit["context"]["story"],
                        "table_id": unit.get("table_id"),
                        "parent_table_id": unit["context"].get("parent_table_id"),
                        "row": unit.get("row_index"),
                        "segment": unit.get("segment_index", 0),
                        "header_sources": [
                            entry.get("legacy_source_id", entry["id"])
                            for entry in headers
                        ],
                    },
                    cells,
                    header,
                    tuple(re.findall(r"(?:19|20)\d{2}", unit["text"])),
                )
            )
        if not regions:
            raise ValueError("document has no source regions")
        return cls(
            manifest["document_sha256"],
            regions,
            manifest["parser_version"],
            [*manifest["unverified_content"], *manifest["images"]],
            manifest["limitations"],
        )

    @property
    def digest(self):
        value = asdict(self)
        if self.parser in {"ordered-word-xml-v1", "bound-text-v1"}:
            if not self.unverified_content:
                value.pop("unverified_content")
            if not self.limitations:
                value.pop("limitations")
        return json_digest(value)

    def sources(self):
        return {r.source_id: r for r in self.regions}


@dataclass
class ChunkTask:
    task_id: str
    owned: list[SourceSlice]
    context: list[SourceSlice]
    prompt_tokens: int = 0

    @property
    def owned_keys(self):
        return [s.key for s in self.owned]

    def text(self):
        def encode(s):
            return {
                "key": s.key,
                "source_id": s.source_id,
                "text": s.text,
                **s.metadata,
            }

        return json.dumps(
            {
                "owned": [encode(s) for s in self.owned],
                "context": [encode(s) for s in self.context],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )


@dataclass
class ChunkPlan:
    snapshot_sha256: str
    tasks: list[ChunkTask]
    input_budget: int
    output_budget: int
    instruction: str = FRAGMENT_INSTRUCTION

    @property
    def digest(self):
        return json_digest(asdict(self))

    def coverage(self, snapshot):
        ranges = {r.source_id: [] for r in snapshot.regions}
        unexpected = []
        for task in self.tasks:
            for s in task.owned:
                if s.source_id not in ranges:
                    unexpected.append(s.key)
                else:
                    ranges[s.source_id].append((s.start, s.end))
        missing = []
        for r in snapshot.regions:
            cursor = 0
            parts = sorted(ranges[r.source_id])
            if not parts:
                missing.append(r.source_id)
            for start, end in parts:
                if start != cursor or end < start or end > len(r.text):
                    missing.append(r.source_id)
                cursor = end
            if cursor != len(r.text):
                missing.append(r.source_id)
        return {
            "complete": not missing and not unexpected,
            "missing": sorted(set(missing)),
            "unexpected": unexpected,
            "sources": len(ranges),
            "document_complete": not missing
            and not unexpected
            and not snapshot.unverified_content
            and snapshot.parser in {
                "evidence-manifest-v2",
                "evidence-manifest-v2+visual-evidence-v1",
            },
            "unverified_content": snapshot.unverified_content,
            "limitations": snapshot.limitations,
            "extraction_completeness": "requires_independent_evaluation",
        }


def _task(owned, context=()):
    return ChunkTask(
        json_digest([s.key for s in owned])[:24], list(owned), list(context)
    )


def source_metadata(region):
    metadata = {
        "kind": region.kind,
        "location": region.location,
        "header": [h[:256] for h in region.header[:16]],
        "header_sources": region.location.get("header_sources", []),
        "years": list(region.years[:16]),
    }
    # Cell structure is in the snapshot. Do not duplicate unbounded row text into metadata.
    metadata["cells"] = [
        {
            k: (v[:128] if k == "merged_context_text" else v)
            for k, v in cell.items()
            if k
            in {
                "column",
                "grid_span",
                "v_merge",
                "merged_from_source",
                "merged_context_text",
            }
        }
        for cell in region.cells[:32]
    ]
    return metadata


def plan_chunks(
    snapshot,
    tokenizer,
    *,
    input_budget=8192,
    output_budget=4096,
    overlap=1,
    instruction=FRAGMENT_INSTRUCTION,
):
    if (
        type(input_budget) is not int
        or input_budget < 1
        or output_budget < 1
        or overlap < 0
    ):
        raise ValueError("invalid chunk budgets")

    def fits(task):
        try:
            ids = build_prompt(
                tokenizer,
                task.text(),
                max_model_len=input_budget + output_budget,
                max_new_tokens=output_budget,
                instruction=instruction,
            )
            task.prompt_tokens = len(ids)
            return True
        except ContextBudgetError:
            return False

    slices = []
    for region in snapshot.regions:
        metadata = source_metadata(region)
        queue = [
            SourceSlice(region.source_id, 0, len(region.text), region.text, metadata)
        ]
        while queue:
            s = queue.pop(0)
            if fits(_task([s])):
                slices.append(s)
            else:
                if len(s.text) <= 1:
                    raise ValueError("template/metadata exceeds input budget")
                mid = s.start + len(s.text) // 2
                queue[:0] = [
                    SourceSlice(
                        s.source_id, s.start, mid, region.text[s.start : mid], metadata
                    ),
                    SourceSlice(
                        s.source_id, mid, s.end, region.text[mid : s.end], metadata
                    ),
                ]
    tasks, owned = [], []
    for s in slices:
        trial = _task(owned + [s])
        if owned and not fits(trial):
            tasks.append(_task(owned))
            owned = []
        owned.append(s)
    if owned:
        tasks.append(_task(owned))
    for i, task in enumerate(tasks):
        context = (
            [s for previous in tasks[max(0, i - overlap) : i] for s in previous.owned][
                -overlap:
            ]
            if overlap
            else []
        )
        task.context = context
        if not fits(task):
            task.context = []
            if not fits(task):
                raise ValueError("chunk exceeds input budget")
    plan = ChunkPlan(snapshot.digest, tasks, input_budget, output_budget, instruction)
    if not plan.coverage(snapshot)["complete"]:
        raise ValueError("source coverage incomplete")
    return plan


def split_task(task, snapshot):
    if len(task.owned) > 1:
        mid = len(task.owned) // 2
        return [
            _task(task.owned[:mid], task.context),
            _task(task.owned[mid:], task.context),
        ]
    s = task.owned[0]
    if len(s.text) < 2:
        raise ValueError("unsplittable length failure")
    mid = s.start + len(s.text) // 2
    return [
        _task(
            [
                SourceSlice(
                    s.source_id, s.start, mid, s.text[: mid - s.start], s.metadata
                )
            ],
            task.context,
        ),
        _task(
            [SourceSlice(s.source_id, mid, s.end, s.text[mid - s.start :], s.metadata)],
            task.context,
        ),
    ]


@dataclass
class ExtractionFragment:
    task_id: str
    covered: list[str]
    basic: dict
    advice: list[dict]
    diseases: list[dict]
    basic_sources: dict = field(default_factory=dict)
    call_id: str = ""
    basic_evidence: dict = field(default_factory=dict)
    basic_observations: list[dict] = field(default_factory=list)


def _scalar(value):
    return (
        value is None
        or isinstance(value, (str, bool, int))
        or (isinstance(value, float) and math.isfinite(value))
    )


def _empty_basic(value):
    """Only JSON null and blank strings express an absent fact, not 0/False."""
    return value is None or (isinstance(value, str) and not value.strip())


def _range(source, snapshot, source_index=None):
    region = (source_index if source_index is not None else snapshot.sources())[
        source.source_id
    ]
    result = {
        "source_id": source.source_id,
        "source_key": source.key,
        "start": source.start,
        "end": source.end,
        "source_path": region.location.get("source_path"),
    }
    if region.kind in {"visual-transcription", "visual-observation"}:
        result.update({
            "evidence_domain": region.kind,
            "upstream_call_id": region.location.get("call_id"),
            "bbox": region.location.get("bbox"),
            "view_sha256": region.location.get("view_sha256"),
            "native_anchor": region.location.get("native_anchor"),
        })
    return result


def _visible_evidence(entries, visible, snapshot, source_index=None):
    if not isinstance(entries, list) or not entries:
        raise ValueError("missing evidence quotes")
    result = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("invalid evidence quote")
        source = visible.get(entry.get("source_key"))
        quote = entry.get("quote")
        if source is None or not isinstance(quote, str) or not quote.strip():
            raise ValueError("evidence quote outside visible source slice")
        resolved = _find_source_quote(quote, source.text)
        if resolved is None:
            raise ValueError("evidence quote outside visible source slice")
        quote = resolved
        local_start = source.text.index(quote)
        result.append(
            {
                **_range(source, snapshot, source_index),
                "quote": quote,
                "quote_start": source.start + local_start,
                "quote_end": source.start + local_start + len(quote),
            }
        )
    return result


def _scalar_in_quote(value, quote):
    rendered = str(value)
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", rendered):
        return (
            re.search(r"(?<![\d.])" + re.escape(rendered) + r"(?!\d|\.\d)", quote)
            is not None
        )
    return rendered in quote


# CJK Unified Ideographs + CJK Symbols / Fullwidth forms
_CJK_RE = re.compile(r"[一-鿿　-〿＀-￯]")
_BOUNDARY_SPACE_RE = re.compile(
    r"(?<=[一-鿿　-〿＀-￯]) (?=[^一-鿿　-〿＀-￯])"
    r"|"
    r"(?<=[^一-鿿　-〿＀-￯]) (?=[一-鿿　-〿＀-￯])"
)


def _strip_boundary_spaces(s: str) -> str:
    """Remove spaces exclusively at CJK/non-CJK character boundaries.

    Only removes a space when one side is CJK and the other is not.
    Spaces between two CJK chars or two non-CJK chars are kept unchanged.
    """
    return _BOUNDARY_SPACE_RE.sub("", s)


def _find_source_quote(model_quote: str, source_text: str) -> str | None:
    """Find the exact source text that corresponds to a model-produced quote.

    Allows the model to insert or omit spaces at CJK/ASCII character boundaries
    (a common LLM output artifact) while still binding the evidence to the exact
    source bytes.  Content substitution, reordering, or other transformations are
    not allowed — the only normalization is boundary-space removal.

    Returns the exact substring from *source_text* that matches, or None if no
    reliable match is found.
    """
    q = model_quote.strip()
    if not q:
        return None

    # Fast path: verbatim match
    if q in source_text:
        return q

    # Normalize boundary spaces in both the quote and a sliding window of
    # source_text.  We need to map the match back to exact source positions.
    nq = _strip_boundary_spaces(q)
    if not nq or nq == q:
        # No boundary spaces in quote → no hope of normalization match
        return None

    # Build a normalized version of source_text with position mapping.
    norm_chars: list[str] = []
    src_positions: list[int] = []  # src_positions[i] = index in source_text
    i = 0
    while i < len(source_text):
        c = source_text[i]
        if c == " ":
            prev_c = source_text[i - 1] if i > 0 else ""
            next_c = source_text[i + 1] if i + 1 < len(source_text) else ""
            prev_cjk = bool(prev_c and _CJK_RE.fullmatch(prev_c))
            next_cjk = bool(next_c and _CJK_RE.fullmatch(next_c))
            # Skip space at a CJK/non-CJK boundary
            if (prev_cjk and not next_cjk) or (not prev_cjk and next_cjk):
                i += 1
                continue
        norm_chars.append(c)
        src_positions.append(i)
        i += 1

    norm_source = "".join(norm_chars)
    if nq not in norm_source:
        return None

    # Map normalized positions back to source_text positions
    nstart = norm_source.index(nq)
    nend = nstart + len(nq)
    src_start = src_positions[nstart]
    src_end = src_positions[nend - 1] + 1  # +1 for exclusive end

    return source_text[src_start:src_end]


def _identity_in_text(value, text):
    value = str(value)
    if not value:
        return False
    # Numeric identity cannot match the suffix of a different pier/span/year.
    pattern = (r"(?<!\d)" if value[0].isdigit() else "") + re.escape(value)
    pattern += r"(?!\d)" if value[-1].isdigit() else ""
    return re.search(pattern, text) is not None


def _explicit_same_defect(
    row, evidence, visible, identity, years, current_text, history_text
):
    """Require an object-bound same-defect predicate, not shared location/type.

    This deliberately narrow deterministic check does not infer continuity from
    similarity. Unsupported disease wording or multiple candidate diseases stay
    unknown until separately grounded evidence is available.
    """
    reported = str(row["fields"].get("病害类型") or row["fields"].get("病害描述") or "")
    nouns = set(ACTIVE_DEFECT_RE.findall(reported))
    if len(nouns) != 1:
        return False
    noun = next(iter(nouns))
    if noun not in current_text or noun not in history_text:
        return False
    # A model cannot quote just the favourable prefix of a contradictory source.
    # Only already visible source intervals are consulted; never unseen content.
    context = "\n".join(visible[item["source_key"]].text for item in evidence)
    if re.search(
        r"不同|另一|新增|新出现|新发现|并非|非同一|不是|不相同|不一致|不能|无法|是否|可能|疑似|未确认|未能|难以|不属于",
        context,
    ):
        return False
    predicate = re.compile(
        r"同一(?:条|处|个|项|道)?(?:的)?"
        + re.escape(noun)
        + r"(?!类型|类别|种类|形式|位置|部位|构件|编号)"
    )
    anchors = [identity[key] for key in ("bridge", "span", "component", "position")]
    anchors.extend(years)
    # Identity, both years and the relationship must occur in one quoted
    # statement. Concatenating unrelated quotes cannot manufacture a link.
    return any(
        predicate.search(statement)
        and all(_identity_in_text(anchor, statement) for anchor in anchors)
        for item in evidence
        for statement in re.split(r"[。！？;；\n]", item["quote"])
    )


def _bind_history(row, selected, visible, snapshot, source_index=None):
    """Predicted identity equality is not evidence of physical continuity."""
    temporal = row.get("temporal")
    if temporal not in ("新增", "已有", "证据不足"):
        temporal = "证据不足"
    row["temporal"] = "证据不足"
    row["temporal_verification"] = "insufficient_evidence"
    normalized = {}
    try:
        ident, historical = row.get("identity", {}), row.get("history_identity", {})
        fields = ("bridge", "span", "component", "position")
        if not isinstance(ident, dict) or not isinstance(historical, dict):
            raise ValueError("invalid history identity")
        if not all(
            isinstance(ident.get(k), str)
            and ident[k].strip()
            and ident[k] == historical.get(k)
            for k in fields
        ):
            raise ValueError("history identity differs or is incomplete")
        current_year, history_year = (
            int(ident.get("year", 0)),
            int(historical.get("year", 0)),
        )
        if not 1900 < history_year < current_year < 2200:
            raise ValueError("history years are invalid")
        normalized["current_evidence"] = _visible_evidence(
            row.get("current_evidence"),
            {selected.key: selected},
            snapshot,
            source_index,
        )
        normalized["history_evidence"] = _visible_evidence(
            row.get("history_evidence"), visible, snapshot, source_index
        )
        normalized["relationship_evidence"] = _visible_evidence(
            row.get("relationship_evidence"), visible, snapshot, source_index
        )
        if any(
            item.get("evidence_domain") == "visual-observation"
            for items in normalized.values() for item in items
        ):
            normalized = {}
            raise ValueError("visual observation cannot establish historical identity")
        supplied_refs = row.get("history_sources", [])
        supported_refs = {item["source_id"] for item in normalized["history_evidence"]}
        if (
            not isinstance(supplied_refs, list)
            or not supplied_refs
            or set(supplied_refs) != supported_refs
        ):
            raise ValueError("history reference lacks visible evidence")
        current_text = "\n".join(
            item["quote"] for item in normalized["current_evidence"]
        )
        history_text = "\n".join(
            item["quote"] for item in normalized["history_evidence"]
        )
        relation_text = "\n".join(
            item["quote"] for item in normalized["relationship_evidence"]
        )
        for identity, text in ((ident, current_text), (historical, history_text)):
            if not all(_identity_in_text(identity[k], text) for k in (*fields, "year")):
                raise ValueError("quoted evidence does not establish identity")
        if not all(
            _identity_in_text(ident[k], relation_text) for k in fields
        ) or not all(
            _identity_in_text(year, relation_text)
            for year in (current_year, history_year)
        ):
            raise ValueError("relationship quote lacks object and years")
        # These patterns deliberately prefer unknown over inventing continuity.
        # A checked match is explicit source support, not a statistical model confidence.
        if re.search(
            r"不是|非同一|无法|不能|不属于|不确定|未确认|未能|难以|是否|可能|疑似|待确认",
            relation_text,
        ):
            raise ValueError("relationship is explicitly uncertain or negative")
        if temporal == "已有" and not _explicit_same_defect(
            row,
            normalized["relationship_evidence"],
            visible,
            ident,
            (current_year, history_year),
            current_text,
            history_text,
        ):
            raise ValueError("no object-bound same-defect relationship")
        if temporal == "新增" and not (
            "新增" in relation_text
            and re.search(r"未发现|未见|无此|不存在", history_text)
        ):
            raise ValueError("new defect lacks explicit historical absence")
        row["temporal"] = temporal
        row["temporal_verification"] = (
            "explicit_visible_evidence"
            if temporal != "证据不足"
            else "insufficient_evidence"
        )
    except (ValueError, TypeError, OverflowError):
        pass
    # Discard model-supplied unvalidated evidence objects. Retain only precise,
    # checked references, and distinguish them from a verified relationship.
    for name in ("current_evidence", "history_evidence", "relationship_evidence"):
        row[name] = normalized.get(name, [])
    row["history_sources"] = list(
        dict.fromkeys(item["source_id"] for item in row["history_evidence"])
    )
    row["fields"]["是否新增"] = {"新增": "是", "已有": "否", "证据不足": None}[
        row["temporal"]
    ]
    if row["temporal"] == "证据不足":
        row["fields"]["上一次定检状态"] = None
        row["fields"]["发展程度"] = None


_SCOPE_KEYS = {"bridge", "span", "component", "position", "year", "rating_system"}
_VISUAL_MEASURE_RE = re.compile(
    r"\d|[零〇○一二两俩三四五六七八九十百千万半数几点]+\s*"
    r"(?:年|毫米|厘米|公分|公尺|米|mm|cm|m|号|跨|墩|幅)",
    re.IGNORECASE,
)


_PLACEHOLDER_VALUES = frozenset({
    "无", "未知", "不详", "未提供", "暂无", "暂无数据", "无数据", "未提及", "不适用", "unknown",
    "none", "null", "n/a", "na", "-", "--", "...", "…", "待填写", "待补充",
})
_NUMBER_RE = re.compile(r"[+-]?\d+(?:\.\d+)?")


def _absent_value(value):
    return _empty_basic(value) or (
        isinstance(value, str) and value.strip().casefold() in _PLACEHOLDER_VALUES
    )


_GRADE_VALUE_RE = re.compile(r"\s*([A-Ea-e])\s*[级类]?\s*")


def _grade_in_text(letter, text):
    """A one-letter grade only counts next to 级/类 or right after a 等级 label."""
    pattern = (r"(?<![A-Za-z0-9])" + letter + r"\s*[级类]"
               + r"|(?:等级|评定|评级)[^\n。；;]{0,6}?(?<![A-Za-z0-9])" + letter + r"(?![A-Za-z0-9])")
    match = re.search(pattern, text)
    return match.group(0) if match else None


def _supported_value(v, quote, source, key=""):
    """(value, verbatim quote) supported by one source slice, or None."""
    if isinstance(quote, str) and quote.strip():
        resolved = _find_source_quote(quote, source.text)
        if resolved is not None:
            if isinstance(v, str) and not _scalar_in_quote(v, resolved):
                compact = _strip_boundary_spaces(v)
                if _scalar_in_quote(compact, resolved):
                    v = compact
            if _scalar_in_quote(v, resolved):
                return v, resolved
    # Fall back to the value itself; numbers keep their digit boundaries and very
    # short words are too ambiguous to locate (e.g. grade "C" inside "C30").
    rendered = str(v).strip()
    grade = _GRADE_VALUE_RE.fullmatch(rendered) if key.endswith("等级") and isinstance(v, str) else None
    if grade:
        found = _grade_in_text(grade.group(1).upper(), source.text)
        return (v, found) if found else None
    if not isinstance(v, str) or _NUMBER_RE.fullmatch(rendered):
        return (v, rendered) if _scalar_in_quote(rendered, source.text) else None
    if len(rendered) >= 2:
        resolved = _find_source_quote(rendered, source.text)
        if resolved is not None:
            return resolved, resolved
    return None


def _basic_claim(v, ref, quote, scope, task, snapshot, sources, visible, owned, *, scoped, key=""):
    """Bind one basic claim to owned text; None drops only this claim."""
    if not _scalar(v):
        return None
    if quote is None and isinstance(ref, dict):
        quote = ref.get("quote")
    if isinstance(ref, str):
        matched = [source for source in task.owned if source.key == ref]
        ref = ({"source_id": matched[0].source_id, "source_key": ref}
               if len(matched) == 1 else {"source_id": ref})
    textual = [s for s in task.owned if sources[s.source_id].kind != "visual-observation"]
    named = []
    if isinstance(ref, dict):
        wanted = ref.get("source_key")
        named = [s for s in textual if (isinstance(wanted, str) and s.key == wanted)
                 or (wanted is None and ref.get("source_id") in owned and s.source_id == ref.get("source_id"))]
    accepted = None
    for source in named:
        found = _supported_value(v, quote, source, key)
        if found is not None:
            accepted = source, *found
            break
    if accepted is None and len(named) == 1 and quote is None:
        # Historical contract: a designated owned slice without a quote is slice evidence.
        accepted = named[0], v, None
    if accepted is None:
        for source in textual:
            found = _supported_value(v, quote, source, key)
            if found is not None:
                accepted = source, *found
                break
    if accepted is None:
        return None
    source, v, resolved = accepted
    if resolved is None:
        evidence = _range(source, snapshot, sources)
    else:
        evidence = _visible_evidence(
            [{"source_key": source.key, "quote": resolved}], visible, snapshot, sources
        )[0]
    if not (
        isinstance(scope, dict)
        and set(scope) <= _SCOPE_KEYS
        and all(
            isinstance(item, (str, int)) and str(item).strip()
            and resolved is not None and _identity_in_text(item, resolved)
            for item in scope.values()
        )
    ):
        if scoped:
            # A multi-object observation without its scope would masquerade as the
            # whole bridge's value; drop it rather than generalize it.
            return None
        scope = {}
    return v, source.source_id, evidence, scope


def _row_source(row, quotes, fields, task, owned):
    key, source_id = row.get("source_key"), row.get("source_id")
    if isinstance(key, str):
        keyed = [s for s in task.owned if s.key == key]
        if len(keyed) == 1 and source_id in (None, keyed[0].source_id):
            return keyed[0]
    texts = [q for q in quotes.values() if isinstance(q, str) and q.strip()] or [
        v for k, v in fields.items() if k != "序号" and isinstance(v, str) and len(v.strip()) >= 2
    ]
    choices = ([s for s in task.owned if s.source_id == source_id]
               if isinstance(source_id, str) and source_id in owned else [])
    if len(choices) == 1:
        return choices[0]
    for source in choices or task.owned:
        if any(_find_source_quote(q, source.text) is not None for q in texts):
            return source
    return None


def _parse_row(section, row, task, owned, snapshot, sources, visible):
    """Normalize one model row; None drops only this row."""
    if not isinstance(row, dict):
        return None
    if "fields" not in row:
        # Flat rows (business fields at the row's top level) from whole-table habits.
        row["fields"] = {k: row.pop(k) for k in list(row) if k in ROW_FIELDS[section]}
    if not isinstance(row["fields"], dict):
        return None
    fields = {k: v for k, v in row["fields"].items() if isinstance(k, str) and _scalar(v)}
    quotes = row.get("quotes", {})
    quotes = {k: q for k, q in quotes.items() if isinstance(k, str)} if isinstance(quotes, dict) else {}
    if section == "维修建议":
        for mapping in (fields, quotes):
            if "建议" in mapping and "建议内容" not in mapping:
                mapping["建议内容"] = mapping.pop("建议")
    allowed = set(fields) | set(ROW_FIELDS[section])
    quotes = {k: q for k, q in quotes.items() if k in allowed}
    if "instance_id" in row and (
        type(row["instance_id"]) not in (str, int) or not str(row["instance_id"]).strip()
    ):
        row.pop("instance_id")
    selected = _row_source(row, quotes, fields, task, owned)
    if selected is None:
        return None
    for key, quote in list(quotes.items()):
        resolved = (_find_source_quote(quote, selected.text)
                    if isinstance(quote, str) and quote.strip() else None)
        if resolved is None:
            del quotes[key]
            if isinstance(quote, str) and quote.strip():
                fields.setdefault(key, quote.strip())
            continue
        fields[key] = quotes[key] = resolved
    unverified = []
    for key, item in list(fields.items()):
        if key in quotes or key == "序号" or _empty_basic(item):
            continue
        resolved = (_find_source_quote(item, selected.text)
                    if isinstance(item, str) and len(item.strip()) >= 2 else None)
        if resolved is None:
            unverified.append(key)  # kept for recall, never presented as evidence
        else:
            fields[key] = quotes[key] = resolved
    if not any(key != "序号" and not _empty_basic(item) for key, item in fields.items()):
        return None
    row.update(fields=fields, quotes=quotes, source_id=selected.source_id, source_key=selected.key,
               source_ranges=[_range(selected, snapshot, sources)])
    if unverified:
        row["unverified_fields"] = unverified
    if sources[selected.source_id].kind == "visual-observation":
        if (
            section != "病害明细"
            or any(key not in {"病害类型", "病害描述"} and not _empty_basic(item)
                   for key, item in fields.items())
            or any(not _empty_basic(item) and key not in quotes for key, item in fields.items())
            or any(_VISUAL_MEASURE_RE.search(str(item))
                   for item in fields.values() if not _empty_basic(item))
        ):
            return None
        row.pop("identity", None)
        row.pop("history_identity", None)
        row["evidence_domain"] = "visual-observation"
    if section == "病害明细":
        try:
            _bind_history(row, selected, visible, snapshot, sources)
        except (KeyError, AttributeError, IndexError):
            pass  # malformed history claims keep the default 证据不足
    return row


_FLAT_OBJECT_RE = re.compile(r"\{[^{}]*\}")
_EVIDENCE_OBJECT_KEYS = frozenset({
    "quotes", "basic_quotes", "basic_sources", "basic_scopes", "identity", "history_identity", "scope",
})


def _salvage_flat_objects(raw):
    """Recover every self-contained flat object from malformed fragment JSON.

    Small models break one brace in long compact output; flat rows and the
    basic object remain individually valid. Raises if nothing is recoverable.
    """
    value = {"基本信息": {}, "维修建议": [], "病害明细": []}
    basic = re.search(r'"基本信息"\s*:\s*(\{[^{}]*\})', raw)
    found = False
    if basic:
        try:
            value["基本信息"] = parse_json_object(basic.group(1), lenient=True)
            found = True
        except ValueError:
            pass
    for match in _FLAT_OBJECT_RE.finditer(raw):
        if basic and match.start() == basic.start(1):
            continue
        owner = re.search(r'"([^"]+)"\s*:\s*$', raw[max(0, match.start() - 40):match.start()])
        if owner and owner.group(1) in _EVIDENCE_OBJECT_KEYS:
            continue
        try:
            item = parse_json_object(match.group(0), lenient=True)
        except ValueError:
            continue
        if "建议内容" in item or "建议" in item:
            value["维修建议"].append(item)
            found = True
        elif {"病害部位", "病害类型", "病害描述"} & set(item):
            value["病害明细"].append(item)
            found = True
    if not found:
        raise json.JSONDecodeError("no recoverable fragment object", raw, 0)
    return value


def parse_fragment(raw, task, snapshot, call_id=""):
    """Bind one fragment response to its task; content defects drop single items.

    Only unparseable JSON or task/snapshot inconsistency rejects the fragment: a
    9B model at temperature 0 repeats the same defect on retry, so rejecting the
    whole chunk turned one bad quote into a failed report.
    """
    try:
        value = parse_json_object(raw, close_unbalanced=True, lenient=True)
    except json.JSONDecodeError:
        value = _salvage_flat_objects(raw)
    # Ownership comes from the task, not from the model echoing every key: the 9B
    # model omits empty/object-only slices from `covered` deterministically, and an
    # echo proves no extraction anyway. Rows are still bound to owned slices below.
    covered = list(task.owned_keys)
    sources = snapshot.sources()
    visible = {source.key: source for source in [*task.owned, *task.context]}
    for source in visible.values():
        original = sources.get(source.source_id)
        if (
            original is None
            or not 0 <= source.start <= source.end <= len(original.text)
            or original.text[source.start : source.end] != source.text
        ):
            raise ValueError("task source does not match snapshot")
    owned = {source.source_id for source in task.owned}

    def mapping(name):
        item = value.get(name, {})
        return item if isinstance(item, dict) else {}

    basic = {k: v for k, v in mapping("基本信息").items() if isinstance(k, str)}
    refs, quotes, scopes = mapping("basic_sources"), mapping("basic_quotes"), mapping("basic_scopes")
    basic_evidence, observations, basic_refs = {}, [], {}
    claims = [
        (key, v, refs.get(key), quotes.get(key), scopes.get(key, {}))
        for key, v in basic.items()
    ]
    basic_claims = len(claims)
    extra = value.get("basic_observations", [])
    for item in extra if isinstance(extra, list) else []:
        if isinstance(item, dict) and isinstance(item.get("field"), str) and item["field"].strip():
            claims.append((item["field"], item.get("value"), item, item.get("quote"),
                           item.get("scope", {})))
    for claim_index, (key, v, ref, quote, scope) in enumerate(claims):
        is_basic = claim_index < basic_claims
        if _absent_value(v):
            # The trained full-table contract includes null slots for missing
            # fields. They assert no fact and must not fabricate source evidence.
            # An extra empty observation must not erase a populated basic field.
            if is_basic or key not in basic or _absent_value(basic[key]):
                basic[key] = None
            continue
        accepted = _basic_claim(v, ref, quote, scope, task, snapshot, sources, visible, owned,
                                scoped=not is_basic, key=key)
        if accepted is None:
            if is_basic:
                basic[key] = None
            continue
        v, source_id, evidence, scope = accepted
        if is_basic:
            basic[key] = v
        basic_refs.setdefault(key, source_id)
        basic_evidence.setdefault(key, evidence)
        observations.append(
            {"field": key, "value": v, "scope": scope, "evidence": evidence, "call_id": call_id}
        )
    rows = []
    for section in ("维修建议", "病害明细"):
        entries = value.get(section)
        parsed = (_parse_row(section, row, task, owned, snapshot, sources, visible)
                  for row in (entries if isinstance(entries, list) else []))
        rows.append([row for row in parsed if row is not None])
    return ExtractionFragment(
        task.task_id,
        covered,
        basic,
        *rows,
        basic_refs,
        call_id,
        basic_evidence,
        observations,
    )


def _meaningful_value(value):
    if isinstance(value, str):
        return not _absent_value(value) and any(character.isalnum() for character in value)
    return type(value) is int or (type(value) is float and math.isfinite(value))


def resolve_basic_conflicts(merged):
    """Flat value for conflicts no model repair settled: most fragments, then document order.

    Cover and summary pages precede historical tables, so earlier observations win ties.
    The executor and the audit apply this same rule to the same merge.
    """
    for item in merged["conflicts"]:
        counts = {}
        for observation in item["observations"]:
            counts[observation["value"]] = counts.get(observation["value"], 0) + 1
        values = item["values"]
        best = max(values, key=lambda value: (counts.get(value, 0), -values.index(value)))
        merged["data"]["基本信息"][item["field"]] = best
        merged["basic_ambiguities"].append(
            {**item, "resolved_value": best, "reason": "resolved_by_majority_then_document_order"}
        )
    merged["conflicts"] = []
    return merged


def merge_fragments(snapshot, plan, fragments):
    if (
        snapshot.digest != plan.snapshot_sha256
        or not plan.coverage(snapshot)["complete"]
    ):
        raise ValueError("snapshot/plan coverage mismatch")
    expected = {task.task_id: task for task in plan.tasks}
    if len(fragments) != len(expected) or {f.task_id for f in fragments} != set(
        expected
    ):
        raise ValueError("fragment coverage incomplete")
    data = {"基本信息": {}, "维修建议": [], "病害明细": []}
    provenance, conflicts, seen = [], [], {}
    observations, ambiguities = [], []
    task_order = {task.task_id: index for index, task in enumerate(plan.tasks)}
    source_index = snapshot.sources()
    for frag in sorted(fragments, key=lambda f: task_order[f.task_id]):
        if sorted(frag.covered) != sorted(expected[frag.task_id].owned_keys):
            raise ValueError("fragment coverage incomplete")
        task = expected[frag.task_id]
        fragment_observations = [
            item
            for item in frag.basic_observations
            if not _empty_basic(item.get("value"))
        ]
        covered_basic = {item["field"] for item in fragment_observations}
        for key, value in frag.basic.items():
            if _empty_basic(value):
                data["基本信息"].setdefault(key, None)
                continue
            if key in covered_basic:
                continue
            evidence = frag.basic_evidence.get(key, {})
            source_id = frag.basic_sources.get(key)
            choices = [source for source in task.owned if source.source_id == source_id]
            if not evidence and len(choices) == 1:
                evidence = _range(choices[0], snapshot, source_index)
            fragment_observations.append(
                {
                    "field": key,
                    "value": value,
                    "scope": {},
                    "evidence": evidence,
                    "call_id": frag.call_id,
                }
            )
        for observation_index, observation in enumerate(fragment_observations):
            key, evidence = observation["field"], observation.get("evidence", {})
            observations.append(observation)
            provenance.append(
                {
                    "section": "基本信息",
                    "field": key,
                    "number": 1,
                    "source_id": evidence.get("source_id"),
                    "source_key": evidence.get("source_key"),
                    "source_ranges": [evidence] if evidence else [],
                    "instance_id": f"{key}:{observation_index}",
                    "scope": observation.get("scope", {}),
                    "quotes": {key: evidence["quote"]} if "quote" in evidence else {},
                    "call_id": frag.call_id,
                }
            )
        for section, rows in (("维修建议", frag.advice), ("病害明细", frag.diseases)):
            for index, row in enumerate(rows):
                # Distinct source occurrences survive, even with identical text.
                key = (
                    section,
                    row.get("source_key", frag.task_id + ":" + row["source_id"]),
                    str(row.get("instance_id", index)),
                )
                if key in seen:
                    if seen[key] != row:
                        raise ValueError(
                            "conflicting source instance requires distinct instance_id"
                        )
                    continue
                seen[key] = row
                content = ("建议内容",) if section == "维修建议" else ("病害部位", "病害类型", "病害描述")
                if not any(_meaningful_value(row["fields"].get(name)) for name in content):
                    continue  # the report contract requires row content; nothing to publish
                fields = dict(row["fields"])
                fields["序号"] = len(data[section]) + 1
                data[section].append(fields)
                provenance.append(
                    {
                        "section": section,
                        "number": fields["序号"],
                        "source_id": row["source_id"],
                        "source_key": row.get("source_key"),
                        "source_ranges": row.get("source_ranges")
                        or [
                            _range(source, snapshot, source_index)
                            for source in task.owned
                            if source.source_id == row["source_id"]
                            and (
                                not row.get("source_key")
                                or source.key == row["source_key"]
                            )
                        ],
                        "instance_id": str(row.get("instance_id", index)),
                        "quotes": row.get("quotes", {}),
                        "history_sources": row.get("history_sources", []),
                        "history_identity": row.get("history_identity"),
                        "current_evidence": row.get("current_evidence", []),
                        "history_evidence": row.get("history_evidence", []),
                        "relationship_evidence": row.get("relationship_evidence", []),
                        "temporal_verification": row.get(
                            "temporal_verification", "unverified"
                        ),
                        "call_id": frag.call_id,
                        "identity": row.get("identity"),
                        "temporal": row.get("temporal"),
                    }
                )
    # Scope is part of a fact's identity, never a license to choose one year or
    # bridge side for a flat output. Distinct valid scopes remain observations;
    # a flat field is populated only when all its supported values agree.
    by_field = {}
    for observation in observations:
        by_field.setdefault(observation["field"], []).append(observation)
    for key, group in by_field.items():
        populated = [item for item in group if item["value"] is not None]
        values = list(dict.fromkeys(item["value"] for item in populated))
        data["基本信息"][key] = values[0] if len(values) == 1 else None
        if len(values) < 2:
            continue
        by_scope = {}
        for item in populated:
            by_scope.setdefault(json_digest(item.get("scope", {})), []).append(item)
        scope_conflict = any(
            len({item["value"] for item in scoped}) > 1 for scoped in by_scope.values()
        )
        unscoped = any(not item.get("scope") for item in populated)
        item = {
            "field": key,
            "values": values,
            "observations": populated,
            "source_id": populated[-1].get("evidence", {}).get("source_id"),
        }
        if scope_conflict or unscoped:
            conflicts.append(item)
        else:
            ambiguities.append(
                {**item, "reason": "multiple_scopes_require_explicit_projection"}
            )
    return {
        "data": data,
        "provenance": provenance,
        "basic_observations": observations,
        "basic_ambiguities": ambiguities,
        "conflicts": conflicts,
        "coverage": plan.coverage(snapshot),
        "contributing_call_ids": [f.call_id for f in fragments if f.call_id],
    }
