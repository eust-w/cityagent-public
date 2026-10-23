from __future__ import annotations

import re


# These are presentation/photo-caption labels, not technical-condition
# diseases.  Keep this list deliberately narrow: the guard may reject a row,
# so it must not treat a legitimate defect synonym as decoration.
NON_DISEASE_LABELS = frozenset({
    "工作照",
    "施工照",
    "整体照",
    "全景照",
    "桥梁现状",
    "桥涵标现状",
    "修复现状",
    "修补现状",
    "现状",
    "外观",
    "全景",
    "照片",
})

ACTIVE_DEFECT_RE = re.compile(
    r"裂缝|开裂|破损|剥落|剝落|脱落|渗水|泛碱|锈蚀|腐蚀|露筋|漏筋|"
    r"沉降|位移|空鼓|空洞|坑洼|坑槽|破洞|松动|缺损|缺失|脱空|"
    r"堵塞|积水|磨损|老化|变形|错台|鼓包|剥离|剝离|麻面|蜂窝|划痕|翘曲"
)

_DISEASE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"破损.{0,4}(?:露筋|漏筋)|(?:露筋|漏筋).{0,4}破损"), "破损、露筋"),
    (re.compile(r"渗水.{0,4}泛碱|泛碱.{0,4}渗水"), "渗水泛碱"),
    (re.compile(r"蜂窝.{0,3}麻面|麻面.{0,3}蜂窝"), "蜂窝麻面"),
    (re.compile(r"剥落.{0,3}掉角|掉角.{0,3}剥落"), "剥落、掉角"),
    (re.compile(r"空洞.{0,3}(?:露筋|漏筋)|(?:露筋|漏筋).{0,3}空洞"), "空洞、漏筋"),
    (re.compile(r"涂层.{0,3}剥落"), "涂层剥落"),
    (re.compile(r"涂层.{0,3}开裂"), "涂层开裂"),
    (re.compile(r"裂缝|开裂"), "裂缝"),
    (re.compile(r"破损|缺损|掉角"), "破损"),
    (re.compile(r"剥落|剝落|脱落"), "剥落"),
    (re.compile(r"渗水"), "渗水"),
    (re.compile(r"泛碱"), "泛碱"),
    (re.compile(r"锈蚀|腐蚀"), "锈蚀"),
    (re.compile(r"露筋|漏筋"), "露筋"),
    (re.compile(r"脱空"), "脱空"),
    (re.compile(r"缺失"), "缺失"),
    (re.compile(r"空鼓"), "空鼓"),
    (re.compile(r"蜂窝|麻面"), "蜂窝麻面"),
    (re.compile(r"坑槽|坑洼|破洞"), "坑槽"),
    (re.compile(r"堵塞"), "堵塞"),
    (re.compile(r"积水"), "积水"),
    (re.compile(r"松动"), "松动"),
    (re.compile(r"老化"), "老化"),
    (re.compile(r"变形|翘曲"), "变形"),
    (re.compile(r"错台"), "错台"),
    (re.compile(r"磨损"), "磨损"),
)


def compact_label(value: str) -> str:
    return re.sub(r"[\s，,。；;：:（）()、|_\-]+", "", str(value or ""))


def is_non_disease_label(value: str) -> bool:
    compact = compact_label(value)
    if compact in NON_DISEASE_LABELS:
        return True
    # OCR/history round-trips can duplicate a presentation token (for example
    # ``现状`` -> ``现状现状``).  It is still a caption, not a defect.  Keep the
    # match anchored to the whole label so legitimate text such as
    # ``现状裂缝`` is not discarded.
    if re.fullmatch(r"(?:桥梁|桥涵标|修复|修补)?(?:现状)+", compact):
        return True
    return bool(re.fullmatch(r"(?:照片)?\d*(?:工作照|施工照|整体照|全景照|现状照片)", compact))


def is_condition_state_label(value: str) -> bool:
    """Return true only for a bare, semantically meaningful condition label.

    Official labels occasionally keep rows such as ``人行道 | 现状 |
    外观良好``.  Those rows are condition observations rather than defects,
    but they still belong to the requested detailed ledger.  The deliberately
    narrow exact match prevents photo captions such as ``桥涵标现状`` or
    ``修复现状`` from re-entering the result.
    """
    return compact_label(value) in {"现状", "外观"}


def is_meaningful_condition_state(value: str, description: str) -> bool:
    """Distinguish an actual condition observation from a tautological caption."""
    if not is_condition_state_label(value):
        return False
    text = str(description or "").strip()
    return bool(
        text
        and re.search(r"良好|正常|完好|稳定|未见明显|无明显|无病害|无异常", text)
    )


def canonical_disease_from_text(value: str) -> str | None:
    for pattern, canonical in _DISEASE_PATTERNS:
        if pattern.search(str(value or "")):
            return canonical
    return None
