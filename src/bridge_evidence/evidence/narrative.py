from __future__ import annotations

import re

from bridge_evidence.models.schema import DocumentState, NarrativeClaim


_MECHANISMS = (
    (re.compile(r"主梁|横向联系"), re.compile(r"裂缝|开裂"), "反复荷载、温度作用或局部受力可能促使裂缝发展", "可能影响构件耐久性，持续发展时需复核结构受力"),
    (re.compile(r"主梁|桥墩|桥台"), re.compile(r"露筋|锈蚀|剥落|破损"), "保护层失效和水分侵入可能加速钢筋锈蚀", "可能削弱局部耐久性并扩大混凝土损伤"),
    (re.compile(r"支座"), re.compile(r"变形|位移|脱空|开裂"), "支座老化、约束异常或梁体位移可能造成传力状态变化", "可能影响上、下部结构之间的正常传力"),
    (re.compile(r"伸缩缝|桥面铺装"), re.compile(r"破损|坑槽|缺失|堵塞"), "车辆冲击、材料老化和排水不畅可能加剧局部损伤", "可能影响行车平顺性并诱发水损害"),
    (re.compile(r"护栏|栏杆"), re.compile(r"破损|缺失|松动|锈蚀"), "碰撞、连接松动或材料老化可能降低防护构件完整性", "可能降低临边防护能力，应优先核查"),
)


class NarrativePlanner:
    """Plan evidence-bound mechanism claims before deterministic or LLM writing."""

    @staticmethod
    def plan(state: DocumentState, *, limit: int = 4) -> None:
        ranked = sorted(
            state.diseases,
            key=lambda item: (
                bool(re.search(r"严重|贯通|脱空|缺失|露筋|变形", f"{item.type}{item.description}")),
                len(item.event_ids),
                item.confidence,
            ),
            reverse=True,
        )
        claims: list[NarrativeClaim] = []
        seen: set[tuple[str, str]] = set()
        for disease in ranked:
            text = f"{disease.component}{disease.type}{disease.description}"
            rule = next(
                (
                    (cause, impact)
                    for component, defect, cause, impact in _MECHANISMS
                    if component.search(disease.component) and defect.search(text)
                ),
                None,
            )
            if rule is None:
                continue
            key = (disease.component, disease.type)
            if key in seen:
                continue
            seen.add(key)
            cause, impact = rule
            prefix = disease.location or disease.component
            claims.extend([
                NarrativeClaim(
                    claim_type="risk",
                    text=f"{prefix}{disease.type}，{impact}",
                    disease_event_ids=list(disease.event_ids),
                    evidence=list(disease.evidence),
                    confidence=max(0.72, disease.confidence),
                ),
                NarrativeClaim(
                    claim_type="cause",
                    text=cause,
                    disease_event_ids=list(disease.event_ids),
                    evidence=list(disease.evidence),
                    confidence=min(0.82, max(0.62, disease.confidence)),
                ),
            ])
            if len(seen) >= limit:
                break
        state.narrative.claims = claims
        risk_texts = [claim.text for claim in claims if claim.claim_type == "risk"]
        cause_texts = [claim.text for claim in claims if claim.claim_type == "cause"]
        # The extractor's compact count summary preserves broad disease
        # coverage.  Mechanism claims are supporting evidence, not a reason to
        # replace that summary with a few long template sentences.
        if risk_texts and not state.narrative.key_risks:
            state.narrative.key_risks = "；".join(risk_texts) + "。"
        if cause_texts and not state.narrative.causes:
            state.narrative.causes = cause_texts
        if risk_texts and not state.narrative.safety_impacts:
            state.narrative.safety_impacts = risk_texts
