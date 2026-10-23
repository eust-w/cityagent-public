"""Parameterized, source-cited report queries; no labels or generated SQL."""

import json
import math
import sqlite3
from pathlib import Path

from .facts import assess_relation, build_fact_graph, normalize_identity
from .run_state import json_digest


class ReportIndex:
    def __init__(self, path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS facts (id TEXT PRIMARY KEY, document TEXT, bridge TEXT, year INTEGER, component TEXT, disease TEXT, source TEXT, fields TEXT)"
            )
            if "document_sha256" not in {
                row[1] for row in db.execute("PRAGMA table_info(facts)")
            }:
                db.execute("ALTER TABLE facts ADD COLUMN document_sha256 TEXT")
                for fid, document in db.execute(
                    "SELECT id,document FROM facts"
                ).fetchall():
                    db.execute(
                        "UPDATE facts SET document_sha256=? WHERE id=?",
                        (json.loads(document).get("sha256"), fid),
                    )
            db.execute(
                "CREATE INDEX IF NOT EXISTS facts_document ON facts(document_sha256)"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS facts_filter ON facts(bridge,year,component,disease)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS graphs (document TEXT PRIMARY KEY, graph TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS observations (id TEXT PRIMARY KEY, document TEXT, entity TEXT, bridge TEXT, year INTEGER, kind TEXT, attribute TEXT, value TEXT, evidence TEXT)"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS observation_filter ON observations(bridge,year,kind,attribute,entity)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS measurements (id TEXT PRIMARY KEY, observation TEXT, bridge TEXT, year INTEGER, component TEXT, disease TEXT, attribute TEXT, value REAL, lower REAL, upper REAL, unit TEXT, statistic TEXT, payload TEXT)"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS measurement_filter ON measurements(bridge,year,attribute,unit,statistic,value)"
            )

    def connect(self):
        return sqlite3.connect(self.path)

    def ingest(self, record, *, allowed_sha256, strict=False):
        doc = record.get("document", {})
        if record.get("status") != "success" or doc.get("sha256") not in allowed_sha256:
            raise ValueError(
                "only successful extraction from explicitly allowed reports may be indexed"
            )
        # Always rebuild; never trust a self-supplied serialized facts field.
        graph = build_fact_graph(
            record, strict=strict, allowed_sha256=allowed_sha256
        ).to_dict()
        basic = record["final_data"]["基本信息"]
        bridge = normalize_identity(basic.get("桥梁名称", ""), "")["bridge"]
        if not bridge or not graph["document"]["report_year"]:
            raise ValueError("bridge/year identity required")
        evidence = {e["id"]: e for e in graph["evidence"]}
        entities = {e["id"]: e for e in graph["entities"]}
        key, serialized_doc = doc["sha256"], json.dumps(doc, sort_keys=True)
        with self.connect() as db:
            # Digest identity is independent of filename and caller path spelling.
            old = db.execute(
                "SELECT id FROM observations WHERE document=?", (key,)
            ).fetchall()
            for (oid,) in old:
                db.execute("DELETE FROM measurements WHERE observation=?", (oid,))
            db.execute("DELETE FROM observations WHERE document=?", (key,))
            db.execute("DELETE FROM facts WHERE document_sha256=?", (key,))
            db.execute(
                "INSERT OR REPLACE INTO graphs VALUES (?,?)",
                (key, json.dumps(graph, ensure_ascii=False)),
            )
            for obs in graph["observations"]:
                year = obs["inspection_year"] or (
                    obs["report_year"] if obs["time_basis"] == "report-only" else None
                )
                refs = [evidence[e] for e in obs["evidence_ids"]]
                row = obs["raw"]
                component = (
                    str(row.get("病害部位", ""))
                    if isinstance(row, dict)
                    else entities[obs["entity_id"]]["raw_location"]
                )
                disease = (
                    str(row.get("病害类型", obs["section"]))
                    if isinstance(row, dict)
                    else ""
                )
                db.execute(
                    "INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        obs["id"],
                        key,
                        obs["entity_id"],
                        bridge,
                        year,
                        obs["kind"],
                        obs["attribute"],
                        json.dumps(row, ensure_ascii=False),
                        json.dumps(refs, ensure_ascii=False),
                    ),
                )
                if obs["kind"] in ("defect", "recommendation"):
                    db.execute(
                        "INSERT INTO facts VALUES (?,?,?,?,?,?,?,?,?)",
                        (
                            obs["id"],
                            serialized_doc,
                            bridge,
                            year,
                            component,
                            disease,
                            json.dumps(
                                {"evidence": refs, "time_basis": obs["time_basis"]},
                                ensure_ascii=False,
                            ),
                            json.dumps(row, ensure_ascii=False),
                            key,
                        ),
                    )
                for index, measurement in enumerate(obs["measurements"]):
                    payload = {
                        **measurement,
                        "observation_id": obs["id"],
                        "entity_id": obs["entity_id"],
                        "document_sha256": key,
                        "evidence": refs,
                        "time_basis": obs["time_basis"],
                        "year": year,
                        "component": component,
                        "disease": disease,
                        "identity": entities[obs["entity_id"]]["identity"],
                        "defect_identity_id": obs.get("defect_identity_id"),
                        "reported_defect_id": row.get("病害编号")
                        if isinstance(row, dict)
                        else None,
                    }
                    db.execute(
                        "INSERT INTO measurements VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            json_digest([obs["id"], index]),
                            obs["id"],
                            bridge,
                            year,
                            component,
                            disease,
                            measurement["attribute"],
                            measurement["value"],
                            measurement["lower"],
                            measurement["upper"],
                            measurement["unit"],
                            measurement["statistic"],
                            json.dumps(payload, ensure_ascii=False),
                        ),
                    )

    @staticmethod
    def _where(filters):
        clauses, values = [], []
        for field, value in filters:
            if value is not None:
                clauses.append(field + "=?")
                values.append(value)
        return clauses, values

    def query(self, *, bridge=None, year=None, component=None, disease=None):
        if bridge is not None:
            bridge = normalize_identity(bridge, "")["bridge"]
        clauses, values = self._where(
            (
                ("bridge", bridge),
                ("year", year),
                ("component", component),
                ("disease", disease),
            )
        )
        with self.connect() as db:
            rows = db.execute(
                "SELECT document,bridge,year,component,disease,source,fields FROM facts"
                + (" WHERE " + " AND ".join(clauses) if clauses else "")
                + " ORDER BY bridge,year,id",
                values,
            ).fetchall()
        return {
            "status": "supported" if rows else "insufficient-evidence",
            "count": len(rows),
            "count_basis": "source-occurrences",
            "sources": [
                {
                    "document": json.loads(r[0]),
                    "bridge": r[1],
                    "year": r[2],
                    "source": json.loads(r[5]),
                }
                for r in rows
            ],
            "facts": [json.loads(r[6]) for r in rows],
        }

    def scores(self, *, bridge=None, year=None, attribute=None):
        clauses, values = self._where(
            (
                (
                    "bridge",
                    normalize_identity(bridge, "")["bridge"] if bridge else None,
                ),
                ("year", year),
                ("kind", "score"),
                ("attribute", attribute),
            )
        )
        with self.connect() as db:
            rows = db.execute(
                "SELECT attribute,value,evidence,entity,year FROM observations WHERE "
                + " AND ".join(clauses)
                + " ORDER BY year,id",
                values,
            ).fetchall()
        return {
            "status": "supported" if rows else "insufficient-evidence",
            "count": len(rows),
            "facts": [
                {
                    "attribute": r[0],
                    "value": json.loads(r[1]),
                    "evidence": json.loads(r[2]),
                    "entity_id": r[3],
                    "year": r[4],
                }
                for r in rows
            ],
        }

    def measurements(
        self,
        *,
        bridge=None,
        year=None,
        component=None,
        disease=None,
        attribute=None,
        unit=None,
        statistic=None,
        minimum=None,
        maximum=None,
    ):
        clauses, values = self._where(
            (
                (
                    "bridge",
                    normalize_identity(bridge, "")["bridge"] if bridge else None,
                ),
                ("year", year),
                ("component", component),
                ("disease", disease),
                ("attribute", attribute),
                ("unit", unit),
                ("statistic", statistic),
            )
        )
        for op, bound in ((">=", minimum), ("<=", maximum)):
            if bound is not None:
                if (
                    isinstance(bound, bool)
                    or not isinstance(bound, (int, float))
                    or not math.isfinite(bound)
                ):
                    raise ValueError("finite numerical measurement threshold required")
                if not unit:
                    raise ValueError(
                        "canonical unit required for a numerical threshold"
                    )
                # Ranges/inequalities are deliberately not compared as scalars.
                clauses.append("value" + op + "?")
                values.append(bound)
        with self.connect() as db:
            rows = db.execute(
                "SELECT payload FROM measurements"
                + (" WHERE " + " AND ".join(clauses) if clauses else "")
                + " ORDER BY year,id",
                values,
            ).fetchall()
        return {
            "status": "supported" if rows else "insufficient-evidence",
            "count": len(rows),
            "count_basis": "measurement-occurrences",
            "measurements": [json.loads(r[0]) for r in rows],
        }

    def count(self, *, bridge=None, year=None, basis="source-occurrences", **filters):
        if basis == "source-occurrences":
            return self.query(bridge=bridge, year=year, **filters)
        if basis != "defect-identities":
            raise ValueError("unknown count basis")
        clauses, values = self._where(
            (
                (
                    "bridge",
                    normalize_identity(bridge, "")["bridge"] if bridge else None,
                ),
                ("year", year),
                ("kind", "defect"),
            )
        )
        with self.connect() as db:
            docs = db.execute(
                "SELECT DISTINCT document FROM observations WHERE "
                + " AND ".join(clauses),
                values,
            ).fetchall()
            graphs = [
                json.loads(
                    db.execute(
                        "SELECT graph FROM graphs WHERE document=?", (key,)
                    ).fetchone()[0]
                )
                for (key,) in docs
            ]
        identities, unknown, sources = set(), 0, []
        for graph in graphs:
            for obs in graph["observations"]:
                observation_year = obs["inspection_year"] or (
                    obs["report_year"] if obs["time_basis"] == "report-only" else None
                )
                if obs["kind"] != "defect" or (
                    year is not None
                    and year != observation_year
                ):
                    continue
                raw = obs["raw"]
                if (
                    filters.get("component") is not None
                    and raw.get("病害部位") != filters["component"]
                ):
                    continue
                if (
                    filters.get("disease") is not None
                    and raw.get("病害类型") != filters["disease"]
                ):
                    continue
                if obs.get("defect_identity_id"):
                    identities.add(obs["defect_identity_id"])
                    sources.extend(obs["evidence_ids"])
                else:
                    unknown += 1
        return {
            "status": "supported"
            if identities and not unknown
            else "insufficient-evidence",
            "count": len(identities),
            "unknown_identity_occurrences": unknown,
            "count_basis": basis,
            "sources": sources,
            "interpretation": "confirmed explicit identifiers only; incomplete identity count is a lower bound",
        }

    def compare_measurements(self, bridge, years, *, attribute, **filters):
        if len(set(years)) < 2:
            raise ValueError("at least two distinct years required")
        years = sorted(set(years))
        answers = {
            str(y): self.measurements(
                bridge=bridge, year=y, attribute=attribute, **filters
            )
            for y in years
        }
        groups = {}
        for year, answer in answers.items():
            for item in answer["measurements"]:
                if (
                    item.get("defect_identity_id")
                    and item["time_basis"] == "inspection"
                    and item["value"] is not None
                ):
                    key = (
                        item["defect_identity_id"],
                        item["attribute"],
                        item["unit"],
                        item["statistic"],
                    )
                    groups.setdefault(key, {}).setdefault(year, []).append(item)
        changes = []
        for group in groups.values():
            # Multiple same-year observations can be inconsistent. Do not select
            # a maximum silently, or compare different statistic definitions.
            if any(len(group.get(str(y), [])) != 1 for y in years):
                continue
            for left_year, right_year in zip(years, years[1:]):
                left, right = group[str(left_year)][0], group[str(right_year)][0]
                relation = assess_relation(
                    left["identity"],
                    right["identity"],
                    left_defect_id=left["reported_defect_id"],
                    right_defect_id=right["reported_defect_id"],
                    evidence_ids=[
                        e["id"] for item in (left, right) for e in item["evidence"]
                    ],
                    evidence_bound=all(
                        e["binding"] == "bound"
                        for item in (left, right)
                        for e in item["evidence"]
                    ),
                )
                if relation["status"] != "confirmed":
                    continue
                changes.append(
                    {
                        "left": left,
                        "right": right,
                        "delta": right["value"] - left["value"],
                        "unit": left["unit"],
                        "statistic": left["statistic"],
                        "relation": relation,
                        "derivation": "right.value - left.value; same unit, attribute, statistic and explicit defect identity",
                    }
                )
        return {
            "status": "supported" if changes else "insufficient-evidence",
            "comparison": answers,
            "changes": changes,
            "reason": None if changes else "same-defect relation not confirmed",
            "interpretation": "measurement changes only; missing observations never prove disappearance",
        }

    def compare(self, bridge, years, **filters):
        if len(set(years)) < 2:
            raise ValueError("at least two years required")
        answers = {
            str(year): self.query(bridge=bridge, year=year, **filters) for year in years
        }
        return {
            "status": "supported"
            if all(a["status"] == "supported" for a in answers.values())
            else "insufficient-evidence",
            "comparison": answers,
            "interpretation": "source occurrence counts; not inferred engineering deterioration",
        }


def validate_index_ledger(record: dict, path: Path) -> None:
    """Read existing trusted receipts. Never create a fresh ledger for import."""
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("existing real timing ledger required")
    try:
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
            successful = [
                c for c in record.get("model_calls", []) if c.get("status") == "success"
            ]
            if not successful:
                raise ValueError("actual successful calls required in ledger")
            for call in successful:
                row = db.execute(
                    "SELECT status,receipt FROM calls WHERE id=?",
                    (call.get("call_id"),),
                ).fetchone()
                receipt = json.loads(row[1]) if row and row[1] else {}
                if (
                    not row
                    or row[0] != "success"
                    or not receipt.get("backend_request_id")
                    or receipt.get("prompt_sha256") != call.get("prompt_sha256")
                    or (
                        call.get("backend_request_id")
                        and call["backend_request_id"] != receipt["backend_request_id"]
                    )
                ):
                    raise ValueError(
                        "call is not bound to actual successful ledger receipt"
                    )
    except (sqlite3.Error, json.JSONDecodeError) as exc:
        raise ValueError("invalid timing ledger: " + str(exc)) from exc
