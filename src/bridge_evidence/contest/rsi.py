"""Content-bound inference strategies; never execute proposed Python in this process."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path

from .candidate import DEFAULT, validate_config
from .run_state import json_digest

ABI = "cityagent-pure-strategy-v1"
SIGNATURES = {
    "choose_action": ("state",),
    "propose_candidates": ("history", "search_state"),
    "update_search": ("search_state", "outcomes"),
}
SOURCE = """def choose_action(state):
    return base_action(state)

def propose_candidates(history, search_state):
    return default_candidates(history, search_state)

def update_search(search_state, outcomes):
    return default_search(search_state, outcomes)
"""
SAFE_CALLS = {
    "base_action",
    "default_candidates",
    "default_search",
    "len",
    "min",
    "max",
    "abs",
    "sorted",
    "range",
    "enumerate",
    "get",
    "copy_dict",
}
ALLOWED_AST = {
    ast.Module,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Return,
    ast.Assign,
    ast.AugAssign,
    ast.If,
    ast.For,
    ast.Break,
    ast.Continue,
    ast.Pass,
    ast.Expr,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.Dict,
    ast.List,
    ast.Tuple,
    ast.Subscript,
    ast.Slice,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.Call,
    ast.keyword,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.Not,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
}


def validate_source(source):
    if not isinstance(source, str) or len(source.encode()) > 65536:
        raise ValueError("strategy source size")
    tree = ast.parse(source)
    if len(tree.body) != 3 or {getattr(n, "name", None) for n in tree.body} != set(
        SIGNATURES
    ):
        raise ValueError("only the three strategy functions are allowed")
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef) or function.decorator_list:
            raise ValueError("invalid strategy function")
        args = function.args
        if (
            tuple(a.arg for a in args.args) != SIGNATURES[function.name]
            or args.posonlyargs
            or args.kwonlyargs
            or args.vararg
            or args.kwarg
            or args.defaults
            or function.returns
            or any(a.annotation for a in args.args)
        ):
            raise ValueError("strategy ABI changed")
    nodes = list(ast.walk(tree))
    if sum(isinstance(n, ast.FunctionDef) for n in nodes) != 3:
        raise ValueError("nested functions forbidden")
    if len(nodes) > 4096:
        raise ValueError("strategy AST size")
    for node in nodes:
        if type(node) not in ALLOWED_AST:
            raise ValueError("forbidden Python syntax: " + type(node).__name__)
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise ValueError("private names forbidden")
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name)
            or node.func.id not in SAFE_CALLS
            or any(k.arg is None for k in node.keywords)
        ):
            raise ValueError("forbidden strategy call")
        if isinstance(node, ast.Constant) and isinstance(node.value, (bytes, complex)):
            raise ValueError("non JSON constant")
    return tree


def base_action(state):
    if state.get("remaining_calls", 1) <= 0 or state.get("remaining_seconds", 1) <= 0:
        return {"action": "fail"}
    phase = state.get("phase")
    if phase == "schedule":
        return {
            "action": "extract",
            "batch_size": max(
                b
                for b in (1, 2, 4)
                if b <= state.get("batch_size", 1)
                and b <= state.get("remaining_calls", 1)
            ),
        }
    if phase == "merge":
        return {
            "action": "repair"
            if state.get("allow_repair")
            and state.get("react_calls", 0) == 0
            and state.get("remaining_calls", 0) >= 2
            else "fail"
        }
    if state.get("error_kind") == "length" and state.get("depth", 0) < state.get(
        "max_depth", 8
    ):
        return {"action": "split"}
    if state.get("error_kind") in ("format", "temporary") and state.get(
        "attempt", 0
    ) < state.get("max_attempts", 2):
        return {"action": "retry"}
    return {"action": "fail"}


def validate_action(action, state):
    if not isinstance(action, dict):
        raise ValueError("action must be JSON object")
    expected = (
        {"action", "batch_size"} if action.get("action") == "extract" else {"action"}
    )
    if set(action) != expected:
        raise ValueError("unexpected action arguments")
    kind = action.get("action")
    if kind == "extract":
        if (
            state.get("phase") != "schedule"
            or type(action["batch_size"]) is not int
            or action["batch_size"] not in (1, 2, 4)
            or action["batch_size"] > state.get("batch_size", 1)
            or action["batch_size"] > state.get("remaining_calls", 0)
        ):
            raise ValueError("action exceeds extraction capacity")
    elif kind == "split":
        if state.get("phase") != "failure" or state.get("depth", 0) >= state.get(
            "max_depth", 8
        ):
            raise ValueError("illegal split")
    elif kind == "retry":
        if (
            state.get("phase") != "failure"
            or state.get("attempt", 0) >= state.get("max_attempts", 2)
            or state.get("error_kind") not in ("temporary", "format")
        ):
            raise ValueError("illegal retry")
    elif kind == "repair":
        if (
            state.get("phase") != "merge"
            or not state.get("allow_repair")
            or state.get("react_calls", 0) > 0
            or state.get("remaining_calls", 0) < 2
        ):
            raise ValueError("illegal repair")
    elif kind != "fail":
        raise ValueError("unknown action")
    if kind != "fail" and (
        state.get("remaining_calls", 1) <= 0 or state.get("remaining_seconds", 1) <= 0
    ):
        raise ValueError("action exceeds budget")
    return action


@dataclass(frozen=True)
class StrategyBundle:
    source: str = SOURCE
    config: dict = field(default_factory=lambda: dict(DEFAULT))
    parent_sha256: str | None = None
    abi: str = ABI

    def __post_init__(self):
        validate_source(self.source)
        if self.abi != ABI:
            raise ValueError("unknown strategy ABI")
        object.__setattr__(
            self,
            "config",
            validate_config(json.loads(json.dumps(self.config, allow_nan=False))),
        )
        if self.parent_sha256 is not None and (
            len(self.parent_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.parent_sha256)
        ):
            raise ValueError("invalid parent identity")

    @classmethod
    def baseline(cls, config=None):
        return cls(config=validate_config(config or {}))

    def as_dict(self):
        return {
            "schema_version": 1,
            "abi": self.abi,
            "source": self.source,
            "config": self.config,
            "parent_sha256": self.parent_sha256,
        }

    @property
    def digest(self):
        return json_digest(self.as_dict())

    @classmethod
    def load(cls, path):
        value = json.loads(Path(path).read_text())
        if (
            set(value) != {"schema_version", "abi", "source", "config", "parent_sha256"}
            or value["schema_version"] != 1
        ):
            raise ValueError("invalid strategy bundle")
        return cls(
            value["source"], value["config"], value["parent_sha256"], value["abi"]
        )


def apply_patch(parent, patch):
    if (
        not isinstance(patch, dict)
        or set(patch) != {"parent_sha256", "target_function", "new_body"}
        or patch["parent_sha256"] != parent.digest
        or patch["target_function"] not in SIGNATURES
        or not isinstance(patch["new_body"], str)
        or len(patch["new_body"].encode()) > 65536
    ):
        raise ValueError("patch must bind parent and one allowed function")
    tree = validate_source(parent.source)
    replacement = ast.parse(
        "def replacement():\n"
        + "\n".join("    " + line for line in patch["new_body"].splitlines())
    )
    for function in tree.body:
        if function.name == patch["target_function"]:
            function.body = replacement.body[0].body
    source = ast.unparse(tree) + "\n"
    return StrategyBundle(source, dict(parent.config), parent.digest)


def default_candidates(history, search_state):
    config = dict(search_state.get("config", DEFAULT))
    candidates = []
    for key, value in (
        ("batch_size", 2),
        ("batch_size", 4),
        ("input_budget", 4096),
        ("output_budget", 6144),
        ("prompt", "source-fragment-concise-v1"),
    ):
        candidate = dict(config)
        candidate[key] = value
        candidates.append(candidate)
    return candidates


def default_search(search_state, outcomes):
    state = dict(search_state)
    state["generation"] = min(state.get("generation", 0) + 1, 3)
    successes = [o for o in outcomes if o.get("status") == "complete"]
    if successes:
        best = min(successes, key=lambda o: o.get("wall_seconds", float("inf")))
        state["config"] = best.get("config", state.get("config", DEFAULT))
    return state
