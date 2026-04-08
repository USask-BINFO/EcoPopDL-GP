#!/usr/bin/env python3
"""Render a configured copy of a Python script by replacing selected assignments.

Supports:
  - top-level assignments
  - assignments inside a named function
  - inserting assignments after a marker line when a variable is missing

Examples
--------
python render_configured_script.py \
  --source Scripts/Model/Diploid/integrated_training_chromomap.py \
  --dest tmp/train.py \
  --top METADATA_FILE=/path/meta.csv \
  --top TARGET_COL=DTF \
  --top N_MONTHS=32

python render_configured_script.py \
  --source Scripts/Chromomap_tensor_generation/Diploid/integrated_tile_generation.py \
  --dest tmp/tensors.py \
  --insert-after "# Global variables" 'chr_info={"1":12345,"2":67890}' \
  --top USE_SUBGENOMES=false \
  --func main vcf_path=/path/final.vcf.gz \
  --func main ped_file=/path/final.ped \
  --func main map_file=/path/final.map
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


AssignMap = Dict[str, object]


def parse_value(raw: str):
    for parser in (ast.literal_eval, json.loads):
        try:
            return parser(raw)
        except Exception:
            pass
    low = raw.strip().lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low == "none":
        return None
    return raw


def render_literal(value) -> str:
    return repr(value)


def parse_kv(items: Iterable[str]) -> AssignMap:
    out: AssignMap = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected NAME=VALUE, got: {item}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty key in override: {item}")
        out[key] = parse_value(raw)
    return out


def parse_func_overrides(items: Iterable[List[str]]) -> Dict[str, AssignMap]:
    out: Dict[str, AssignMap] = {}
    for func_name, kv in items:
        out.setdefault(func_name, {}).update(parse_kv([kv]))
    return out


def make_assignment_line(name: str, value, indent: str = "") -> str:
    return f"{indent}{name} = {render_literal(value)}\n"


def collect_replacements_for_scope(
    nodes: Iterable[ast.stmt],
    overrides: AssignMap,
    source_lines: List[str],
    *,
    scope_label: str,
) -> Tuple[List[Tuple[int, int, str]], set[str]]:
    replacements: List[Tuple[int, int, str]] = []
    found: set[str] = set()

    for node in nodes:
        target_name = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_name = node.targets[0].id
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_name = node.target.id
        if target_name is None or target_name not in overrides:
            continue
        start = node.lineno - 1
        end = node.end_lineno - 1
        indent = source_lines[start][: len(source_lines[start]) - len(source_lines[start].lstrip())]
        replacements.append((start, end, make_assignment_line(target_name, overrides[target_name], indent)))
        found.add(target_name)

    missing = sorted(set(overrides) - found)
    if missing:
        raise KeyError(f"Missing assignment(s) in {scope_label}: {', '.join(missing)}")
    return replacements, found


def apply_replacements(source_lines: List[str], replacements: List[Tuple[int, int, str]]) -> List[str]:
    lines = list(source_lines)
    for start, end, new_text in sorted(replacements, key=lambda x: (x[0], x[1]), reverse=True):
        lines[start : end + 1] = [new_text]
    return lines


def insert_after_markers(lines: List[str], insert_specs: List[Tuple[str, AssignMap]]) -> List[str]:
    out = list(lines)
    for marker, kvs in insert_specs:
        idx = next((i for i, line in enumerate(out) if marker in line), None)
        if idx is None:
            raise KeyError(f"Marker not found for insertion: {marker!r}")
        rendered = [make_assignment_line(name, value) for name, value in kvs.items()]
        out[idx + 1 : idx + 1] = rendered
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Path to source Python script.")
    parser.add_argument("--dest", required=True, help="Path to write configured copy.")
    parser.add_argument(
        "--top",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Override a top-level assignment.",
    )
    parser.add_argument(
        "--func",
        action="append",
        nargs=2,
        default=[],
        metavar=("FUNCTION", "NAME=VALUE"),
        help="Override an assignment inside a named function.",
    )
    parser.add_argument(
        "--insert-after",
        action="append",
        nargs=2,
        default=[],
        metavar=("MARKER", "NAME=VALUE"),
        help="Insert an assignment after the first line containing MARKER.",
    )
    args = parser.parse_args()

    source_path = Path(args.source)
    dest_path = Path(args.dest)
    text = source_path.read_text(encoding="utf-8")
    source_lines = text.splitlines(keepends=True)
    tree = ast.parse(text)

    replacements: List[Tuple[int, int, str]] = []

    top_overrides = parse_kv(args.top)
    if top_overrides:
        repl, _ = collect_replacements_for_scope(tree.body, top_overrides, source_lines, scope_label="top level")
        replacements.extend(repl)

    func_overrides = parse_func_overrides(args.func)
    if func_overrides:
        fn_map = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        for func_name, kvs in func_overrides.items():
            if func_name not in fn_map:
                raise KeyError(f"Function not found: {func_name}")
            repl, _ = collect_replacements_for_scope(
                fn_map[func_name].body,
                kvs,
                source_lines,
                scope_label=f"function {func_name}",
            )
            replacements.extend(repl)

    rendered_lines = apply_replacements(source_lines, replacements)

    insert_specs: List[Tuple[str, AssignMap]] = []
    for marker, kv in args.insert_after:
        insert_specs.append((marker, parse_kv([kv])))
    if insert_specs:
        rendered_lines = insert_after_markers(rendered_lines, insert_specs)

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text("".join(rendered_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
