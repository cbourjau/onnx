# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Refactor C++ defs files to use FillUsing with generated FillSpec functions.

Usage:
    python specs/tools/refactor_defs.py --list <file.cc> [file2.cc ...]
    python specs/tools/refactor_defs.py --apply <file.cc> [file2.cc ...]

--list:  Print migratable ops (op_name, version, domain) without modifying files.
--apply: Perform the refactoring.

Uses tree-sitter to parse the C++ AST, identifies the method chain in each
ONNX_OPERATOR_SET_SCHEMA block, and replaces spec methods with .FillUsing().
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tree_sitter_cpp as tscpp
from tree_sitter import Language, Node, Parser

# Spec methods — these get replaced by FillUsing
_SPEC_METHODS = frozenset({"SetDoc", "Input", "Output", "Attr", "TypeConstraint"})

_CPP_PARSER = Parser(Language(tscpp.language()))


def _find_schema_calls(root: Node) -> list[Node]:
    """Find all ONNX_OPERATOR_SET_SCHEMA(...) call expressions."""
    results = []
    for node in _walk(root):
        if (
            node.type == "call_expression"
            and node.children
            and node.children[0].type == "identifier"
            and node.children[0].text == b"ONNX_OPERATOR_SET_SCHEMA"
        ):
            results.append(node)  # noqa: PERF401
    return results


def _walk(node: Node):
    """Depth-first walk of the tree."""
    yield node
    for child in node.children:
        yield from _walk(child)


def _get_macro_args(call_node: Node) -> tuple[str, str, Node] | None:
    """Extract (op_name, version, schema_chain_node) from macro call."""
    arg_list = call_node.child_by_field_name("arguments")
    if arg_list is None:
        return None
    args = [c for c in arg_list.children if c.type not in ("(", ")", ",")]
    if len(args) != 3:
        return None
    op_name = args[0].text.decode().strip()
    version = args[1].text.decode().strip()
    return op_name, version, args[2]


def _unwind_chain(node: Node) -> list[tuple[str, Node]]:
    """Unwind a fluent method chain into [(method_name, call_node), ...].

    The chain is nested: outermost call is the last method in the chain.
    Returns in source order (first method first).
    """
    chain: list[tuple[str, Node]] = []
    current = node
    while current.type == "call_expression":
        field_expr = current.child_by_field_name("function")
        if field_expr is None or field_expr.type != "field_expression":
            break
        method_node = field_expr.child_by_field_name("field")
        if method_node is None:
            break
        method_name = method_node.text.decode()
        chain.append((method_name, current))
        current = field_expr.child_by_field_name("argument")
        if current is None:
            break
    chain.reverse()
    return chain


def _is_migratable(chain: list[tuple[str, Node]]) -> bool:
    """Determine if a chain can be migrated.

    A chain is migratable if:
    - It has at least one spec method
    - It does NOT already use FillUsing
    - All spec methods come BEFORE all non-spec methods (no interleaving)
    """
    if not chain:
        return False
    has_spec = False
    past_boundary = False
    for name, _ in chain:
        if name == "FillUsing":
            return False
        if name in _SPEC_METHODS:
            if past_boundary:
                return False  # Spec method after non-spec — can't cleanly split
            has_spec = True
        else:
            past_boundary = True
    return has_spec


def _find_boundary(chain: list[tuple[str, Node]]) -> int | None:
    """Find the index of the first non-spec method in the chain.

    Returns None if all methods are spec methods (no non-spec calls).
    """
    for i, (name, _) in enumerate(chain):
        if name not in _SPEC_METHODS:
            return i
    return None


def find_migratable_ops(filepath: Path) -> list[tuple[str, str]]:
    """Return list of (op_name, version) that can be migrated in this file."""
    source = filepath.read_bytes()
    tree = _CPP_PARSER.parse(source)
    schema_calls = _find_schema_calls(tree.root_node)

    ops = []
    for call_node in schema_calls:
        result = _get_macro_args(call_node)
        if result is None:
            continue
        op_name, version, chain_node = result
        chain = _unwind_chain(chain_node)
        if _is_migratable(chain):
            ops.append((op_name, version))
    return ops


def refactor_file(filepath: Path) -> int:
    """Refactor a single .cc file. Returns count of refactored blocks."""
    source = filepath.read_bytes()
    tree = _CPP_PARSER.parse(source)

    schema_calls = _find_schema_calls(tree.root_node)
    if not schema_calls:
        return 0

    replacements: list[tuple[int, int, bytes]] = []

    for call_node in schema_calls:
        result = _get_macro_args(call_node)
        if result is None:
            continue
        op_name, version, chain_node = result

        chain = _unwind_chain(chain_node)
        if not _is_migratable(chain):
            continue

        boundary = _find_boundary(chain)
        fill_using = f".FillUsing({op_name}_v{version}_FillSpec)"

        if boundary is None:
            new_text = f"OpSchema(){fill_using}".encode()
            replacements.append((chain_node.start_byte, chain_node.end_byte, new_text))
        else:
            boundary_call = chain[boundary][1]
            boundary_field_expr = boundary_call.child_by_field_name("function")
            inner_node = boundary_field_expr.child_by_field_name("argument")
            new_text = f"OpSchema(){fill_using}".encode()
            replacements.append((inner_node.start_byte, inner_node.end_byte, new_text))

    if not replacements:
        return 0

    # Apply replacements in reverse byte order
    replacements.sort(key=lambda r: r[0], reverse=True)
    buf = bytearray(source)
    for start, end, new_text in replacements:
        buf[start:end] = new_text
    source = bytes(buf)

    # Add include if not present
    include_line = b'#include "onnx/defs/generated/op_specs_generated.h"\n'
    if include_line not in source:
        last_include_end = 0
        for line in source.split(b"\n"):
            if line.startswith(b"#include"):
                last_include_end = source.index(line) + len(line) + 1
        if last_include_end > 0:
            source = (
                source[:last_include_end] + include_line + source[last_include_end:]
            )

    filepath.write_bytes(source)
    return len(replacements)


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="List migratable ops")
    group.add_argument("--apply", action="store_true", help="Apply refactoring")
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()

    if args.list:
        for filepath in args.files:
            if not filepath.exists():
                continue
            ops = find_migratable_ops(filepath)
            for op_name, version in ops:
                print(f"{op_name} {version}")
    else:
        for filepath in args.files:
            if not filepath.exists():
                continue
            count = refactor_file(filepath)
            if count:
                print(f"  Refactored {count} blocks in {filepath}")


if __name__ == "__main__":
    main()
