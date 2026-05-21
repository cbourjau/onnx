# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Remove unused generator functions and their declarations.

Uses tree-sitter to find function definitions and declarations by name,
then removes them (including any leading comment block).

Usage:
    pixi run python specs/tools/remove_unused_generators.py
"""

from __future__ import annotations

import re
from pathlib import Path

import tree_sitter_cpp as tscpp
from tree_sitter import Language, Node, Parser

DEFS_DIR = Path(__file__).resolve().parent.parent.parent / "onnx" / "defs"

# Generators to remove: (function_name, definition_file, declaration_locations)
# declaration_locations is a list of (file, kind) where kind is "decl" or "forward"
GENERATORS_TO_REMOVE = [
    "BinaryLogicDocGenerator",
    "PadDocGenerator",
    "TopKOpGenerator",
    "UnaryFloatMathOpGenerator",
    "SplitToSequenceOpGenerator",
    "ReduceOpGenerator",
    "ReduceOpDynamicAxes",
    "ReduceFunctionOp",
]


def _get_parser() -> Parser:
    language = Language(tscpp.language())
    return Parser(language)


def _find_nodes_by_name(root: Node, name: str, source: bytes) -> list[Node]:
    """Find all top-level function definitions or declarations matching name."""
    results: list[Node] = []

    def search(nodes: list[Node]) -> None:
        for node in nodes:
            if node.type in ("function_definition", "declaration"):
                # Check if name appears in the declarator
                declarator = node.child_by_field_name("declarator")
                if declarator is None:
                    continue
                decl_text = source[declarator.start_byte : declarator.end_byte].decode(
                    errors="replace"
                )
                if re.search(r"\b" + re.escape(name) + r"\b", decl_text):
                    results.append(node)
            elif node.type == "namespace_definition":
                body = node.child_by_field_name("body")
                if body:
                    search(list(body.children))
            elif node.type == "preproc_ifdef":
                search(list(node.children))

    search(list(root.children))
    return results


def _remove_node_with_comment(source: bytes, node: Node) -> bytes:
    """Remove a node and any comment lines directly above it."""
    start = node.start_byte
    end = node.end_byte

    # Walk backwards line-by-line to include preceding comment lines
    lines_before = source[:start].split(b"\n")
    # Skip the partial/empty last line (content after last \n before start)
    if lines_before and lines_before[-1].strip() == b"":
        lines_before.pop()
    while lines_before and lines_before[-1].strip().startswith(b"//"):
        removed_line = lines_before.pop()
        start -= len(removed_line) + 1  # +1 for the newline

    # Include trailing newline(s)
    while end < len(source) and source[end : end + 1] in (b"\n", b"\r"):
        end += 1

    # Include one preceding blank line if present
    if start > 0 and source[start - 1 : start] == b"\n":
        start -= 1

    return source[:start] + source[end:]


def remove_function_from_file(filepath: Path, name: str, parser: Parser) -> bool:
    """Remove a function definition or declaration by name from a file.

    Returns True if something was removed.
    """
    source = filepath.read_bytes()
    if name.encode() not in source:
        return False

    tree = parser.parse(source)
    nodes = _find_nodes_by_name(tree.root_node, name, source)

    if not nodes:
        return False

    # Remove nodes in reverse order (by start_byte) to preserve earlier offsets
    nodes.sort(key=lambda n: n.start_byte, reverse=True)
    for node in nodes:
        source = _remove_node_with_comment(source, node)

    filepath.write_bytes(source)
    return True


def main() -> None:
    parser = _get_parser()

    # Also remove from header files
    all_files = sorted(DEFS_DIR.glob("**/*.cc")) + sorted(DEFS_DIR.glob("**/*.h"))

    total_removed = 0
    for name in GENERATORS_TO_REMOVE:
        for filepath in all_files:
            if remove_function_from_file(filepath, name, parser):
                rel = filepath.relative_to(DEFS_DIR.parent.parent)
                print(f"  Removed '{name}' from {rel}")
                total_removed += 1

    if total_removed:
        print(f"\nRemoved {total_removed} generator definitions/declarations total.")
    else:
        print("No generators to remove.")


if __name__ == "__main__":
    main()
