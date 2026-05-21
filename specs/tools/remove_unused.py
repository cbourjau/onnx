# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Remove unused functions and const variables from C++ source files.

Parses clang build output for -Wunused-function and -Wunused-const-variable
diagnostics, then uses tree-sitter to find and remove the full declarations.
Also scans for unused `static constexpr const char*` doc variables.

Usage:
    pixi run install 2>&1 | python specs/tools/remove_unused.py
    python specs/tools/remove_unused.py build.log
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import tree_sitter_cpp as tscpp
from tree_sitter import Language, Node, Parser

DEFS_DIR = Path(__file__).resolve().parent.parent.parent / "onnx" / "defs"

_DIAG_RE = re.compile(
    r"^\s*(?P<file>[^\s:]+):(?P<line>\d+):(?P<col>\d+): (?:error|warning): "
    r"unused (?P<kind>function|variable) '(?P<name>[^']+)' "
    r"\[-(?:Werror,)?-?Wunused-(?:function|const-variable)\]",
)

# Matches: static constexpr const char* Foo_doc = R"DOC(...)DOC";
_DOC_RDOC_RE = re.compile(
    rb"static (?:constexpr )?const char\s*\*\s*(?:const\s+)?(\w+_doc\w*)\s*=\s*"
    rb'R"DOC\(.*?\)DOC";',
    re.DOTALL,
)

# Matches: static constexpr const char* Foo_doc = "..." "...";  (concatenated strings)
# Also matches simple aliases like: = OtherVar;
_DOC_STR_RE = re.compile(
    rb"static (?:constexpr )?const char\s*\*\s*(?:const\s+)?(\w+_doc\w*)\s*=\s*"
    rb'(?:"(?:[^"\\]|\\.)*"\s*)*"(?:[^"\\]|\\.)*";',
    re.DOTALL,
)

_DOC_ALIAS_RE = re.compile(
    rb"static (?:constexpr )?const char\s*\*\s*(?:const\s+)?(\w+_doc\w*)\s*=\s*"
    rb"(\w+);",
)


def _remove_range(source: bytes, start: int, end: int) -> bytes:
    """Remove bytes[start:end] plus trailing newlines and one preceding blank line."""
    while end < len(source) and source[end : end + 1] in (b"\n", b"\r"):
        end += 1
    if start > 1 and source[start - 2 : start] == b"\n\n":
        start -= 1
    return source[:start] + source[end:]


def _remove_unused_doc_vars(filepath: Path) -> int:
    """Remove unused *_doc variables from a file. Returns count removed."""
    count = 0
    while True:
        source = filepath.read_bytes()
        source_text = source.decode()
        found = False
        for pattern in (_DOC_RDOC_RE, _DOC_STR_RE, _DOC_ALIAS_RE):
            for m in pattern.finditer(source):
                var_name = m.group(1).decode()
                if (
                    len(re.findall(r"\b" + re.escape(var_name) + r"\b", source_text))
                    <= 1
                ):
                    filepath.write_bytes(_remove_range(source, m.start(), m.end()))
                    count += 1
                    found = True
                    break
            if found:
                break
        if not found:
            break
    return count


def _remove_unused_static_functions(filepath: Path) -> int:
    """Remove unused static functions from a file. Returns count removed."""
    language = Language(tscpp.language())
    parser = Parser(language)
    count = 0
    while True:
        source = filepath.read_bytes()
        source_text = source.decode()
        tree = parser.parse(source)
        found = False

        for node in tree.root_node.children:
            # Also look inside namespaces
            nodes_to_check = [node]
            if node.type == "namespace_definition":
                body = node.child_by_field_name("body")
                if body:
                    nodes_to_check = list(body.children)

            for n in nodes_to_check:
                if n.type != "function_definition":
                    continue
                text = source[n.start_byte : n.end_byte]
                if not text.startswith(b"static "):
                    continue
                declarator = n.child_by_field_name("declarator")
                if declarator is None:
                    continue
                # Extract function name from declarator
                func_name_node = declarator.child_by_field_name("declarator")
                if func_name_node is None:
                    func_name_node = declarator
                name = source[
                    func_name_node.start_byte : func_name_node.end_byte
                ].decode()
                # Only consider simple identifiers
                if not name.isidentifier():
                    continue
                if len(re.findall(r"\b" + re.escape(name) + r"\b", source_text)) <= 1:
                    filepath.write_bytes(
                        _remove_range(source, n.start_byte, n.end_byte)
                    )
                    count += 1
                    found = True
                    break
            if found:
                break
        if not found:
            break
    return count


def _find_declaration_by_name(root: Node, name: str, source: bytes) -> Node | None:
    """Find a top-level declaration (function or variable) by name."""

    def search(nodes: list[Node]) -> Node | None:
        for node in nodes:
            if node.type in ("function_definition", "declaration"):
                parent = node.parent
                if parent is None or parent.type not in (
                    "translation_unit",
                    "declaration_list",
                ):
                    continue
                declarator = node.child_by_field_name("declarator")
                if (
                    declarator is not None
                    and name.encode()
                    in source[declarator.start_byte : declarator.end_byte]
                ):
                    return node
            elif node.type == "namespace_definition":
                body = node.child_by_field_name("body")
                if body:
                    result = search(list(body.children))
                    if result:
                        return result
        return None

    return search(list(root.children))


def _remove_symbols(diagnostics: list[dict]) -> None:
    """Remove unused symbols found via compiler diagnostics."""
    language = Language(tscpp.language())
    parser = Parser(language)

    by_file: dict[Path, list[dict]] = {}
    for diag in diagnostics:
        by_file.setdefault(diag["file"], []).append(diag)

    for filepath, diags in sorted(by_file.items()):
        for diag in diags:
            source = filepath.read_bytes()
            tree = parser.parse(source)
            node = _find_declaration_by_name(tree.root_node, diag["name"], source)
            if node is None:
                print(f"  WARNING: Could not find '{diag['name']}' in {filepath}")
                continue
            filepath.write_bytes(_remove_range(source, node.start_byte, node.end_byte))


def main() -> None:
    if len(sys.argv) > 1:
        lines = Path(sys.argv[1]).read_text().splitlines()
    else:
        lines = sys.stdin.read().splitlines()

    diagnostics = [
        {"file": Path(m.group("file")), "name": m.group("name")}
        for line in lines
        if (m := _DIAG_RE.match(line))
    ]
    if diagnostics:
        _remove_symbols(diagnostics)

    cc_files = sorted(DEFS_DIR.glob("**/*.cc"))
    total = sum(_remove_unused_doc_vars(f) for f in cc_files)
    total += sum(_remove_unused_static_functions(f) for f in cc_files)

    if not diagnostics and not total:
        print("No unused symbols found.")


if __name__ == "__main__":
    main()
