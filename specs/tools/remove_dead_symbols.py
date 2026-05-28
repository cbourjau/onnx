# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Remove unused static symbols from C++ files after refactoring.

Usage:
    python specs/tools/remove_dead_symbols.py

Iteratively builds the project, parses -Werror,-Wunused errors,
and removes the offending declarations until the build succeeds.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

_REMOVAL_PATTERNS = [
    # static constexpr const char* NAME = R"DOC(...)DOC";
    r'^static constexpr const char\* {sym}\s*=\s*R"DOC\(.*?\)DOC";\n*',
    # static const char* const NAME = R"DOC(...)DOC";
    r'^static const char\* const {sym}\s*=\s*R"DOC\(.*?\)DOC";\n*',
    # static constexpr const char* NAME = "..."; (single line)
    r'^static constexpr const char\* {sym}\s*=\s*"[^"]*";\n*',
    # static constexpr const char* NAME =\n    R"DOC(...)DOC";
    r'^static constexpr const char\* {sym}\s*=\n\s*R"DOC\(.*?\)DOC";\n*',
    # static const char* const NAME = other_var;
    r"^static const char\* const {sym}\s*=\s*\w+;\n*",
    # constexpr const char* NAME = R"DOC(...)DOC"; (no static)
    r'^constexpr const char\* {sym}\s*=\s*R"DOC\(.*?\)DOC";\n*',
    # constexpr const char* NAME = "..."; (no static)
    r'^constexpr const char\* {sym}\s*=\s*"[^"]*";\n*',
    # Multi-line C string: static constexpr const char* NAME =\n    "..."\n    "...";
    r'^static constexpr const char\* {sym}\s*=\s*\n(\s*".*?"\n)*\s*".*?";\n*',
    # static std::vector<std::string> NAME() { ... }
    r"^static std::vector<std::string> {sym}\(\)[\s]*\{{.*?\n\}}\n*",
]


def _remove_symbol(content: str, sym: str) -> str:
    """Try to remove a symbol declaration from content."""
    for pattern_template in _REMOVAL_PATTERNS:
        pattern = pattern_template.format(sym=re.escape(sym))
        new_content = re.sub(
            pattern, "", content, count=1, flags=re.DOTALL | re.MULTILINE
        )
        if new_content != content:
            return new_content
    return content


def _parse_unused_errors(output: str) -> list[tuple[str, str]]:
    """Parse build output for unused variable/function errors."""
    errors = []
    for line in output.splitlines():
        m = re.match(r"\s+(/\S+\.cc):(\d+):\d+: error: unused \w+ '(\w+)'", line)
        if m:
            errors.append((m.group(1), m.group(3)))
    return errors


def main() -> None:
    max_iterations = 20

    for iteration in range(1, max_iterations + 1):
        gen_dir = Path(".setuptools-cmake-build/onnx/defs/generated")
        if gen_dir.exists():
            shutil.rmtree(gen_dir)

        result = subprocess.run(
            ["pixi", "run", "install"],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        output = result.stdout + result.stderr

        if "Successfully" in output:
            print("Build succeeded!")
            return

        errors = _parse_unused_errors(output)
        if not errors:
            print("Build failed for non-unused reason:")
            for line in output.splitlines():
                if "error:" in line and "unused" not in line and "too many" not in line:
                    print(f"  {line.strip()}")
            return

        print(f"  Iteration {iteration}: removing {len(errors)} unused symbols...")

        by_file: dict[str, set[str]] = {}
        for filepath, sym in errors:
            by_file.setdefault(filepath, set()).add(sym)

        for filepath, symbols in by_file.items():
            p = Path(filepath)
            content = p.read_text()
            for sym in symbols:
                content = _remove_symbol(content, sym)
            p.write_text(content)

    print(f"Failed to converge after {max_iterations} iterations")


if __name__ == "__main__":
    main()
