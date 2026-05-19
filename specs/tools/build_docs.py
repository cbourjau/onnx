# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Build Sphinx-compatible markdown docs from YAML+frontmatter spec files.

Usage:
    python specs/tools/build_docs.py

Reads spec files from specs/opset/, resolves inheritance, and writes rendered
markdown into the Sphinx source tree at docs/docsgen/source/operators_spec/.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import jinja2
import yaml
from models import (
    BaseSpec,
    InheritedSpec,
    NamedListDiff,
    ResolvedSpec,
    Variable,
    Variadic,
    resolve_versions,
)

SPECS_DIR = Path(__file__).resolve().parent.parent / "opset"
DOCS_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "docs"
    / "docsgen"
    / "source"
    / "_generated"
    / "operators_spec"
)
TEMPLATE_DIR = Path(__file__).resolve().parent


# --- Parsing ---


def parse_spec_file(path: Path) -> BaseSpec | InheritedSpec:
    """Parse a spec file into a validated BaseSpec or InheritedSpec."""
    content = path.read_text()
    match = re.match(r"^---\n(.+?)\n---\n?(.*)", content, re.DOTALL)
    if not match:
        raise ValueError(f"No front matter in {path}")
    meta = yaml.safe_load(match.group(1))
    body = match.group(2).strip()

    if "inherit" in meta:
        meta["doc"] = body or None
        return InheritedSpec.model_validate(meta)
    meta["doc"] = body
    return BaseSpec.model_validate(meta)


def collect_ops(
    specs_dir: Path,
) -> dict[str, list[tuple[Path, BaseSpec | InheritedSpec]]]:
    """Collect all ops grouped by (domain/op_name), each as a sorted version chain."""
    ops: dict[str, list[tuple[Path, BaseSpec | InheritedSpec]]] = defaultdict(list)

    for path in sorted(specs_dir.rglob("*.md")):
        spec = parse_spec_file(path)
        op_key = f"{path.parent.parent.name}/{path.stem}"
        ops[op_key].append((path, spec))

    for chain in ops.values():
        chain.sort(key=lambda x: x[1].since_version)

    return dict(ops)


# --- Rendering ---


def _changelog_for_diff(diff: NamedListDiff, category: str) -> list[str]:
    """Generate changelog lines for a single NamedListDiff."""
    return [
        *[f"Added {category} `{item.name}`." for item in diff.add],
        *[f"Removed {category} `{name}`." for name in diff.remove],
        *[f"Updated {category} `{item.name}`." for item in diff.update],
    ]


def render_changelog(diff: InheritedSpec) -> list[str]:
    """Generate a brief changelog from an InheritedSpec."""
    lines: list[str] = []

    if diff.doc is not None:
        lines.append("Updated description.")

    for field, category in [
        (diff.type_constraints, "type constraint"),
        (diff.inputs, "input"),
        (diff.outputs, "output"),
        (diff.attributes, "attribute"),
    ]:
        if field:
            lines.extend(_changelog_for_diff(field, category))

    return lines


def _param_annotation(param: Variable) -> str:
    """Render the kind annotation for an input/output in markdown."""
    if isinstance(param.kind, Variadic):
        homo = "homogeneous" if param.kind.homogeneous else "heterogeneous"
        return f" (variadic, {homo})"
    if param.kind == "Option":
        return " (optional)"
    return ""


def _load_template() -> jinja2.Template:
    env = jinja2.Environment(  # noqa: S701
        loader=jinja2.FileSystemLoader(TEMPLATE_DIR),
        keep_trailing_newline=True,
        lstrip_blocks=True,
        trim_blocks=True,
    )
    env.filters["backtick"] = lambda s: f"`{s}`"
    env.filters["param_annotation"] = _param_annotation
    return env.get_template("op_page.md.jinja")


def render_op_page(template: jinja2.Template, resolved: ResolvedSpec) -> str:
    """Render a ResolvedSpec into a Sphinx-compatible markdown page."""
    changelog = render_changelog(resolved.diff) if resolved.diff else []
    return template.render(spec=resolved.base, changelog=changelog)


# --- Main ---


def main() -> None:
    ops = collect_ops(SPECS_DIR)

    if not ops:
        raise RuntimeError(
            f"No spec files found in {SPECS_DIR}. Run gen_specs.py first."
        )

    template = _load_template()
    count = 0
    # Track which ops land in each (domain, version) for index generation
    version_ops: dict[tuple[str, int], list[str]] = defaultdict(list)

    for version_chain in ops.values():
        specs = [spec for _, spec in version_chain]
        resolved = resolve_versions(specs)

        for res in resolved:
            domain_dir = res.base.domain.replace(".", "_")
            dest = (
                DOCS_DIR
                / domain_dir
                / f"v{res.base.since_version}"
                / f"{res.base.op.lower()}.md"
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(render_op_page(template, res))
            version_ops[(domain_dir, res.base.since_version)].append(
                res.base.op.lower()
            )
            count += 1

    # Generate per-version index pages
    domains: set[str] = set()
    for (domain_dir, version), op_names in sorted(version_ops.items()):
        domains.add(domain_dir)
        index_path = DOCS_DIR / domain_dir / f"v{version}" / "index.md"
        lines = [
            f"# v{version}",
            "",
            "```{toctree}",
            ":maxdepth: 1",
            "",
        ]
        lines.extend(sorted(op_names))
        lines.append("```")
        lines.append("")
        index_path.write_text("\n".join(lines))

    # Generate per-domain index page
    for domain_dir in sorted(domains):
        domain_versions = sorted(v for (d, v) in version_ops if d == domain_dir)
        index_path = DOCS_DIR / domain_dir / "index.md"
        lines = [
            f"# {domain_dir.replace('_', '.')}",
            "",
            "```{toctree}",
            ":maxdepth: 1",
            "",
        ]
        for v in domain_versions:
            lines.append(f"v{v}/index")
        lines.append("```")
        lines.append("")
        index_path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
