# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Generate C++ schema registration helpers from YAML+frontmatter specs.

Usage:
    python specs/tools/gen_cpp.py --output-dir <path>

Generates FillSpec functions that populate an OpSchema with spec data
(doc, inputs, outputs, attributes, type constraints). Inference functions
remain hand-written and are attached separately via .FillUsing().
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jinja2
from models import (
    Attribute,
    BaseSpec,
    TypeConstraint,
    Variable,
    Variadic,
    collect_ops,
    resolve_versions,
)

TEMPLATE_DIR = Path(__file__).resolve().parent
SPECS_DIR = TEMPLATE_DIR.parent / "opset"

_ATTR_TYPE_MAP = {
    "float": "AttributeProto::FLOAT",
    "floats": "AttributeProto::FLOATS",
    "graph": "AttributeProto::GRAPH",
    "int": "AttributeProto::INT",
    "ints": "AttributeProto::INTS",
    "sparse_tensor": "AttributeProto::SPARSE_TENSOR",
    "string": "AttributeProto::STRING",
    "strings": "AttributeProto::STRINGS",
    "tensor": "AttributeProto::TENSOR",
    "type_proto": "AttributeProto::TYPE_PROTO",
}

_KIND_ENUM_MAP = {
    "Single": "OpSchema::Single",
    "Option": "OpSchema::Optional",
}

_DIFF_ENUM_MAP = {
    None: "OpSchema::Unknown",
    True: "OpSchema::Differentiable",
    False: "OpSchema::NonDifferentiable",
}


def _escape_cpp_str(s: str) -> str:
    """Escape a string for use inside a C++ quoted string literal."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class CppVariable:
    """Wraps a Variable model and exposes C++ rendering properties."""

    def __init__(self, var: Variable) -> None:
        self._var = var

    @property
    def name(self) -> str:
        return self._var.name

    @property
    def doc(self) -> str:
        return _escape_cpp_str(self._var.doc)

    @property
    def type(self) -> str:
        return self._var.type

    @property
    def kind_enum(self) -> str:
        if isinstance(self._var.kind, Variadic):
            return "OpSchema::Variadic"
        if self._var.kind not in _KIND_ENUM_MAP:
            raise ValueError(f"Unknown formal parameter kind: {self._var.kind!r}")
        return _KIND_ENUM_MAP[self._var.kind]

    @property
    def homogeneous(self) -> str:
        if isinstance(self._var.kind, Variadic):
            return "true" if self._var.kind.homogeneous else "false"
        return "true"

    @property
    def min_arity(self) -> str:
        if isinstance(self._var.kind, Variadic):
            return str(self._var.kind.min_arity)
        return "1"

    @property
    def diff_enum(self) -> str:
        if self._var.differentiable not in _DIFF_ENUM_MAP:
            raise ValueError(
                f"Unknown differentiable value: {self._var.differentiable!r}"
            )
        return _DIFF_ENUM_MAP[self._var.differentiable]


class CppAttribute:
    """Wraps an Attribute model and exposes C++ rendering properties."""

    def __init__(self, attr: Attribute) -> None:
        self._attr = attr

    @property
    def name(self) -> str:
        return self._attr.name

    @property
    def doc(self) -> str:
        return _escape_cpp_str(self._attr.doc)

    @property
    def type_enum(self) -> str:
        if self._attr.type not in _ATTR_TYPE_MAP:
            raise ValueError(f"Unknown attribute type: {self._attr.type!r}")
        return _ATTR_TYPE_MAP[self._attr.type]

    @property
    def default(self) -> str:  # noqa: PLR0911
        if self._attr.required:
            return "true"
        if self._attr.default is None:
            return "false"

        match self._attr.type:
            case "int":
                return f"static_cast<int64_t>({self._attr.default})"
            case "float":
                return f"static_cast<float>({self._attr.default})"
            case "string":
                return f'std::string("{self._attr.default}")'
            case "ints":
                vals = ", ".join(
                    f"static_cast<int64_t>({v})" for v in self._attr.default
                )
                return f"std::vector<int64_t>{{{vals}}}"
            case "floats":
                vals = ", ".join(f"static_cast<float>({v})" for v in self._attr.default)
                return f"std::vector<float>{{{vals}}}"
            case "strings":
                vals = ", ".join(f'std::string("{v}")' for v in self._attr.default)
                return f"std::vector<std::string>{{{vals}}}"
            case _:
                raise ValueError(
                    f"Cannot render default for attribute type {self._attr.type!r}"
                )


class CppTypeConstraint:
    """Wraps a TypeConstraint model and exposes C++ rendering properties."""

    def __init__(self, tc: TypeConstraint) -> None:
        self._tc = tc

    @property
    def name(self) -> str:
        return self._tc.name

    @property
    def doc(self) -> str:
        return _escape_cpp_str(self._tc.doc)

    @property
    def allowed(self) -> str:
        return "{" + ", ".join(f'"{s}"' for s in self._tc.allowed) + "}"


class CppSpec:
    """Wraps a BaseSpec model and exposes C++ rendering properties."""

    def __init__(self, spec: BaseSpec) -> None:
        self._spec = spec

    @property
    def func_name(self) -> str:
        return f"{self._spec.op}_v{self._spec.since_version}_FillSpec"

    @property
    def support_level_enum(self) -> str | None:
        if self._spec.support_level == "experimental":
            return "OpSchema::SupportType::EXPERIMENTAL"
        return None

    @property
    def deprecated(self) -> bool:
        return self._spec.deprecated

    @property
    def non_deterministic(self) -> bool:
        return not self._spec.deterministic

    @property
    def op(self) -> str:
        return self._spec.op

    @property
    def since_version(self) -> int:
        return self._spec.since_version

    @property
    def doc(self) -> str:
        return self._spec.doc

    @property
    def inputs(self) -> list[CppVariable]:
        return [CppVariable(v) for v in self._spec.inputs]

    @property
    def outputs(self) -> list[CppVariable]:
        return [CppVariable(v) for v in self._spec.outputs]

    @property
    def attributes(self) -> list[CppAttribute]:
        return [CppAttribute(a) for a in self._spec.attributes]

    @property
    def type_constraints(self) -> list[CppTypeConstraint]:
        return [CppTypeConstraint(tc) for tc in self._spec.type_constraints]


def _load_env() -> jinja2.Environment:
    return jinja2.Environment(  # noqa: S701
        loader=jinja2.FileSystemLoader(TEMPLATE_DIR),
        keep_trailing_newline=True,
        lstrip_blocks=True,
        trim_blocks=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate C++ schema helpers from ONNX specs."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for generated files",
    )
    args = parser.parse_args()

    if not SPECS_DIR.exists():
        # No specs generated yet — write empty stubs
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "op_specs_generated.h").write_text(
            "#pragma once\n// No specs generated yet.\n"
        )
        (args.output_dir / "op_specs_generated.cc").write_text(
            "// No specs generated yet.\n"
        )
        return

    ops = collect_ops(SPECS_DIR)
    if not ops:
        raise RuntimeError(
            f"No spec files found in {SPECS_DIR}. Run gen_specs.py first."
        )

    all_specs: list[CppSpec] = [
        CppSpec(res.base)
        for version_chain in ops.values()
        for res in resolve_versions([spec for _, spec in version_chain])
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = _load_env()

    header_path = args.output_dir / "op_specs_generated.h"
    header_path.write_text(
        env.get_template("op_specs_generated.h.jinja").render(specs=all_specs)
    )

    source_path = args.output_dir / "op_specs_generated.cc"
    source_path.write_text(
        env.get_template("op_specs_generated.cc.jinja").render(specs=all_specs)
    )


if __name__ == "__main__":
    main()
