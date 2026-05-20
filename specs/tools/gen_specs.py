# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Generate spec files with inheritance from the ONNX OpSchema registry.

Usage:
    python specs/tools/gen_specs.py TopK
    python specs/tools/gen_specs.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from models import (
    Attribute,
    BaseSpec,
    InheritedSpec,
    SupportLevel,
    TypeConstraint,
    Variable,
    Variadic,
)

import onnx.defs
import onnx.helper

SPECS_DIR = Path(__file__).resolve().parent.parent / "opset"


def _differentiable(param) -> bool | None:
    """Convert a FormalParameter's differentiation_category to a bool."""
    cat = str(param.differentiation_category)
    if "Differentiable" in cat and "Non" not in cat:
        return True
    if "NonDifferentiable" in cat:
        return False
    return None


def _param_kind(param) -> str | Variadic:
    """Convert a FormalParameter's option to a kind value."""
    opt = str(param.option)
    if "Variadic" in opt:
        return Variadic(homogeneous=param.is_homogeneous, min_arity=param.min_arity)
    if "Optional" in opt:
        return "Option"
    return "Single"


# --- Schema extraction ---


def get_schemas() -> dict[tuple[str, str], list[onnx.defs.OpSchema]]:
    """Get all schemas grouped by (op_name, domain), sorted by version."""
    out: dict[tuple[str, str], list[onnx.defs.OpSchema]] = {}
    for schema in onnx.defs.get_all_schemas_with_history():
        domain = schema.domain or "ai.onnx"
        out.setdefault((schema.name, domain), []).append(schema)
    return {k: sorted(v, key=lambda s: s.since_version) for k, v in out.items()}


# --- Schema to model conversion ---


def _support_level(schema: onnx.defs.OpSchema) -> SupportLevel:
    """Convert an OpSchema's support_level to our enum."""
    level = str(schema.support_level)
    if "EXPERIMENTAL" in level:
        return "experimental"
    return "common"


def _is_deterministic(schema: onnx.defs.OpSchema) -> bool:
    """Return whether the op is deterministic."""
    return not schema.non_deterministic


def schema_to_base_spec(schema: onnx.defs.OpSchema) -> BaseSpec:
    """Convert an OpSchema to a BaseSpec model."""
    domain = schema.domain or "ai.onnx"

    type_constraints = [
        TypeConstraint(
            name=t.type_param_str,
            doc=t.description,
            allowed=sorted(t.allowed_type_strs),
        )
        for t in schema.type_constraints
    ]

    inputs: list[Variable] = []
    for inp in schema.inputs:
        kind = _param_kind(inp)
        inputs.append(
            Variable(
                name=inp.name,
                type=inp.type_str,
                doc=inp.description,
                kind=kind,
                differentiable=_differentiable(inp),
            )
        )

    outputs: list[Variable] = []
    for out in schema.outputs:
        kind = _param_kind(out)
        outputs.append(
            Variable(
                name=out.name,
                type=out.type_str,
                doc=out.description,
                kind=kind,
                differentiable=_differentiable(out),
            )
        )

    attributes: list[Attribute] = []
    for attr in schema.attributes.values():
        attr_type = str(attr.type).split(".")[-1].lower()
        default = None
        if attr.default_value.name:
            default = onnx.helper.get_attribute_value(attr.default_value)
            if isinstance(default, bytes):
                default = default.decode("utf-8")
        attributes.append(
            Attribute(
                name=attr.name,
                type=attr_type,
                doc=attr.description,
                required=attr.required,
                default=default,
            )
        )
    attributes.sort(key=lambda a: (not a.required, a.name))

    return BaseSpec(
        op=schema.name,
        domain=domain,
        since_version=schema.since_version,
        support_level=_support_level(schema),
        deprecated=schema.deprecated,
        deterministic=_is_deterministic(schema),
        doc=(schema.doc or "").strip(),
        type_constraints=type_constraints,
        inputs=inputs,
        outputs=outputs,
        attributes=attributes,
    )


# --- Serialization ---


def _prune_empty(d: dict) -> None:
    """Recursively remove empty dicts/lists and None values from a dict."""
    keys_to_remove: list[str] = []
    for key, value in d.items():
        if isinstance(value, dict):
            _prune_empty(value)
            if not value:
                keys_to_remove.append(key)
        elif isinstance(value, list) and not value:
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del d[key]


def generate_spec_file(spec: BaseSpec | InheritedSpec) -> str:
    """Generate a spec file, placing doc in the markdown body."""
    data = spec.model_dump(exclude_none=True, exclude_defaults=True)
    _prune_empty(data)
    doc = data.pop("doc", "")
    yaml_str = yaml.dump(
        data, default_flow_style=False, sort_keys=False, allow_unicode=True
    )
    if doc:
        return f"---\n{yaml_str}---\n\n{doc}\n"
    return f"---\n{yaml_str}---\n"


# --- File writing ---


def spec_path(domain: str, version: int, op_name: str) -> Path:
    """Build the filesystem path for a spec file."""
    return SPECS_DIR / domain / f"v{version}" / f"{op_name.lower()}.md"


def inherit_ref(op_name: str, version: int) -> str:
    """Build an inherit reference string like 'v10/topk'."""
    return f"v{version}/{op_name.lower()}"


def write_op_versions(versions: list[onnx.defs.OpSchema]) -> None:
    """Write all versions of an op, using inheritance for v2+."""
    base_spec = schema_to_base_spec(versions[0])

    path = spec_path(base_spec.domain, base_spec.since_version, base_spec.op)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generate_spec_file(base_spec))

    prev_spec = base_spec
    for schema in versions[1:]:
        child_spec = schema_to_base_spec(schema)
        ref = inherit_ref(schema.name, prev_spec.since_version)
        diff = prev_spec.diff(child_spec, inherit=ref)

        path = spec_path(child_spec.domain, child_spec.since_version, child_spec.op)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(generate_spec_file(diff))

        prev_spec = child_spec


# --- CLI ---


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate spec files from ONNX OpSchema."
    )
    parser.add_argument("op_name", nargs="?", help="Operator name (e.g. TopK)")
    parser.add_argument("--domain", default="ai.onnx", help="Domain (default: ai.onnx)")
    parser.add_argument("--all", action="store_true", help="Generate all operators")
    args = parser.parse_args()

    if not args.op_name and not args.all:
        parser.error("Provide an op name or --all")

    schemas = get_schemas()

    if args.all:
        for versions in schemas.values():
            write_op_versions(versions)
    else:
        key = (args.op_name, args.domain)
        if key not in schemas:
            sys.exit(1)
        write_op_versions(schemas[key])


if __name__ == "__main__":
    main()
