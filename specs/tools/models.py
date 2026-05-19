# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Pydantic models for ONNX operator specs."""

from __future__ import annotations

from typing import Annotated, Generic, Literal, TypeVar

from pydantic import AfterValidator, BaseModel, TypeAdapter

AttributeType = Literal[
    "float",
    "floats",
    "graph",
    "int",
    "ints",
    "sparse_tensor",
    "string",
    "strings",
    "tensor",
    "type_proto",
]

FormalParameterOption = Literal["optional", "variadic"]

DifferentiationCategory = Literal["differentiable", "non_differentiable"]

TensorElementType = Literal[
    "bfloat16",
    "bool",
    "complex128",
    "complex64",
    "double",
    "float",
    "float16",
    "float4e2m1",
    "float8e4m3fn",
    "float8e4m3fnuz",
    "float8e5m2",
    "float8e5m2fnuz",
    "float8e8m0",
    "int16",
    "int2",
    "int32",
    "int4",
    "int64",
    "int8",
    "string",
    "uint16",
    "uint2",
    "uint32",
    "uint4",
    "uint64",
    "uint8",
]

MapKeyType = Literal["int64", "string"]


# --- ONNX type representation ---


class Tensor(BaseModel):
    element: TensorElementType

    def __str__(self) -> str:
        return f"tensor({self.element})"


class Map(BaseModel):
    key: MapKeyType
    value: TensorElementType

    def __str__(self) -> str:
        return f"map({self.key}, {self.value})"


class Seq(BaseModel):
    inner: Tensor | Map

    def __str__(self) -> str:
        return f"seq({self.inner})"


class Optional(BaseModel):
    inner: Tensor | Seq

    def __str__(self) -> str:
        return f"optional({self.inner})"


OnnxType = Tensor | Seq | Optional | Map


def parse_onnx_type(s: str) -> OnnxType:
    """Parse an ONNX type string into a typed object."""
    if s.startswith("tensor("):
        return _parse_tensor(s)
    if s.startswith("seq("):
        return _parse_seq(s)
    if s.startswith("optional("):
        return _parse_optional(s)
    if s.startswith("map("):
        return _parse_map(s)
    raise ValueError(f"Invalid ONNX type string: {s!r}")


_element_type_adapter = TypeAdapter(TensorElementType)
_map_key_adapter = TypeAdapter(MapKeyType)


def _parse_tensor(s: str) -> Tensor:
    elem = _unwrap("tensor", s)
    return Tensor(element=_element_type_adapter.validate_python(elem))


def _parse_map(s: str) -> Map:
    inner = _unwrap("map", s)
    key, value = inner.split(", ", 1)
    return Map(
        key=_map_key_adapter.validate_python(key),
        value=_element_type_adapter.validate_python(value),
    )


def _parse_seq(s: str) -> Seq:
    inner = _unwrap("seq", s)
    if inner.startswith("tensor("):
        return Seq(inner=_parse_tensor(inner))
    if inner.startswith("map("):
        return Seq(inner=_parse_map(inner))
    raise ValueError(f"seq() must contain tensor() or map(), got: {inner!r}")


def _parse_optional(s: str) -> Optional:
    inner = _unwrap("optional", s)
    if inner.startswith("tensor("):
        return Optional(inner=_parse_tensor(inner))
    if inner.startswith("seq("):
        return Optional(inner=_parse_seq(inner))
    raise ValueError(f"optional() must contain tensor() or seq(), got: {inner!r}")


def _unwrap(prefix: str, s: str) -> str:
    """Strip `prefix(` and trailing `)`, returning the inner content."""
    if not s.startswith(prefix + "(") or not s.endswith(")"):
        raise ValueError(f"Expected {prefix}(...), got: {s!r}")
    return s[len(prefix) + 1 : -1]


def _validate_onnx_type_str(v: str) -> str:
    """Validate that a type string conforms to the ONNX type grammar."""
    parse_onnx_type(v)
    return v


OnnxTypeStr = Annotated[str, AfterValidator(_validate_onnx_type_str)]


class TypeConstraint(BaseModel):
    name: str
    doc: str
    allowed: list[OnnxTypeStr]


class Variadic(BaseModel):
    homogeneous: bool
    min_arity: int


class Variable(BaseModel):
    name: str
    type: str
    doc: str
    kind: Literal["Single", "Option"] | Variadic = "Single"
    differentiation_category: DifferentiationCategory | None = None


class Attribute(BaseModel):
    name: str
    type: AttributeType
    doc: str
    required: bool
    default: str | int | float | list | None = None


T = TypeVar("T")


class NamedListDiff(BaseModel, Generic[T]):
    add: list[T] = []
    remove: list[str] = []
    update: list[T] = []

    def is_empty(self) -> bool:
        return not self.add and not self.remove and not self.update


class BaseSpec(BaseModel):
    """A fully-specified operator (no inheritance)."""

    op: str
    domain: str
    since_version: int
    doc: str
    type_constraints: list[TypeConstraint] = []
    inputs: list[Variable] = []
    outputs: list[Variable] = []
    attributes: list[Attribute] = []

    def diff(self, child: BaseSpec, inherit: str) -> InheritedSpec:
        """Compute the inheritance diff from self (parent) to child."""
        tc_diff = _diff_named_list(self.type_constraints, child.type_constraints)
        inputs_diff = _diff_named_list(self.inputs, child.inputs)
        outputs_diff = _diff_named_list(self.outputs, child.outputs)
        attrs_diff = _diff_named_list(self.attributes, child.attributes)

        return InheritedSpec(
            inherit=inherit,
            since_version=child.since_version,
            doc=child.doc if child.doc != self.doc else None,
            type_constraints=tc_diff if not tc_diff.is_empty() else None,
            inputs=inputs_diff if not inputs_diff.is_empty() else None,
            outputs=outputs_diff if not outputs_diff.is_empty() else None,
            attributes=attrs_diff if not attrs_diff.is_empty() else None,
        )

    def apply_diff(self, diff: InheritedSpec) -> BaseSpec:
        """Apply an InheritedSpec's diffs to self, returning a new BaseSpec."""
        return BaseSpec(
            op=self.op,
            domain=self.domain,
            since_version=diff.since_version,
            doc=diff.doc if diff.doc is not None else self.doc,
            type_constraints=_apply_list_diff(
                self.type_constraints, diff.type_constraints
            ),
            inputs=_apply_list_diff(self.inputs, diff.inputs),
            outputs=_apply_list_diff(self.outputs, diff.outputs),
            attributes=_apply_list_diff(self.attributes, diff.attributes),
        )


class InheritedSpec(BaseModel):
    """A spec that inherits from a previous version and applies diffs."""

    inherit: str
    since_version: int
    doc: str | None = None
    type_constraints: NamedListDiff[TypeConstraint] | None = None
    inputs: NamedListDiff[Variable] | None = None
    outputs: NamedListDiff[Variable] | None = None
    attributes: NamedListDiff[Attribute] | None = None


class ResolvedSpec(BaseModel):
    """A fully-resolved spec paired with its diff (if inherited)."""

    base: BaseSpec
    diff: InheritedSpec | None = None


def resolve_versions(specs: list[BaseSpec | InheritedSpec]) -> list[ResolvedSpec]:
    """Resolve a version chain into a list of ResolvedSpecs."""
    if not specs:
        return []

    first = specs[0]
    if not isinstance(first, BaseSpec):
        raise TypeError("First spec in a version chain must be a BaseSpec")

    resolved: list[ResolvedSpec] = [ResolvedSpec(base=first)]
    for spec in specs[1:]:
        if isinstance(spec, BaseSpec):
            resolved.append(ResolvedSpec(base=spec))
        else:
            resolved.append(
                ResolvedSpec(base=resolved[-1].base.apply_diff(spec), diff=spec)
            )

    return resolved


def _apply_list_diff(
    items: list[T],
    diff: NamedListDiff[T] | None,
) -> list[T]:
    """Apply a NamedListDiff to a list, returning a new list."""
    if diff is None:
        return list(items)

    result = [i for i in items if i.name not in set(diff.remove)]
    result.extend(diff.add)

    for update_item in diff.update:
        result = [update_item if i.name == update_item.name else i for i in result]

    return result


def _diff_named_list(
    parent_items: list[T],
    child_items: list[T],
) -> NamedListDiff[T]:
    """Diff a named list."""
    parent_by_name = {item.name: item for item in parent_items}

    remove = [
        name
        for name in parent_by_name
        if not any(item.name == name for item in child_items)
    ]
    add = [item for item in child_items if item.name not in parent_by_name]
    update = [
        item
        for item in child_items
        if item.name in parent_by_name and item != parent_by_name[item.name]
    ]

    return NamedListDiff(add=add, remove=remove, update=update)
