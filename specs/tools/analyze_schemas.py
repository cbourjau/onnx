# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Analyze and migrate C++ schema registrations.

Uses tree-sitter to parse C++ files under onnx/defs/ and identify
ONNX_OPERATOR_SET_SCHEMA invocations. Reports which ops can be migrated
(i.e. have their SetDoc/Attr/Input/Output/TypeConstraint calls replaced
with a single .FillUsing(Op_vN_FillSpec)) and which cannot.

Usage:
    python specs/tools/analyze_schemas.py [--verbose]
    python specs/tools/analyze_schemas.py --migrate [FILE ...]
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import tree_sitter_cpp as tscpp
from tree_sitter import Language, Node, Parser, Tree

if TYPE_CHECKING:
    from collections.abc import Generator

DEFS_DIR = Path(__file__).resolve().parent.parent.parent / "onnx" / "defs"

# Methods that will be replaced by FillUsing
REMOVABLE_METHODS = frozenset(
    {
        "SetDoc",
        "Attr",
        "Input",
        "Output",
        "TypeConstraint",
        "SetSupportLevel",
        "Deprecate",
        "SetNodeDeterminism",
    }
)

# Methods that must remain in the chain after FillUsing
KEEPABLE_METHODS = frozenset(
    {
        "TypeAndShapeInferenceFunction",
        "PartialDataPropagationFunction",
        "FunctionBody",
        "SetContextDependentFunctionBodyBuilder",
        "NumOutputs",
        "AllowUncheckedAttributes",
    }
)

# Macros we handle
SUPPORTED_MACROS = frozenset(
    {"ONNX_OPERATOR_SET_SCHEMA", "ONNX_ML_OPERATOR_SET_SCHEMA"}
)

# Generators that use a named (already-defined) function for TypeAndShapeInference
# rather than an inline lambda. These are trivially migratable: replace the generator
# with Op_vN_FillSpec and add .TypeAndShapeInferenceFunction(named_func) to the chain.
GENERATORS_WITH_NAMED_INFERENCE: dict[str, str] = {
    # Already used named functions
    "defs::math::utils::UnaryFloatMathOpGenerator": "propagateShapeAndTypeFromFirstInput",
    "BinaryLogicDocGenerator": "binaryLogicalOpInference",
    "BinaryLogicDocGenerator_opset1": "logicalOpInference_opset1",
    "BinaryLogicDocGenerator_opset7": "binaryLogicalOpInference_opset7",
    "BinaryLogicDocGenerator_opset12": "binaryLogicalOpInference_opset7",
    "BinaryBitwiseDocGenerator": "bitwiseBroadcastShapeInference",
    "RNNDocGenerator": "RNNShapeInference",
    "RNNDocGenerator_opset14": "RNNShapeInference_opset14",
    "RNNDocGeneratorOld": "RNNShapeInference_opset1_to_6",
    "RNNDocGenerator_opset1_to_6": "RNNShapeInference_opset1_to_6",
    "RNNDocGenerator_opset7_to_13": "RNNShapeInference_opset7_to_13",
    "GlobalPoolingOpSchemaGenerator": "globalPoolTypeShapeInference",
    "GlobalPoolingOpSchemaGenerator_opset1": "globalPoolTypeShapeInference",
    "GlobalLpPoolingOpSchemaGenerator": "globalPoolTypeShapeInference",
    "GlobalLpPoolingOpSchemaGenerator_opset2": "globalPoolTypeShapeInference",
    # Reduce generators — lambda extracted to named function
    "ReduceOpGenerator": "reduceShapeInference",
    "ReduceOpDynamicAxes": "reduceShapeInference",
    "ReduceFunctionOp": "reduceShapeInference",
    "ReduceDocGenerator_opset12": "reduceShapeInference_opset12",
    "ReduceDocGenerator_opset1": "reduceShapeInference_opset1",
    "ArgReduceDocGenerator": "argReduceShapeInference",
    "ArgReduceDocGenerator_opset12": "argReduceShapeInference_opset12",
    "ArgReduceDocGenerator_opset1": "argReduceShapeInference_opset1",
    "ArgReduceDocGenerator_opset11": "argReduceShapeInference_opset12",
    # Math generators — lambda extracted to named function
    "MathDocGenerator": "binaryBroadcastShapeInference",
    "MathDocGenerator_opset13": "binaryBroadcastShapeInference",
    "MathDocGenerator_opset_7": "binaryBroadcastShapeInference",
    "MathDocGenerator_old": "propagateShapeAndTypeFromFirstInput",
    "MathDocGenerator_old_opset6": "propagateShapeAndTypeFromFirstInput",
    "ElementwiseMultiOpDocGenerator": "elementwiseMultiOpShapeInference",
    "ElementwiseMultiOpDocGenerator_opset8": "elementwiseMultiOpShapeInference_opset8",
    "ElementwiseMultiOpDocGenerator_old": "elementwiseMultiOpShapeInference_opset8",
    "SoftmaxFamilyDocGenerator": "softmaxShapeInference",
    "SoftmaxFamilyDocGenerator_opset_11": "softmaxShapeInference_opset11",
    "SoftmaxFamilyDocGenerator_opset1": "propagateShapeAndTypeFromFirstInput",
    "CosineSumWindowOpDocGenerator": "cosineSumWindowShapeInference",
    "defs::math::utils::TopKOpGenerator": "defs::math::utils::topKShapeInference",
    # NN generators — lambda extracted to named function
    "ConvOpSchemaGenerator": "convShapeInference",
    "ConvOpSchemaGenerator_opset11": "convShapeInference_opset11",
    "ConvOpSchemaGenerator_opset1": "convShapeInference_opset1",
    "ConvTransposeOpSchemaGenerator": "convTransposeInference",
    "ConvTransposeOpSchemaGenerator_opset11": "convTransposeShapeInference_opset11",
    "ConvTransposeOpSchemaGenerator_opset1": "convTransposeShapeInference_opset1",
    "RoiPoolOpSchemaGenerator": "roiPoolInference",
    "RoiPoolOpSchemaGenerator_opset1": "roiPoolTypeShapeInference_opset1",
    "LpPoolOpSchemaGenerator": "lpPoolShapeInference",
    "LpPoolOpSchemaGenerator_opset18": "lpPoolShapeInference_opset18",
    "LpPoolOpSchemaGenerator_opset11": "lpPoolShapeInference_opset11",
    "LpPoolOpSchemaGenerator_opset2": "lpPoolShapeInference_opset2",
    "PoolOpSchemaGenerator_with_dilation": "poolShapeInference_with_dilation",
    "PoolOpSchemaGenerator_opset19_with_dilation": "poolShapeInference_opset19_with_dilation",
    "PoolOpSchemaGenerator_opset10_no_dilation": "poolShapeInference_opset1_to_11_no_dilation",
    "PoolOpSchemaGenerator_opset10_to_11": "poolShapeInference_opset1_to_11_with_dilation",
    "PoolOpSchemaGenerator_opset11_with_dilation": "poolShapeInference_opset1_to_11_with_dilation",
    "PoolOpSchemaGenerator_opset1_to_8": "poolShapeInference_opset1_to_8",
    # Tensor/sequence generators
    "PadDocGenerator": "padShapeInference",
    "defs::sequence::utils::SplitToSequenceOpGenerator": "defs::sequence::utils::splitToSequenceShapeInference",
}


class AnalysisError(Exception):
    """Raised when the script encounters unexpected state that indicates a bug."""


@dataclasses.dataclass
class MethodInfo:
    """A method call in the OpSchema chain with its source text."""

    name: str
    args_text: (
        str  # source text of the argument list including parens, e.g. "(foo, bar)"
    )


@dataclasses.dataclass
class MacroLocation:
    """Location and identity of a macro invocation in source."""

    file: Path
    line: int
    macro: str
    op_name: str
    version: int
    methods: list[str]  # method names in the chain
    schema_expr_start: int  # byte offset of the schema expression
    schema_expr_end: int  # byte offset end of the schema expression
    schema_expr_col: int  # column of OpSchema() for indentation


@dataclasses.dataclass
class SchemaRegistration:
    """Represents a single ONNX_OPERATOR_SET_SCHEMA invocation."""

    location: MacroLocation
    doc_var: str | None  # variable name passed to SetDoc, if any
    fill_using_generator: str | None  # generator expression passed to FillUsing, if any
    can_migrate: bool
    skip_reason: str | None = None
    kept_methods: list[MethodInfo] = dataclasses.field(default_factory=list)
    inference_func: str | None = None  # named inference function to add

    def apply(self) -> None:
        """Apply migration edit to the file in-place.

        Replaces the schema expression with the new FillUsing chain and
        removes the doc string variable declaration if present.
        """
        loc = self.location
        indent = " " * (loc.schema_expr_col + 4)
        fill_func = f"{loc.op_name}_v{loc.version}_FillSpec"

        parts = [f"OpSchema()\n{indent}.FillUsing({fill_func})"]
        if self.inference_func:
            parts.append(
                f"\n{indent}.TypeAndShapeInferenceFunction({self.inference_func})"
            )
        parts.extend(f"\n{indent}.{m.name}{m.args_text}" for m in self.kept_methods)
        replacement = "".join(parts)

        source = loc.file.read_bytes()
        result = (
            source[: loc.schema_expr_start]
            + replacement.encode()
            + source[loc.schema_expr_end :]
        )

        loc.file.write_bytes(result)


def _get_parser() -> Parser:
    """Create a tree-sitter parser for C++."""
    language = Language(tscpp.language())
    return Parser(language)


def _get_method_chain(node: Node, source: bytes) -> list[tuple[str, Node | None]]:
    """Extract the method chain from an OpSchema() expression.

    Tree-sitter represents `OpSchema().A().B()` as nested call_expressions:
    the outermost node is `.B()`, whose receiver is `.A()`, whose receiver
    is `OpSchema()`. This function walks inward and reverses to return the
    chain in source order.

    Returns list of (method_name, arguments_node) tuples in order.
    Raises AnalysisError if the chain structure is unexpected.
    """
    chain: list[tuple[str, Node | None]] = []

    while True:
        if node.type != "call_expression":
            raise AnalysisError(
                f"Expected call_expression, got {node.type!r} at byte {node.start_byte}"
            )

        func = node.child_by_field_name("function")
        if func is None:
            raise AnalysisError(
                f"call_expression at byte {node.start_byte} has no 'function' child"
            )

        if func.type == "field_expression":
            obj = func.child_by_field_name("argument")
            field = func.child_by_field_name("field")
            if obj is None:
                raise AnalysisError(
                    f"field_expression at byte {func.start_byte} has no 'argument' child"
                )
            if field is None:
                raise AnalysisError(
                    f"field_expression at byte {func.start_byte} has no 'field' child"
                )
            method_name = source[field.start_byte : field.end_byte].decode()
            args = node.child_by_field_name("arguments")
            chain.append((method_name, args))
            node = obj
        elif func.type == "identifier":
            name = source[func.start_byte : func.end_byte].decode()
            chain.append((name, node.child_by_field_name("arguments")))
            break
        else:
            raise AnalysisError(
                f"Unexpected function node type {func.type!r} at byte {func.start_byte}. "
                f"Text: {source[func.start_byte : func.end_byte].decode()!r}"
            )

    chain.reverse()
    return chain


def _extract_doc_var(args_node: Node | None, source: bytes) -> str | None:
    """Extract the variable name from a SetDoc(var_name) call.

    Returns the variable name if it's a simple identifier, None if the doc
    is an inline string literal (which is valid but means no separate variable
    to remove).
    """
    if args_node is None:
        raise AnalysisError("SetDoc call has no arguments node")
    # Get the text of the arguments (excluding parens)
    args_text = source[args_node.start_byte : args_node.end_byte].decode()
    # Strip parens
    inner = args_text.strip()
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1].strip()
    # If it's a simple identifier (variable name), return it
    if inner.isidentifier():
        return inner
    # Otherwise it's an inline string or expression — no variable to track
    return None


def _find_macro_invocations(tree: Tree, source: bytes) -> list[tuple[int, str, Node]]:
    """Find all ONNX_OPERATOR_SET_SCHEMA(...) macro invocations.

    Returns list of (line, macro_name, node) for each invocation.
    """
    results: list[tuple[int, str, Node]] = []
    root = tree.root_node

    # Macro invocations appear as expression_statement -> call_expression
    # in tree-sitter's C++ grammar
    for node in _iter_nodes(root):
        if node.type == "call_expression":
            func = node.child_by_field_name("function")
            if func is not None and func.type == "identifier":
                name = source[func.start_byte : func.end_byte].decode()
                if name in SUPPORTED_MACROS:
                    results.append((node.start_point[0] + 1, name, node))

    return results


def _iter_nodes(node: Node) -> Generator[Node, None, None]:
    """Iterate all nodes in BFS order."""
    queue: list[Node] = [node]
    while queue:
        current = queue.pop(0)
        yield current
        queue.extend(current.children)


def _generator_name(expr: str) -> str:
    """Extract the generator function name from a FillUsing argument expression.

    E.g. 'BinaryLogicDocGenerator("and")' -> 'BinaryLogicDocGenerator'
         'MathDocGenerator("Add")' -> 'MathDocGenerator'
    """
    paren_idx = expr.find("(")
    if paren_idx == -1:
        return expr.strip()
    return expr[:paren_idx].strip()


def _extract_macro_args(
    node: Node, source: bytes, filepath: Path, line: int
) -> tuple[str, int, Node]:
    """Extract (op_name, version, schema_expr) from macro call arguments.

    ONNX_OPERATOR_SET_SCHEMA(OpName, version, OpSchema()...)

    Raises AnalysisError if the macro structure cannot be parsed.
    """
    args_node = node.child_by_field_name("arguments")
    if args_node is None:
        raise AnalysisError(
            f"{filepath}:{line}: Macro invocation has no arguments node"
        )

    # The argument_list children (skipping '(' and ')' and ',')
    arg_children = [
        c for c in args_node.children if c.type not in ("(", ")", ",", "comment")
    ]

    if len(arg_children) < 3:
        raise AnalysisError(
            f"{filepath}:{line}: Expected at least 3 macro arguments, got {len(arg_children)}"
        )
    if len(arg_children) > 3:
        raise AnalysisError(
            f"{filepath}:{line}: Expected exactly 3 macro arguments, got {len(arg_children)}"
        )

    op_name = source[arg_children[0].start_byte : arg_children[0].end_byte].decode()
    version_text = source[
        arg_children[1].start_byte : arg_children[1].end_byte
    ].decode()

    try:
        version = int(version_text.strip())
    except ValueError as e:
        raise AnalysisError(
            f"{filepath}:{line}: Cannot parse version {version_text!r} as integer"
        ) from e

    schema_expr = arg_children[2]
    return (op_name, version, schema_expr)


def _parse_registration(
    filepath: Path, line: int, macro_name: str, node: Node, source: bytes
) -> SchemaRegistration:
    """Parse a single macro invocation into a SchemaRegistration.

    Raises AnalysisError if the structure cannot be parsed.
    """
    op_name, version, schema_expr = _extract_macro_args(node, source, filepath, line)
    chain = _get_method_chain(schema_expr, source)

    if not chain:
        raise AnalysisError(
            f"{filepath}:{line}: Empty method chain for {op_name} v{version}"
        )
    if chain[0][0] != "OpSchema":
        raise AnalysisError(
            f"{filepath}:{line}: Chain for {op_name} v{version} starts with "
            f"{chain[0][0]!r}, expected 'OpSchema'"
        )

    methods = chain[1:]  # Skip OpSchema() itself
    method_names = [m for m, _ in methods]

    # Common fields for all registrations
    common = MacroLocation(
        file=filepath,
        line=line,
        macro=macro_name,
        op_name=op_name,
        version=version,
        methods=method_names,
        schema_expr_start=schema_expr.start_byte,
        schema_expr_end=schema_expr.end_byte,
        schema_expr_col=schema_expr.start_point[1],
    )

    # Check for existing FillUsing — uses a shared generator
    if "FillUsing" in method_names:
        return _parse_fill_using_registration(common, methods, method_names, source)

    # Every method MUST be either removable or keepable — fail on unknowns
    unknown_methods = [
        m
        for m in method_names
        if m not in REMOVABLE_METHODS and m not in KEEPABLE_METHODS
    ]
    if unknown_methods:
        raise AnalysisError(
            f"{filepath}:{line}: {op_name} v{version} has unknown methods: {unknown_methods}. "
            f"Add them to REMOVABLE_METHODS or KEEPABLE_METHODS."
        )

    # Must have at least one removable method to be worth migrating
    removable = [m for m in method_names if m in REMOVABLE_METHODS]
    if not removable:
        raise AnalysisError(
            f"{filepath}:{line}: {op_name} v{version} has no removable methods "
            f"but is not using FillUsing. Methods: {method_names}"
        )

    # Extract doc variable name if SetDoc is used
    doc_var = None
    for method_name, args in methods:
        if method_name == "SetDoc":
            doc_var = _extract_doc_var(args, source)
            break

    return SchemaRegistration(
        location=common,
        doc_var=doc_var,
        fill_using_generator=None,
        can_migrate=True,
        kept_methods=_collect_kept_methods(methods, source),
    )


def _parse_fill_using_registration(
    common: MacroLocation,
    methods: list[tuple[str, Node | None]],
    method_names: list[str],
    source: bytes,
) -> SchemaRegistration:
    """Parse a registration that already uses FillUsing with a generator."""
    generator_expr = None
    for method_name, args in methods:
        if method_name == "FillUsing" and args is not None:
            generator_expr = (
                source[args.start_byte + 1 : args.end_byte - 1].decode().strip()
            )
            break

    gen_name = _generator_name(generator_expr) if generator_expr else None

    # Already migrated — nothing to do
    if generator_expr and generator_expr.endswith("_FillSpec"):
        return SchemaRegistration(
            location=common,
            doc_var=None,
            fill_using_generator=generator_expr,
            can_migrate=False,
            skip_reason="Already migrated",
        )

    has_external_inference = (
        "TypeAndShapeInferenceFunction" in method_names
        or gen_name in GENERATORS_WITH_NAMED_INFERENCE
    )

    if not has_external_inference:
        return SchemaRegistration(
            location=common,
            doc_var=None,
            fill_using_generator=generator_expr,
            can_migrate=False,
            skip_reason="Uses FillUsing with inference inside generator",
        )

    # Determine inference function to add
    inference_func = None
    if "TypeAndShapeInferenceFunction" not in method_names:
        if gen_name and gen_name in GENERATORS_WITH_NAMED_INFERENCE:
            inference_func = GENERATORS_WITH_NAMED_INFERENCE[gen_name]

    return SchemaRegistration(
        location=common,
        doc_var=None,
        fill_using_generator=generator_expr,
        can_migrate=True,
        kept_methods=_collect_kept_methods(methods, source),
        inference_func=inference_func,
    )


def _collect_kept_methods(
    methods: list[tuple[str, Node | None]],
    source: bytes,
) -> list[MethodInfo]:
    """Extract MethodInfo for methods that should be kept in the chain."""
    kept: list[MethodInfo] = []
    for name, args_node in methods:
        if name in KEEPABLE_METHODS and args_node is not None:
            args_text = source[args_node.start_byte : args_node.end_byte].decode()
            kept.append(MethodInfo(name=name, args_text=args_text))
    return kept


def analyze_file(filepath: Path, parser: Parser) -> list[SchemaRegistration]:
    """Analyze a single C++ file for schema registrations.

    Raises AnalysisError on unexpected parse states.
    """
    source = filepath.read_bytes()
    tree = parser.parse(source)
    invocations = _find_macro_invocations(tree, source)
    return [
        _parse_registration(filepath, line, macro_name, node, source)
        for line, macro_name, node in invocations
    ]


def analyze_all() -> list[SchemaRegistration]:
    """Analyze all C++ files under onnx/defs/.

    Raises AnalysisError on any unexpected state.
    """
    parser = _get_parser()
    all_registrations = []

    cc_files = sorted(DEFS_DIR.glob("**/*.cc"))
    if not cc_files:
        raise AnalysisError(f"No .cc files found under {DEFS_DIR}")

    for filepath in cc_files:
        registrations = analyze_file(filepath, parser)
        all_registrations.extend(registrations)

    if not all_registrations:
        raise AnalysisError("No schema registrations found in any file")

    return all_registrations


def print_report(registrations: list[SchemaRegistration], verbose: bool) -> None:
    """Print a summary report of migration eligibility."""
    migratable = [r for r in registrations if r.can_migrate]
    non_migratable = [r for r in registrations if not r.can_migrate]

    print(
        f"\nTotal: {len(registrations)} | Migratable: {len(migratable)} | Non-migratable: {len(non_migratable)}"
    )

    if verbose:
        by_file: dict[Path, list[SchemaRegistration]] = {}
        for r in migratable:
            by_file.setdefault(r.location.file, []).append(r)
        for filepath, regs in sorted(by_file.items()):
            rel = filepath.relative_to(DEFS_DIR.parent.parent)
            print(f"\n  {rel} ({len(regs)} ops):")
            for r in regs:
                print(f"    {r.location.op_name} v{r.location.version}")

    if non_migratable:
        print(f"\nNon-migratable ({len(non_migratable)}):")
        for r in non_migratable:
            loc = r.location
            rel = loc.file.relative_to(DEFS_DIR.parent.parent)
            print(
                f"  {loc.op_name} v{loc.version} ({rel}:{loc.line}) — {r.skip_reason}"
            )


# --- Migration ---

_GENERATED_INCLUDE = b'#include "onnx/defs/generated/op_specs_generated.h"'


def _ensure_generated_include(filepath: Path) -> None:
    """Add the generated header include if not already present."""
    source = filepath.read_bytes()
    if _GENERATED_INCLUDE in source:
        return

    # Insert after the last #include line
    lines = source.split(b"\n")
    last_include_idx = -1
    for i, line in enumerate(lines):
        if line.startswith(b"#include"):
            last_include_idx = i

    if last_include_idx >= 0:
        lines.insert(last_include_idx + 1, _GENERATED_INCLUDE)
        filepath.write_bytes(b"\n".join(lines))


def migrate_file(filepath: Path, parser: Parser) -> int:
    """Apply migration edits to a single file, re-parsing after each edit.

    Returns the number of ops migrated.
    """
    count = 0
    while True:
        registrations = analyze_file(filepath, parser)
        migratable = [r for r in registrations if r.can_migrate]
        if not migratable:
            break

        migratable[0].apply()
        count += 1

    if count:
        _ensure_generated_include(filepath)

    return count


def migrate(files: list[Path] | None) -> None:
    """Run the migration on specified files or all files under onnx/defs/."""
    parser = _get_parser()

    if files:
        cc_files = [f.resolve() for f in files]
    else:
        cc_files = sorted(DEFS_DIR.glob("**/*.cc"))

    total_migrated = 0
    for filepath in cc_files:
        count = migrate_file(filepath, parser)
        if count:
            rel = filepath.relative_to(DEFS_DIR.parent.parent)
            print(f"  Migrated {count} ops in {rel}")
            total_migrated += count

    print(f"\nTotal migrated: {total_migrated}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze and migrate ONNX schema registrations."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed information"
    )
    parser.add_argument(
        "--migrate",
        action="store_true",
        help="Apply migration edits to C++ files",
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Specific files to analyze/migrate (default: all under onnx/defs/)",
    )
    args = parser.parse_args()

    if args.migrate:
        try:
            migrate(args.files or None)
        except AnalysisError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            if args.files:
                ts_parser = _get_parser()
                registrations = []
                for f in args.files:
                    registrations.extend(analyze_file(f.resolve(), ts_parser))
            else:
                registrations = analyze_all()
        except AnalysisError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        print_report(registrations, verbose=args.verbose)


if __name__ == "__main__":
    main()
