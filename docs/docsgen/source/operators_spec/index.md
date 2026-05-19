<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# Operator Specification

This section defines all operators defined by the ONNX specification.
Operators are grouped in domains and opsets.
Operators from older opsets are available in newer opsets until they are overridden by an operator of the same name and domain.
The ONNX specification defines the `ai.onnx` and `ai.onnx.ml` domains.
Users may define custom operators in their own domains.

Each operator may specify Attributes, Inputs, Outputs, and type constraints.
Attributes are constants that parametrize the behavior of the operator.
Inputs (Outputs) enumerate the expected inputs (outputs) of the operation at run-time and their expected type.
The type constraints may appear in the former three items and constitute generic type parameters.

```{toctree}
:hidden:
:glob:

../_generated/operators_spec/*/index
```
