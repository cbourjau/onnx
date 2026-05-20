---
op: Clip
domain: ai.onnx
since_version: 1
type_constraints:
- name: T
  doc: Constrain input and output types to float tensors.
  allowed:
  - tensor(double)
  - tensor(float)
  - tensor(float16)
inputs:
- name: input
  type: T
  doc: Input tensor whose elements to be clipped
outputs:
- name: output
  type: T
  doc: Output tensor with clipped input elements
attributes:
- name: consumed_inputs
  type: ints
  doc: legacy optimization attribute.
  required: false
- name: max
  type: float
  doc: Maximum value, above which element is replaced by max
  required: false
- name: min
  type: float
  doc: Minimum value, under which element is replaced by min
  required: false
---

Clip operator limits the given input within an interval. The interval is
specified with arguments 'min' and 'max'. They default to
numeric_limits::lowest() and numeric_limits::max() respectively.
