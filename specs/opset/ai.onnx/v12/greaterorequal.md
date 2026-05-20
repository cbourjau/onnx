---
op: GreaterOrEqual
domain: ai.onnx
since_version: 12
function_body: |-
  <
    domain: "",
    opset_import: ["" : 12]
  >
  GreaterOrEqual (A, B) => (C)
  {
     O1 = Greater (A, B)
     O2 = Equal (A, B)
     C = Or (O1, O2)
  }
type_constraints:
- name: T
  doc: Constrain input types to all numeric tensors.
  allowed:
  - tensor(double)
  - tensor(float)
  - tensor(float16)
  - tensor(int16)
  - tensor(int32)
  - tensor(int64)
  - tensor(int8)
  - tensor(uint16)
  - tensor(uint32)
  - tensor(uint64)
  - tensor(uint8)
- name: T1
  doc: Constrain output to boolean tensor.
  allowed:
  - tensor(bool)
inputs:
- name: A
  type: T
  doc: First input operand for the logical operator.
  differentiable: false
- name: B
  type: T
  doc: Second input operand for the logical operator.
  differentiable: false
outputs:
- name: C
  type: T1
  doc: Result tensor.
  differentiable: false
---

Returns the tensor resulted from performing the `greater_equal` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
