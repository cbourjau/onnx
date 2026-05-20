---
op: If
domain: ai.onnx
since_version: 1
deterministic: false
type_constraints:
- name: V
  doc: All Tensor types
  allowed:
  - tensor(bool)
  - tensor(complex128)
  - tensor(complex64)
  - tensor(double)
  - tensor(float)
  - tensor(float16)
  - tensor(int16)
  - tensor(int32)
  - tensor(int64)
  - tensor(int8)
  - tensor(string)
  - tensor(uint16)
  - tensor(uint32)
  - tensor(uint64)
  - tensor(uint8)
- name: B
  doc: Only bool
  allowed:
  - tensor(bool)
inputs:
- name: cond
  type: B
  doc: Condition for the if. The tensor must contain a single element.
outputs:
- name: outputs
  type: V
  doc: Values that are live-out to the enclosing scope. The return values in the `then_branch`
    and `else_branch` must be of the same shape and same data type.
  kind:
    homogeneous: false
    min_arity: 1
attributes:
- name: else_branch
  type: graph
  doc: 'Graph to run if condition is false. Has N outputs: values you wish to be live-out
    to the enclosing scope. The number of outputs must match the number of outputs
    in the then_branch.'
  required: true
- name: then_branch
  type: graph
  doc: 'Graph to run if condition is true. Has N outputs: values you wish to be live-out
    to the enclosing scope. The number of outputs must match the number of outputs
    in the else_branch.'
  required: true
---

If conditional
