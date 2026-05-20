---
op: Xor
domain: ai.onnx
since_version: 1
type_constraints:
- name: T
  doc: Constrain input to boolean tensor.
  allowed:
  - tensor(bool)
- name: T1
  doc: Constrain output to boolean tensor.
  allowed:
  - tensor(bool)
inputs:
- name: A
  type: T
  doc: Left input tensor for the logical operator.
- name: B
  type: T
  doc: Right input tensor for the logical operator.
outputs:
- name: C
  type: T1
  doc: Result tensor.
attributes:
- name: axis
  type: int
  doc: If set, defines the broadcast dimensions.
  required: false
- name: broadcast
  type: int
  doc: Enable broadcasting
  required: false
  default: 0
---

Returns the tensor resulted from performing the `xor` logical operation
elementwise on the input tensors `A` and `B`.

If broadcasting is enabled, the right-hand-side argument will be broadcasted
to match the shape of left-hand-side argument. See the doc of `Add` for a
detailed description of the broadcasting rules.
