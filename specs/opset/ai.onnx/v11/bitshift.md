---
op: BitShift
domain: ai.onnx
since_version: 11
type_constraints:
- name: T
  doc: Constrain input and output types to integer tensors.
  allowed:
  - tensor(uint16)
  - tensor(uint32)
  - tensor(uint64)
  - tensor(uint8)
inputs:
- name: X
  type: T
  doc: First operand, input to be shifted.
  differentiable: false
- name: Y
  type: T
  doc: Second operand, amounts of shift.
  differentiable: false
outputs:
- name: Z
  type: T
  doc: Output tensor
  differentiable: false
attributes:
- name: direction
  type: string
  doc: Direction of moving bits. It can be either "RIGHT" (for right shift) or "LEFT"
    (for left shift).
  required: true
---

Bitwise shift operator performs element-wise operation. For each input element, if the
attribute "direction" is "RIGHT", this operator moves its binary representation toward
the right side so that the input value is effectively decreased. If the attribute "direction"
is "LEFT", bits of binary representation moves toward the left side, which results the
increase of its actual value. The input X is the tensor to be shifted and another input
Y specifies the amounts of shifting. For example, if "direction" is "Right", X is [1, 4],
and S is [1, 1], the corresponding output Z would be [0, 2]. If "direction" is "LEFT" with
X=[1, 2] and S=[1, 2], the corresponding output Y would be [2, 8].

Because this operator supports Numpy-style broadcasting, X's and Y's shapes are
not necessarily identical.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
