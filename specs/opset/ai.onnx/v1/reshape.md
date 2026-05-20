---
op: Reshape
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
- name: data
  type: T
  doc: An input tensor.
outputs:
- name: reshaped
  type: T
  doc: Reshaped data.
attributes:
- name: consumed_inputs
  type: ints
  doc: legacy optimization attribute.
  required: false
- name: shape
  type: ints
  doc: New shape
  required: false
---

Reshape the input tensor similar to numpy.reshape.
It takes a tensor as input and an argument `shape`. It outputs the reshaped tensor.
At most one dimension of the new shape can be -1. In this case, the value is
inferred from the size of the tensor and the remaining dimensions. A dimension
could also be 0, in which case the actual dimension value is unchanged (i.e. taken
from the input tensor). Shape (second input) could be an empty shape, which means converting to a scalar.
The input tensor's shape and the output tensor's shape are required to have the same number of elements.
