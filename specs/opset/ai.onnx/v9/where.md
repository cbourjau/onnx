---
op: Where
domain: ai.onnx
since_version: 9
type_constraints:
- name: B
  doc: Constrain to boolean tensors.
  allowed:
  - tensor(bool)
- name: T
  doc: Constrain input and output types to all tensor types.
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
inputs:
- name: condition
  type: B
  doc: When True (nonzero), yield X, otherwise yield Y
  differentiable: false
- name: X
  type: T
  doc: values selected at indices where condition is True
  differentiable: true
- name: Y
  type: T
  doc: values selected at indices where condition is False
  differentiable: true
outputs:
- name: output
  type: T
  doc: Tensor of shape equal to the broadcasted shape of condition, X, and Y.
  differentiable: true
---

Return elements, either from X or Y, depending on condition.
Where behaves like
[numpy.where](https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html)
with three parameters.

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
