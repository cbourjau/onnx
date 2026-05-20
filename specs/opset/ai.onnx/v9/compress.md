---
op: Compress
domain: ai.onnx
since_version: 9
type_constraints:
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
- name: T1
  doc: Constrain to boolean tensors.
  allowed:
  - tensor(bool)
inputs:
- name: input
  type: T
  doc: Tensor of rank r >= 1.
- name: condition
  type: T1
  doc: Rank 1 tensor of booleans to indicate which slices or data elements to be selected.
    Its length can be less than the input length alone the axis or the flattened input
    size if axis is not specified. In such cases data slices or elements exceeding
    the condition length are discarded.
outputs:
- name: output
  type: T
  doc: Tensor of rank r if axis is specified. Otherwise output is a Tensor of rank
    1.
attributes:
- name: axis
  type: int
  doc: (Optional) Axis along which to take slices. If not specified, input is flattened
    before elements being selected.
  required: false
---

Selects slices from an input tensor along a given axis where condition evaluates to True for each axis index.
    In case axis is not provided, input is flattened before elements are selected.
    Compress behaves like numpy.compress: https://docs.scipy.org/doc/numpy/reference/generated/numpy.compress.html
