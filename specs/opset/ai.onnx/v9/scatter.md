---
op: Scatter
domain: ai.onnx
since_version: 9
type_constraints:
- name: T
  doc: Input and output types can be of any tensor type.
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
- name: Tind
  doc: Constrain indices to integer types
  allowed:
  - tensor(int32)
  - tensor(int64)
inputs:
- name: data
  type: T
  doc: Tensor of rank r >= 1.
- name: indices
  type: Tind
  doc: Tensor of int32/int64 indices, of r >= 1 (same rank as input).
- name: updates
  type: T
  doc: Tensor of rank r >=1 (same rank and shape as indices)
outputs:
- name: output
  type: T
  doc: Tensor of rank r >= 1 (same rank as input).
attributes:
- name: axis
  type: int
  doc: Which axis to scatter on. Negative value means counting dimensions from the
    back. Accepted range is [-r, r-1]
  required: false
  default: 0
---

Given `data`, `updates` and `indices` input tensors of rank r >= 1, write the values provided by `updates`
into the first input, `data`, along `axis` dimension of `data` (by default outer-most one as axis=0) at corresponding `indices`.
For each entry in `updates`, the target index in `data` is specified by corresponding entry in `indices`
for dimension = axis, and index in source for dimension != axis. For instance, in a 2-D tensor case,
data[indices[i][j]][j] = updates[i][j] if axis = 0, or data[i][indices[i][j]] = updates[i][j] if axis = 1,
where i and j are loop counters from 0 up to the respective size in `updates` - 1.
Example 1:
  data = [
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
  ]
  indices = [
      [1, 0, 2],
      [0, 2, 1],
  ]
  updates = [
      [1.0, 1.1, 1.2],
      [2.0, 2.1, 2.2],
  ]
  output = [
      [2.0, 1.1, 0.0]
      [1.0, 0.0, 2.2]
      [0.0, 2.1, 1.2]
  ]
Example 2:
  data = [[1.0, 2.0, 3.0, 4.0, 5.0]]
  indices = [[1, 3]]
  updates = [[1.1, 2.1]]
  axis = 1
  output = [[1.0, 1.1, 3.0, 2.1, 5.0]]
