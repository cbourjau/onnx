---
op: Slice
domain: ai.onnx
since_version: 1
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
inputs:
- name: data
  type: T
  doc: Tensor of data to extract slices from.
outputs:
- name: output
  type: T
  doc: Sliced data tensor.
attributes:
- name: ends
  type: ints
  doc: Ending indices (exclusive) of corresponding axis in axes`
  required: true
- name: starts
  type: ints
  doc: Starting indices of corresponding axis in `axes`
  required: true
- name: axes
  type: ints
  doc: Axes that `starts` and `ends` apply to. It's optional. If not present, will
    be treated as [0, 1, ..., len(`starts`) - 1].
  required: false
---

Produces a slice of the input tensor along multiple axes. Similar to numpy:
https://numpy.org/doc/stable/reference/routines.indexing.html
Slices uses `axes`, `starts` and `ends` attributes to specify the start and end
dimension for each axis in the list of axes, it uses this information to
slice the input `data` tensor. If a negative value is passed for any of the
start or end indices, it represent number of elements before the end of that
dimension. If the value passed to start or end is larger than the `n` (the
number of elements in this dimension), it represents `n`. For slicing to the
end of a dimension with unknown size, it is recommended to pass in `INT_MAX`.
If `axes` are omitted, they are set to `[0, ..., ndim-1]`.
Example 1:
  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  axes = [0, 1]
  starts = [1, 0]
  ends = [2, 3]
  result = [
      [5, 6, 7],
  ]
Example 2:
  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  starts = [0, 1]
  ends = [-1, 1000]
  result = [
      [2, 3, 4],
  ]
