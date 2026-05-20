---
inherit: v1/slice
since_version: 10
type_constraints:
  add:
  - name: Tind
    doc: Constrain indices to integer types
    allowed:
    - tensor(int32)
    - tensor(int64)
inputs:
  add:
  - name: starts
    type: Tind
    doc: 1-D tensor of starting indices of corresponding axis in `axes`
  - name: ends
    type: Tind
    doc: 1-D tensor of ending indices (exclusive) of corresponding axis in `axes`
  - name: axes
    type: Tind
    doc: 1-D tensor of axes that `starts` and `ends` apply to.
    kind: Option
  - name: steps
    type: Tind
    doc: '1-D tensor of slice step of corresponding axis in `axes`. Default to 1. '
    kind: Option
attributes:
  remove:
  - ends
  - starts
  - axes
---

Produces a slice of the input tensor along multiple axes. Similar to numpy:
https://numpy.org/doc/stable/reference/routines.indexing.html
Slices uses `starts`, `ends`, `axes` and `steps` inputs to specify the start and end
dimension and step for each axis in the list of axes, it uses this information to
slice the input `data` tensor. If a negative value is passed for any of the
start or end indices, it represent number of elements before the end of that
dimension. If the value passed to start or end is larger than the `n` (the
number of elements in this dimension), it represents `n`. For slicing to the
end of a dimension with unknown size, it is recommended to pass in `INT_MAX`.
If a negative value is passed for step, it represents slicing backward.
If `axes` are omitted, they are set to `[0, ..., ndim-1]`.
If `steps` are omitted, they are set to `[1, ..., 1]` of length `len(starts)`
Example 1:
  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  axes = [0, 1]
  starts = [1, 0]
  ends = [2, 3]
  steps = [1, 2]
  result = [
      [5, 7],
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
