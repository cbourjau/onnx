---
inherit: v11/reducesum
since_version: 13
type_constraints:
  update:
  - name: T
    doc: Constrain input and output types to numeric tensors.
    allowed:
    - tensor(bfloat16)
    - tensor(double)
    - tensor(float)
    - tensor(float16)
    - tensor(int32)
    - tensor(int64)
    - tensor(uint32)
    - tensor(uint64)
inputs:
  add:
  - name: axes
    type: tensor(int64)
    doc: 'Optional input list of integers, along which to reduce. The default is to
      reduce over empty axes. When axes is empty (either not provided or explicitly
      empty), behavior depends on ''noop_with_empty_axes'': reduction over all axes
      if ''noop_with_empty_axes'' is false, and reduction over the empty set of axes
      when ''noop_with_empty_axes'' is true. Accepted range is [-r, r-1] where r =
      rank(data).'
    kind: Option
    differentiable: false
  update:
  - name: data
    type: T
    doc: An input tensor.
    differentiable: true
outputs:
  update:
  - name: reduced
    type: T
    doc: Reduced output tensor.
    differentiable: true
attributes:
  add:
  - name: noop_with_empty_axes
    type: int
    doc: Defines behavior when axes is not provided or is empty. If false (default),
      reduction happens over all axes (similar to the case when `axis=None` in numpy).
      If true, reduction happens over an empty set of axes (similar to the case when
      `axis=()` in numpy). Note that reduction over an empty set of axes means that
      the reduction step behaves like a no-op (identity function), but composite-reduction
      operators will still perform the non-reduction steps as needed. Thus, ReduceLogSum
      returns the Log of input tensor, and ReduceSumSquare returns the Square of the
      input tensor, in this case.
    required: false
    default: 0
  remove:
  - axes
---

Computes the sum of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields 0.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.
