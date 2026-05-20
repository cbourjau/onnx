---
inherit: v13/reducel1
since_version: 18
function_body: |-
  <
    domain: "",
    opset_import: ["" : 18]
  >
  ReduceL1 <keepdims,noop_with_empty_axes>(data, axes) => (reduced)
  {
     data_abs = Abs (data)
     reduced = ReduceSum <keepdims: int = @keepdims, noop_with_empty_axes: int = @noop_with_empty_axes> (data_abs, axes)
  }
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
