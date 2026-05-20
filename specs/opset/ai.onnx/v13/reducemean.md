---
inherit: v11/reducemean
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
---

Computes the mean of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields undefined.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.
