---
inherit: v18/reducemin
since_version: 20
type_constraints:
  update:
  - name: T
    doc: Constrain input and output types to numeric and Boolean tensors.
    allowed:
    - tensor(bfloat16)
    - tensor(bool)
    - tensor(double)
    - tensor(float)
    - tensor(float16)
    - tensor(int32)
    - tensor(int64)
    - tensor(int8)
    - tensor(uint32)
    - tensor(uint64)
    - tensor(uint8)
---

Computes the min of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields plus infinity (if supported by the datatype) or the maximum value of the data type otherwise.


If the input data type is Boolean, the comparison should consider `False < True`.

The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.
