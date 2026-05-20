---
inherit: v11/reducemax
since_version: 12
type_constraints:
  update:
  - name: T
    doc: Constrain input and output types to high-precision and 8 bit numeric tensors.
    allowed:
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

Computes the max of the input tensor's element along the provided axes. The resulting
tensor has the same rank as the input if keepdims equals 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy defaults keepdims to
False instead of True.
