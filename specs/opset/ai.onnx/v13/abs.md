---
inherit: v6/abs
since_version: 13
type_constraints:
  update:
  - name: T
    doc: Constrain input and output types to all numeric tensors.
    allowed:
    - tensor(bfloat16)
    - tensor(double)
    - tensor(float)
    - tensor(float16)
    - tensor(int16)
    - tensor(int32)
    - tensor(int64)
    - tensor(int8)
    - tensor(uint16)
    - tensor(uint32)
    - tensor(uint64)
    - tensor(uint8)
inputs:
  update:
  - name: X
    type: T
    doc: Input tensor
    differentiable: true
outputs:
  update:
  - name: Y
    type: T
    doc: Output tensor
    differentiable: true
---

Absolute takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where absolute value, y = abs(x), is applied to
the tensor elementwise.
