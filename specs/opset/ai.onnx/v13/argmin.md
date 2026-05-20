---
inherit: v12/argmin
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
  - name: data
    type: T
    doc: An input tensor.
    differentiable: false
outputs:
  update:
  - name: reduced
    type: tensor(int64)
    doc: Reduced output tensor with integer data type.
    differentiable: false
---

Computes the indices of the min elements of the input tensor's element along the
provided axis. The resulting tensor has the same rank as the input if keepdims equals 1.
If keepdims equals 0, then the resulting tensor has the reduced dimension pruned.
If select_last_index is True (default False), the index of the last occurrence of the min
is selected if the min appears more than once in the input. Otherwise the index of the
first occurrence is selected.
The type of the output tensor is integer.
