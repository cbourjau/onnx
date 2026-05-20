---
inherit: v11/squeeze
since_version: 13
type_constraints:
  update:
  - name: T
    doc: Constrain input and output types to all tensor types.
    allowed:
    - tensor(bfloat16)
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
  add:
  - name: axes
    type: tensor(int64)
    doc: List of integers indicating the dimensions to squeeze. Negative value means
      counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data).
    kind: Option
    differentiable: false
  update:
  - name: data
    type: T
    doc: Tensors with at least max(dims) dimensions.
    differentiable: true
outputs:
  update:
  - name: squeezed
    type: T
    doc: Reshaped tensor with same data as input.
    differentiable: true
attributes:
  remove:
  - axes
---

Remove single-dimensional entries from the shape of a tensor.
Takes an input `axes` with a list of axes to squeeze.
If `axes` is not provided, all the single dimensions will be removed from
the shape. If an axis is selected with shape entry not equal to one, an error is raised.
