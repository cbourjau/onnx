---
inherit: v1/maxpool
since_version: 8
type_constraints:
  add:
  - name: I
    doc: Constrain index tensor to int64
    allowed:
    - tensor(int64)
outputs:
  add:
  - name: Indices
    type: I
    doc: Indices tensor from max pooling across the input tensor. The dimensions of
      indices are the same as output tensor. The values in indices of are the indices
      of the selected values during pooling. The indices are computed as flatten 1-D
      tensor, and the indices do not consider padding. So the values in indices are
      in [0, N x C x D1 x ... x Dn).
    kind: Option
attributes:
  add:
  - name: storage_order
    type: int
    doc: The storage order of the tensor. 0 is row major, and 1 is column major.
    required: false
    default: 0
---
