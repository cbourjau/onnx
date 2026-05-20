---
inherit: v9/compress
since_version: 11
inputs:
  update:
  - name: input
    type: T
    doc: Tensor of rank r >= 1.
    differentiable: true
  - name: condition
    type: T1
    doc: Rank 1 tensor of booleans to indicate which slices or data elements to be
      selected. Its length can be less than the input length along the axis or the
      flattened input size if axis is not specified. In such cases data slices or
      elements exceeding the condition length are discarded.
    differentiable: false
outputs:
  update:
  - name: output
    type: T
    doc: Tensor of rank r if axis is specified. Otherwise output is a Tensor of rank
      1.
    differentiable: true
attributes:
  update:
  - name: axis
    type: int
    doc: (Optional) Axis along which to take slices. If not specified, input is flattened
      before elements being selected. Negative value means counting dimensions from
      the back. Accepted range is [-r, r-1] where r = rank(input).
    required: false
---
