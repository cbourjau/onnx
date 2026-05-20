---
inherit: v9/constant
since_version: 11
attributes:
  add:
  - name: sparse_value
    type: sparse_tensor
    doc: The value for the elements of the output tensor in sparse format.
    required: false
  update:
  - name: value
    type: tensor
    doc: The value for the elements of the output tensor.
    required: false
---

A constant tensor. Exactly one of the two attributes, either value or sparse_value,
must be specified.
