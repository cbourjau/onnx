---
inherit: v11/constant
since_version: 12
attributes:
  add:
  - name: value_float
    type: float
    doc: The value for the sole element for the scalar, float32, output tensor.
    required: false
  - name: value_floats
    type: floats
    doc: The values for the elements for the 1D, float32, output tensor.
    required: false
  - name: value_int
    type: int
    doc: The value for the sole element for the scalar, int64, output tensor.
    required: false
  - name: value_ints
    type: ints
    doc: The values for the elements for the 1D, int64, output tensor.
    required: false
  - name: value_string
    type: string
    doc: The value for the sole element for the scalar, UTF-8 string, output tensor.
    required: false
  - name: value_strings
    type: strings
    doc: The values for the elements for the 1D, UTF-8 string, output tensor.
    required: false
---

This operator produces a constant tensor. Exactly one of the provided attributes, either value, sparse_value,
or value_* must be specified.
