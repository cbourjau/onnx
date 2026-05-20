---
inherit: v1/cast
since_version: 6
attributes:
  update:
  - name: to
    type: int
    doc: The data type to which the elements of the input tensor are cast. Strictly
      must be one of the types from DataType enum in TensorProto
    required: true
---
