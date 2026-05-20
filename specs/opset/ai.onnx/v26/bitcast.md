---
op: BitCast
domain: ai.onnx
since_version: 26
type_constraints:
- name: T1
  doc: Constrain input types. Bitcasting from string is not supported.
  allowed:
  - tensor(bfloat16)
  - tensor(bool)
  - tensor(complex128)
  - tensor(complex64)
  - tensor(double)
  - tensor(float)
  - tensor(float16)
  - tensor(float4e2m1)
  - tensor(float8e4m3fn)
  - tensor(float8e4m3fnuz)
  - tensor(float8e5m2)
  - tensor(float8e5m2fnuz)
  - tensor(float8e8m0)
  - tensor(int16)
  - tensor(int2)
  - tensor(int32)
  - tensor(int4)
  - tensor(int64)
  - tensor(int8)
  - tensor(uint16)
  - tensor(uint2)
  - tensor(uint32)
  - tensor(uint4)
  - tensor(uint64)
  - tensor(uint8)
- name: T2
  doc: Constrain output types. Bitcasting to string is not supported.
  allowed:
  - tensor(bfloat16)
  - tensor(bool)
  - tensor(complex128)
  - tensor(complex64)
  - tensor(double)
  - tensor(float)
  - tensor(float16)
  - tensor(float4e2m1)
  - tensor(float8e4m3fn)
  - tensor(float8e4m3fnuz)
  - tensor(float8e5m2)
  - tensor(float8e5m2fnuz)
  - tensor(float8e8m0)
  - tensor(int16)
  - tensor(int2)
  - tensor(int32)
  - tensor(int4)
  - tensor(int64)
  - tensor(int8)
  - tensor(uint16)
  - tensor(uint2)
  - tensor(uint32)
  - tensor(uint4)
  - tensor(uint64)
  - tensor(uint8)
inputs:
- name: input
  type: T1
  doc: Input tensor to be bitcast.
  differentiable: false
outputs:
- name: output
  type: T2
  doc: Output tensor with the same shape as the input.
  differentiable: false
attributes:
- name: to
  type: int
  doc: The data type to which the input tensor is bitwise reinterpreted. Must be one
    of the non-string types from DataType enum in TensorProto. The target type must
    have the same bit-width as the input type.
  required: true
---

Reinterprets the binary representation of a tensor as a different data type,
specified by the 'to' attribute. Unlike Cast, BitCast preserves the exact bit
pattern without any value conversion.

The target data type must have the same bit-width as the input data type.
The output tensor has the same shape as the input tensor.
All types except string are supported. Implementations must treat the
underlying bytes as little endian.
