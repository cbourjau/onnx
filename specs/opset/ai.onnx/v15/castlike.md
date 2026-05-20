---
op: CastLike
domain: ai.onnx
since_version: 15
type_constraints:
- name: T1
  doc: Constrain input types. Casting from complex is not supported.
  allowed:
  - tensor(bfloat16)
  - tensor(bool)
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
- name: T2
  doc: Constrain output types. Casting to complex is not supported.
  allowed:
  - tensor(bfloat16)
  - tensor(bool)
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
- name: input
  type: T1
  doc: Input tensor to be cast.
  differentiable: true
- name: target_type
  type: T2
  doc: The (first) input tensor will be cast to produce a tensor of the same type
    as this (second input) tensor.
  differentiable: false
outputs:
- name: output
  type: T2
  doc: Output tensor produced by casting the first input tensor to have the same type
    as the second input tensor.
  differentiable: true
---

The operator casts the elements of a given input tensor (the first input) to
the same data type as the elements of the second input tensor.
See documentation of the Cast operator for further details.
