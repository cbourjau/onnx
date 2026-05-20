---
op: SequenceConstruct
domain: ai.onnx
since_version: 11
type_constraints:
- name: T
  doc: Constrain input types to any tensor type.
  allowed:
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
- name: S
  doc: Constrain output types to any tensor type.
  allowed:
  - seq(tensor(bool))
  - seq(tensor(complex128))
  - seq(tensor(complex64))
  - seq(tensor(double))
  - seq(tensor(float))
  - seq(tensor(float16))
  - seq(tensor(int16))
  - seq(tensor(int32))
  - seq(tensor(int64))
  - seq(tensor(int8))
  - seq(tensor(string))
  - seq(tensor(uint16))
  - seq(tensor(uint32))
  - seq(tensor(uint64))
  - seq(tensor(uint8))
inputs:
- name: inputs
  type: T
  doc: Tensors.
  kind:
    homogeneous: true
    min_arity: 1
outputs:
- name: output_sequence
  type: S
  doc: Sequence enclosing the input tensors.
---

Construct a tensor sequence containing 'inputs' tensors.
All tensors in 'inputs' must have the same data type.
