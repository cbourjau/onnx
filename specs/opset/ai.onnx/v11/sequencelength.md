---
op: SequenceLength
domain: ai.onnx
since_version: 11
type_constraints:
- name: S
  doc: Constrain to any tensor type.
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
- name: I
  doc: Constrain output to integral tensor. It must be a scalar(tensor of empty shape).
  allowed:
  - tensor(int64)
inputs:
- name: input_sequence
  type: S
  doc: Input sequence.
outputs:
- name: length
  type: I
  doc: Length of input sequence. It must be a scalar(tensor of empty shape).
---

Produces a scalar(tensor of empty shape) containing the number of tensors in 'input_sequence'.
