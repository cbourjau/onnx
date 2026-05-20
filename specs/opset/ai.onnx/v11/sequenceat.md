---
op: SequenceAt
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
- name: T
  doc: Constrain to any tensor type.
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
- name: I
  doc: Constrain position to integral tensor. It must be a scalar(tensor of empty
    shape).
  allowed:
  - tensor(int32)
  - tensor(int64)
inputs:
- name: input_sequence
  type: S
  doc: Input sequence.
- name: position
  type: I
  doc: Position of the tensor in the sequence. Negative value means counting positions
    from the back. Accepted range in `[-n, n - 1]`, where `n` is the number of tensors
    in 'input_sequence'. It is an error if any of the index values are out of bounds.
    It must be a scalar(tensor of empty shape).
outputs:
- name: tensor
  type: T
  doc: Output tensor at the specified position in the input sequence.
---

Outputs a tensor copy from the tensor at 'position' in 'input_sequence'.
Accepted range for 'position' is in `[-n, n - 1]`, where `n` is the number of tensors in 'input_sequence'.
Negative value means counting positions from the back.
