---
op: SequenceEmpty
domain: ai.onnx
since_version: 11
type_constraints:
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
outputs:
- name: output
  type: S
  doc: Empty sequence.
attributes:
- name: dtype
  type: int
  doc: (Optional) The data type of the tensors in the output sequence. The default
    type is 'float'.
  required: false
---

Construct an empty tensor sequence, with given data type.
