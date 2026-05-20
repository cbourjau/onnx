---
op: Sign
domain: ai.onnx
since_version: 9
type_constraints:
- name: T
  doc: Constrain input and output types to all numeric tensors.
  allowed:
  - tensor(double)
  - tensor(float)
  - tensor(float16)
  - tensor(int16)
  - tensor(int32)
  - tensor(int64)
  - tensor(int8)
  - tensor(uint16)
  - tensor(uint32)
  - tensor(uint64)
  - tensor(uint8)
inputs:
- name: input
  type: T
  doc: Input tensor
outputs:
- name: output
  type: T
  doc: The sign of the input tensor computed element-wise. It has the same shape and
    type of the input.
---

Calculate the sign of the given input tensor element-wise.
If input > 0, output 1. if input < 0, output -1. if input == 0, output 0.
