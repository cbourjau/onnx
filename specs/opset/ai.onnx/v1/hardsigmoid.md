---
op: HardSigmoid
domain: ai.onnx
since_version: 1
type_constraints:
- name: T
  doc: Constrain input and output types to float tensors.
  allowed:
  - tensor(double)
  - tensor(float)
  - tensor(float16)
inputs:
- name: X
  type: T
  doc: Input tensor
outputs:
- name: Y
  type: T
  doc: Output tensor
attributes:
- name: alpha
  type: float
  doc: Value of alpha default to 0.2
  required: false
  default: 0.20000000298023224
- name: beta
  type: float
  doc: Value of beta default to 0.5
  required: false
  default: 0.5
- name: consumed_inputs
  type: ints
  doc: legacy optimization attribute.
  required: false
---

HardSigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the HardSigmoid function, y = max(0, min(1, alpha * x + beta)),
is applied to the tensor elementwise.
