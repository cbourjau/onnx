---
op: Dropout
domain: ai.onnx
since_version: 1
deterministic: false
type_constraints:
- name: T
  doc: Constrain input and output types to float tensors.
  allowed:
  - tensor(double)
  - tensor(float)
  - tensor(float16)
inputs:
- name: data
  type: T
  doc: The input data as Tensor.
outputs:
- name: output
  type: T
  doc: The output.
- name: mask
  type: T
  doc: The output mask. If is_test is nonzero, this output is not filled.
  kind: Option
attributes:
- name: consumed_inputs
  type: ints
  doc: legacy optimization attribute.
  required: false
- name: is_test
  type: int
  doc: (int, default 0) if nonzero, run dropout in test mode where the output is simply
    Y = X.
  required: false
  default: 0
- name: ratio
  type: float
  doc: (float, default 0.5) the ratio of random dropout
  required: false
  default: 0.5
---

Dropout takes one input data (Tensor<float>) and produces two Tensor outputs,
output (Tensor<float>) and mask (Tensor<bool>). Depending on whether it is in
test mode or not, the output Y will either be a random dropout, or a simple
copy of the input. Note that our implementation of Dropout does scaling in
the training phase, so during testing nothing needs to be done.
