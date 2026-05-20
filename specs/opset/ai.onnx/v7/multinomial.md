---
op: Multinomial
domain: ai.onnx
since_version: 7
deterministic: false
type_constraints:
- name: T1
  doc: Constrain input types to float tensors.
  allowed:
  - tensor(double)
  - tensor(float)
  - tensor(float16)
- name: T2
  doc: Constrain output types to integral tensors.
  allowed:
  - tensor(int32)
  - tensor(int64)
inputs:
- name: input
  type: T1
  doc: Input tensor with shape [batch_size, class_size], where class_size is the number
    of all possible outcomes. Each value along the axis zero represents the unnormalized
    log-probability of each corresponding outcome in a batch.
outputs:
- name: output
  type: T2
  doc: Output tensor with shape [batch_size, sample_size], where sample_size is the
    number of times to sample. Each value along the axis zero represents the outcome
    of the corresponding sample in a batch.
attributes:
- name: dtype
  type: int
  doc: (Optional) The data type for the elements of the output tensor, if not specified,
    we will use int32.
  required: false
  default: 6
- name: sample_size
  type: int
  doc: Number of times to sample.
  required: false
  default: 1
- name: seed
  type: float
  doc: (Optional) Seed to the random generator, if not specified we will auto generate
    one.
  required: false
---

Generate a tensor of samples from a multinomial distribution according to the probabilities
of each of the possible outcomes.
