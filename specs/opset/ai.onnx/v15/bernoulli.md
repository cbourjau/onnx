---
op: Bernoulli
domain: ai.onnx
since_version: 15
deterministic: false
type_constraints:
- name: T1
  doc: Constrain input types to float tensors.
  allowed:
  - tensor(double)
  - tensor(float)
  - tensor(float16)
- name: T2
  doc: Constrain output types to all numeric tensors and bool tensors.
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
  - tensor(uint16)
  - tensor(uint32)
  - tensor(uint64)
  - tensor(uint8)
inputs:
- name: input
  type: T1
  doc: All values in input have to be in the range:[0, 1].
outputs:
- name: output
  type: T2
  doc: The returned output tensor only has values 0 or 1, same shape as input tensor.
attributes:
- name: dtype
  type: int
  doc: The data type for the elements of the output tensor. if not specified, we will
    use the data type of the input tensor.
  required: false
- name: seed
  type: float
  doc: (Optional) Seed to the random generator, if not specified we will auto generate
    one.
  required: false
---

Draws binary random numbers (0 or 1) from a Bernoulli distribution. The input tensor should be a tensor
containing probabilities p (a value in the range [0,1]) to be used for drawing the binary random number,
where an output of 1 is produced with probability p and an output of 0 is produced with probability (1-p).

This operator is non-deterministic and may not produce the same values in different
implementations (even if a seed is specified).
