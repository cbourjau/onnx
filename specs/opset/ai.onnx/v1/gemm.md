---
op: Gemm
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
- name: A
  type: T
  doc: Input tensor A
- name: B
  type: T
  doc: Input tensor B
- name: C
  type: T
  doc: Input tensor C, can be inplace.
outputs:
- name: Y
  type: T
  doc: Output tensor.
attributes:
- name: alpha
  type: float
  doc: Scalar multiplier for the product of input tensors A * B, the default value
    is 1.0.
  required: false
  default: 1.0
- name: beta
  type: float
  doc: Scalar multiplier for input tensor C, the default value is 1.0.
  required: false
  default: 1.0
- name: broadcast
  type: int
  doc: Whether C should be broadcasted
  required: false
  default: 0
- name: transA
  type: int
  doc: Whether A should be transposed
  required: false
  default: 0
- name: transB
  type: int
  doc: Whether B should be transposed
  required: false
  default: 0
---

General Matrix multiplication:
https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3
Compute Y = alpha * A * B + beta * C, where input tensor A has
dimension (M X K), input tensor B has dimension (K X N), input tensor C and
output tensor Y have dimension (M X N).
If attribute broadcast is non-zero, input tensor C will be broadcasted to match
the dimension requirement. A will be transposed before doing the computation
if attribute transA is non-zero, same for B and transB.
