---
inherit: v11/gemm
since_version: 13
type_constraints:
  update:
  - name: T
    doc: Constrain input and output types to float/int tensors.
    allowed:
    - tensor(bfloat16)
    - tensor(double)
    - tensor(float)
    - tensor(float16)
    - tensor(int32)
    - tensor(int64)
    - tensor(uint32)
    - tensor(uint64)
inputs:
  update:
  - name: A
    type: T
    doc: Input tensor A. The shape of A should be (M, K) if transA is 0, or (K, M)
      if transA is non-zero.
    differentiable: true
  - name: B
    type: T
    doc: Input tensor B. The shape of B should be (K, N) if transB is 0, or (N, K)
      if transB is non-zero.
    differentiable: true
  - name: C
    type: T
    doc: Optional input tensor C. If not specified, the computation is done as if
      C is a scalar 0. The shape of C should be unidirectional broadcastable to (M,
      N).
    kind: Option
    differentiable: true
outputs:
  update:
  - name: Y
    type: T
    doc: Output tensor of shape (M, N).
    differentiable: true
---

General Matrix multiplication:
https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3

* A' = transpose(A) if transA else A
* B' = transpose(B) if transB else B

Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K) or (K, M),
input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N),
and output tensor Y has shape (M, N). A will be transposed before doing the
computation if attribute transA is non-zero, same for B and transB.
This operator supports **unidirectional broadcasting** (tensor C should be unidirectional broadcastable to tensor A * B); for more details please check [the doc](Broadcasting.md).
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
