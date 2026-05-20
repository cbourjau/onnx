---
inherit: v6/gemm
since_version: 7
inputs:
  update:
  - name: A
    type: T
    doc: Input tensor A. The shape of A should be (M, K) if transA is 0, or (K, M)
      if transA is non-zero.
  - name: B
    type: T
    doc: Input tensor B. The shape of B should be (K, N) if transB is 0, or (N, K)
      if transB is non-zero.
  - name: C
    type: T
    doc: Input tensor C. The shape of C should be unidirectional broadcastable to
      (M, N).
outputs:
  update:
  - name: Y
    type: T
    doc: Output tensor of shape (M, N).
attributes:
  remove:
  - broadcast
  update:
  - name: alpha
    type: float
    doc: Scalar multiplier for the product of input tensors A * B.
    required: false
    default: 1.0
  - name: beta
    type: float
    doc: Scalar multiplier for input tensor C.
    required: false
    default: 1.0
---

General Matrix multiplication:
https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3

A' = transpose(A) if transA else A

B' = transpose(B) if transB else B

Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K) or (K, M),
input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N),
and output tensor Y has shape (M, N). A will be transposed before doing the
computation if attribute transA is non-zero, same for B and transB.
This operator supports **unidirectional broadcasting** (tensor C should be unidirectional broadcastable to tensor A * B); for more details please check [the doc](Broadcasting.md).
