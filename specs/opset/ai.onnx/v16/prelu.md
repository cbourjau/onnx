---
inherit: v9/prelu
since_version: 16
function_body: |-
  <
    domain: "",
    opset_import: ["" : 16]
  >
  PRelu (X, slope) => (Y)
  {
     Zero = Constant <value: tensor = float {0}> ()
     ZeroCast = CastLike (Zero, X)
     XLessThanZero = Less (X, ZeroCast)
     SlopeMulX = Mul (slope, X)
     Y = Where (XLessThanZero, SlopeMulX, X)
  }
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
---

PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
`f(x) = x for x >= 0`., is applied to the data tensor elementwise.
This operator supports **unidirectional broadcasting** (tensor slope should be unidirectional broadcastable to input tensor X); for more details please check [the doc](Broadcasting.md).
