---
op: Celu
domain: ai.onnx
since_version: 12
type_constraints:
- name: T
  doc: Constrain input and output types to float32 tensors.
  allowed:
  - tensor(float)
inputs:
- name: X
  type: T
  doc: Input tensor
  differentiable: true
outputs:
- name: Y
  type: T
  doc: Output tensor
  differentiable: true
attributes:
- name: alpha
  type: float
  doc: The Alpha value in Celu formula which control the shape of the unit. The default
    value is 1.0.
  required: false
  default: 1.0
---

Continuously Differentiable Exponential Linear Units:
Perform the linear unit element-wise on the input tensor X
using formula:

```
max(0,x) + min(0,alpha*(exp(x/alpha)-1))
```
