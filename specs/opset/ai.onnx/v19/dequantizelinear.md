---
inherit: v13/dequantizelinear
since_version: 19
type_constraints:
  add:
  - name: T1
    doc: Constrain 'x_zero_point' and 'x' to 8-bit integer or float, or /32-bit integer
      tensor.
    allowed:
    - tensor(float8e4m3fn)
    - tensor(float8e4m3fnuz)
    - tensor(float8e5m2)
    - tensor(float8e5m2fnuz)
    - tensor(int32)
    - tensor(int8)
    - tensor(uint8)
  - name: T2
    doc: '''x_scale'' determines the output type.'
    allowed:
    - tensor(bfloat16)
    - tensor(float)
    - tensor(float16)
  remove:
  - T
inputs:
  update:
  - name: x
    type: T1
    doc: N-D quantized input tensor to be de-quantized.
  - name: x_scale
    type: T2
    doc: Scale for input 'x'. It can be a scalar, which means a per-tensor/layer dequantization,
      or a 1-D tensor for per-axis dequantization.
  - name: x_zero_point
    type: T1
    doc: Zero point for input 'x'. Shape must match x_scale. It's optional. Zero point
      is 0 when it's not specified.
    kind: Option
outputs:
  update:
  - name: y
    type: T2
    doc: N-D full precision output tensor. It has same shape as input 'x'.
attributes:
  update:
  - name: axis
    type: int
    doc: (Optional) The axis of the dequantizing dimension of the input tensor. Used
      only for per-axis quantization. Negative value means counting dimensions from
      the back. Accepted range is `[-r, r-1]` where `r = rank(input)`. When the rank
      of the input is 1, per-tensor quantization is applied, rendering the axis unnecessary
      in this scenario.
    required: false
    default: 1
---

The linear dequantization operator. It consumes a quantized tensor, a scale, and a zero point to compute the full precision tensor.
The dequantization formula is `y = (x - x_zero_point) * x_scale`. `x_scale` and `x_zero_point` must have same shape, and can be either a scalar
for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantization.
`x_zero_point` and `x` must have same type. `x` and `y` must have same shape. In the case of dequantizing int32,
there's no zero point (zero point is supposed to be 0).
`zero-point` is usually not used in the case of float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz quantization,
but the dequantization formula remains the same for consistency and 'x_scale' still determines the output type.
