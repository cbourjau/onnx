---
inherit: v10/dequantizelinear
since_version: 13
inputs:
  update:
  - name: x_scale
    type: tensor(float)
    doc: Scale for input 'x'. It can be a scalar, which means a per-tensor/layer dequantization,
      or a 1-D tensor for per-axis dequantization.
  - name: x_zero_point
    type: T
    doc: Zero point for input 'x'. Shape must match x_scale. It's optional. Zero point
      is 0 when it's not specified.
    kind: Option
attributes:
  add:
  - name: axis
    type: int
    doc: (Optional) The axis of the dequantizing dimension of the input tensor. Ignored
      for per-tensor quantization. Negative value means counting dimensions from the
      back. Accepted range is [-r, r-1] where r = rank(input).
    required: false
    default: 1
---

The linear dequantization operator. It consumes a quantized tensor, a scale, and a zero point to compute the full precision tensor.
The dequantization formula is `y = (x - x_zero_point) * x_scale`. `x_scale` and `x_zero_point` must have same shape, and can be either a scalar
for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantization.
`x_zero_point` and `x` must have same type. `x` and `y` must have same shape. In the case of dequantizing int32,
there's no zero point (zero point is supposed to be 0).
