---
op: DynamicQuantizeLinear
domain: ai.onnx
since_version: 11
function_body: |-
  <
    domain: "",
    opset_import: ["" : 11]
  >
  DynamicQuantizeLinear (x) => (y, y_scale, y_zero_point)
  {
     Q_Min = Constant <value: tensor = float {0}> ()
     Q_Max = Constant <value: tensor = float {255}> ()
     X_Min = ReduceMin <keepdims: int = 0> (x)
     X_Min_Adjusted = Min (X_Min, Q_Min)
     X_Max = ReduceMax <keepdims: int = 0> (x)
     X_Max_Adjusted = Max (X_Max, Q_Min)
     X_Range = Sub (X_Max_Adjusted, X_Min_Adjusted)
     Scale = Div (X_Range, Q_Max)
     Min_Scaled = Div (X_Min_Adjusted, Scale)
     Initial_ZeroPoint_FP = Sub (Q_Min, Min_Scaled)
     Clipped_ZeroPoint_FP = Clip (Initial_ZeroPoint_FP, Q_Min, Q_Max)
     Rounded_ZeroPoint_FP = Round (Clipped_ZeroPoint_FP)
     Zeropoint = Cast <to: int = 2> (Rounded_ZeroPoint_FP)
     y_scale = Identity (Scale)
     y_zero_point = Identity (Zeropoint)
     y = QuantizeLinear (x, Scale, Zeropoint)
  }
type_constraints:
- name: T1
  doc: Constrain 'x' to float tensor.
  allowed:
  - tensor(float)
- name: T2
  doc: Constrain 'y_zero_point' and 'y' to 8-bit unsigned integer tensor.
  allowed:
  - tensor(uint8)
inputs:
- name: x
  type: T1
  doc: Input tensor
outputs:
- name: y
  type: T2
  doc: Quantized output tensor
- name: y_scale
  type: tensor(float)
  doc: Output scale. It's a scalar, which means a per-tensor/layer quantization.
- name: y_zero_point
  type: T2
  doc: Output zero point. It's a scalar, which means a per-tensor/layer quantization.
---

A Function to fuse calculation for Scale, Zero Point and FP32->8Bit conversion of FP32 Input data.
Outputs Scale, ZeroPoint and Quantized Input for a given FP32 Input.
Scale is calculated as:
```
y_scale = (maximum(0, max(x)) - minimum(0, min(x))) / (qmax - qmin)
```

* where qmax and qmin are max and min values for quantization range i.e. [0, 255] in case of uint8
* data range is adjusted to include 0.

Zero point is calculated as:
```
intermediate_zero_point = qmin - min(x)/y_scale
y_zero_point = cast(round(saturate(intermediate_zero_point)))
```

* where qmax and qmin are max and min values for quantization range .i.e [0, 255] in case of uint8
* for saturation, it saturates to [0, 255] if it's uint8, or [-127, 127] if it's int8. Right now only uint8 is supported.
* rounding to nearest ties to even.

Data quantization formula is:
```
y = saturate (round (x / y_scale) + y_zero_point)
```

* for saturation, it saturates to [0, 255] if it's uint8, or [-127, 127] if it's int8. Right now only uint8 is supported.
* rounding to nearest ties to even.
