// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/defs/doc_strings.h"
#include "onnx/defs/generated/op_specs_generated.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

static constexpr const char* QuantizeLinear_ver25_doc = R"DOC(
The linear quantization operator consumes a high-precision tensor, a scale, and a zero point to compute the
low-precision/quantized tensor. The scale factor and zero point must have the same shape, determining the quantization
granularity. The quantization formula is `y = saturate((x / y_scale) + y_zero_point)`.

Saturation is done according to:
- uint16: [0, 65535]
- int16: [-32768, 32767]
- uint8: [0, 255]
- int8: [-128, 127]
- uint4: [0, 15]
- int4: [-8, 7]
- uint2: [0, 3]
- int2: [-2, 1]

For `(x / y_scale)`, it rounds to the nearest even. Refer to https://en.wikipedia.org/wiki/Rounding for details.

`y_zero_point` and `y` must have the same type. `y_zero_point` is usually not used for quantization to float8 and 4bit types, but the quantization
formula remains the same for consistency, and the type of the attribute `y_zero_point` still determines the quantization type.
`x` and `y_scale` are allowed to have different types. The type of `y_scale` determines the precision of the division operation between `x` and
`y_scale`, unless the `precision` attribute is specified.

There are three supported quantization granularities, determined by the shape of `y_scale`.
In all cases, `y_zero_point` must have the same shape as `y_scale`.
- Per-tensor (per-layer) quantization: `y_scale` is a scalar.
- Per-axis quantization: The scale must be a 1-D tensor, with the length of the quantization axis. For an input shape
 `(D0, ..., Di, ..., Dn)` and `axis=i`, `y_scale` is a 1-D tensor of length `Di`.
- Blocked quantization: The scale's shape is identical to the input's shape, except for one dimension, in which
  blocking is performed. Given `x` shape `(D0, ..., Di, ..., Dn)`, `axis=i`, and block size `B`: `y_scale` shape is
  `(D0, ..., ceil(Di/B), ..., Dn)`.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    QuantizeLinear,
    25,
    OpSchema()
        .FillUsing(QuantizeLinear_v25_FillSpec)
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          auto const zp_type = ctx.hasInput(2) ? ctx.getInputType(2) : nullptr;
          auto const output_dtype =
              static_cast<TensorProto_DataType>(getAttribute(ctx, "output_dtype", TensorProto::UNDEFINED));
          if (zp_type != nullptr) {
            auto const zp_elem_type = static_cast<TensorProto_DataType>(getTensorElementType(*zp_type));
            if (output_dtype != TensorProto::UNDEFINED && output_dtype != zp_elem_type) {
              fail_type_inference(
                  "output_dtype ",
                  TensorProto_DataType_Name(output_dtype),
                  " does not match y_zero_point type ",
                  TensorProto_DataType_Name(zp_elem_type),
                  ".");
            }
            propagateElemTypeFromInputToOutput(ctx, 2, 0);
          } else if (output_dtype != TensorProto::UNDEFINED) {
            propagateElemTypeFromAttributeToOutput(ctx, "output_dtype", 0);
          } else {
            updateOutputElemType(ctx, 0, TensorProto::UINT8);
          }
          if (!hasInputShape(ctx, 0)) {
            return;
          }

          auto& input_shape = getInputShape(ctx, 0);
          updateOutputShape(ctx, 0, input_shape);
        }));

ONNX_OPERATOR_SET_SCHEMA(
    DequantizeLinear,
    25,
    OpSchema()
        .FillUsing(DequantizeLinear_v25_FillSpec)
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          auto const output_dtype =
              static_cast<TensorProto_DataType>(getAttribute(ctx, "output_dtype", TensorProto::UNDEFINED));
          if (output_dtype != TensorProto::UNDEFINED) {
            propagateElemTypeFromAttributeToOutput(ctx, "output_dtype", 0);
          } else {
            propagateElemTypeFromInputToOutput(ctx, 1, 0);
          }
          if (!hasInputShape(ctx, 0)) {
            return;
          }
          auto& input_shape = getInputShape(ctx, 0);
          updateOutputShape(ctx, 0, input_shape);
        }));

static constexpr const char* DynamicQuantizeLinear_ver11_doc = R"DOC(
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
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    DynamicQuantizeLinear,
    11,
    OpSchema()
        .FillUsing(DynamicQuantizeLinear_v11_FillSpec)
        .FunctionBody(R"ONNX(
        {
           Q_Min = Constant<value = float {0.0}>()
           Q_Max = Constant<value = float {255.0}>()
           X_Min = ReduceMin <keepdims = 0> (x)
           X_Min_Adjusted = Min (X_Min, Q_Min)
           X_Max = ReduceMax <keepdims = 0> (x)
           X_Max_Adjusted = Max (X_Max, Q_Min)
           X_Range = Sub (X_Max_Adjusted, X_Min_Adjusted)
           Scale = Div (X_Range, Q_Max)
           Min_Scaled = Div (X_Min_Adjusted, Scale)
           Initial_ZeroPoint_FP = Sub (Q_Min, Min_Scaled)
           Clipped_ZeroPoint_FP = Clip (Initial_ZeroPoint_FP, Q_Min, Q_Max)
           Rounded_ZeroPoint_FP = Round (Clipped_ZeroPoint_FP)
           Zeropoint = Cast <to = 2> (Rounded_ZeroPoint_FP)
           y_scale = Identity (Scale)
           y_zero_point = Identity (Zeropoint)
           y = QuantizeLinear (x, Scale, Zeropoint)
        }
        )ONNX")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          updateOutputElemType(ctx, 0, TensorProto::UINT8);
          updateOutputElemType(ctx, 1, TensorProto::FLOAT);
          updateOutputElemType(ctx, 2, TensorProto::UINT8);

          ctx.getOutputType(1)->mutable_tensor_type()->mutable_shape();
          ctx.getOutputType(2)->mutable_tensor_type()->mutable_shape();

          if (!hasInputShape(ctx, 0))
            return;

          auto& input_shape = getInputShape(ctx, 0);
          updateOutputShape(ctx, 0, input_shape);
        }));

} // namespace ONNX_NAMESPACE
