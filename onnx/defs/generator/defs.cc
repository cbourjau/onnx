// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>

#include "onnx/defs/doc_strings.h"
#include "onnx/defs/function.h"
#include "onnx/defs/generated/op_specs_generated.h"
#include "onnx/defs/generator/utils.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {
ONNX_OPERATOR_SET_SCHEMA(
    Constant,
    25,
    OpSchema().FillUsing(Constant_v25_FillSpec).TypeAndShapeInferenceFunction(ConstantOpInference));

ONNX_OPERATOR_SET_SCHEMA(
    ConstantOfShape,
    25,
    OpSchema().FillUsing(ConstantOfShape_v25_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      if (ctx.getAttribute("value") != nullptr) {
        propagateElemTypeFromDtypeToOutput(ctx, ctx.getAttribute("value"), 0);
      } else {
        propagateElemTypeFromDtypeToOutput(ctx, TensorProto::FLOAT, 0);
      }

      bool found = false;
      TensorShapeProto output_shape = getShapeInput(ctx, 0, true, found);
      if (found) {
        *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() = output_shape;
      }
    }));

ONNX_OPERATOR_SET_SCHEMA(
    EyeLike,
    22,
    OpSchema().FillUsing(EyeLike_v22_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      if (ctx.getAttribute("dtype") != nullptr) {
        propagateElemTypeFromAttributeToOutput(ctx, "dtype", 0);
      } else {
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
      }
      if (hasInputShape(ctx, 0)) {
        auto& input_shape = getInputShape(ctx, 0);
        if (input_shape.dim_size() != 2) {
          fail_shape_inference("Input tensor must be 2-dimensional");
        }
      }
      propagateShapeFromInputToOutput(ctx, 0, 0);
    }));

ONNX_OPERATOR_SET_SCHEMA(
    RandomUniform,
    22,
    OpSchema().FillUsing(RandomUniform_v22_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromAttributeToOutput(ctx, "dtype", 0, TensorProto::FLOAT);
      propagateShapeFromAttributeToOutput(ctx, "shape", 0);
    }));

ONNX_OPERATOR_SET_SCHEMA(
    RandomNormal,
    22,
    OpSchema().FillUsing(RandomNormal_v22_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromAttributeToOutput(ctx, "dtype", 0, TensorProto::FLOAT);
      propagateShapeFromAttributeToOutput(ctx, "shape", 0);
    }));

ONNX_OPERATOR_SET_SCHEMA(
    RandomUniformLike,
    22,
    OpSchema().FillUsing(RandomUniformLike_v22_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      if (ctx.getAttribute("dtype") != nullptr)
        propagateElemTypeFromAttributeToOutput(ctx, "dtype", 0);
      else
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (!hasNInputShapes(ctx, 1)) {
        return;
      }
      propagateShapeFromInputToOutput(ctx, 0, 0);
    }));

ONNX_OPERATOR_SET_SCHEMA(
    RandomNormalLike,
    22,
    OpSchema().FillUsing(RandomNormalLike_v22_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      if (ctx.getAttribute("dtype") != nullptr)
        propagateElemTypeFromAttributeToOutput(ctx, "dtype", 0);
      else
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (!hasNInputShapes(ctx, 1)) {
        return;
      }
      propagateShapeFromInputToOutput(ctx, 0, 0);
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Multinomial,
    22,
    OpSchema().FillUsing(Multinomial_v22_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      auto dtype = ctx.getAttribute("dtype");
      auto dataType = TensorProto_DataType::TensorProto_DataType_INT32;
      if (dtype != nullptr) {
        dataType = static_cast<TensorProto_DataType>(dtype->i());
        if (dataType != TensorProto_DataType::TensorProto_DataType_INT32 &&
            dataType != TensorProto_DataType::TensorProto_DataType_INT64) {
          fail_type_inference("Output type must be int32 or int64");
        }
      }
      updateOutputElemType(ctx, 0, dataType);

      TensorShapeProto::Dimension batch_size, sample_size;
      if (hasInputShape(ctx, 0)) {
        auto& input_shape = getInputShape(ctx, 0);
        if (input_shape.dim_size() != 2) {
          fail_shape_inference("Input tensor must have rank 2");
        }
        batch_size = input_shape.dim(0);
      } // else statically-unknown batch-size
      sample_size.set_dim_value(getAttribute(ctx, "sample_size", 1));
      updateOutputShape(ctx, 0, {batch_size, sample_size});
    }));
template <typename T>
static int64_t
compute_output_dim_for_range(const TensorProto* start, const TensorProto* limit, const TensorProto* delta) {
  if (!start->dims().empty() || !limit->dims().empty() || !delta->dims().empty()) {
    fail_shape_inference("Input to 'Range' op should be scalars (Tensor with only one element and shape empty)");
  }

  const auto start_data = ParseData<T>(start);
  const auto limit_data = ParseData<T>(limit);
  const auto delta_data = ParseData<T>(delta);

  int64_t n = static_cast<int64_t>(ceil((1.0 * (limit_data[0] - start_data[0])) / delta_data[0]));

  n = std::max<int64_t>(n, 0);

  return n;
}

ONNX_OPERATOR_SET_SCHEMA(
    Range,
    11,
    OpSchema()
        .FillUsing(Range_v11_FillSpec)
        .FunctionBody(R"ONNX(
          {
            sub_result = Sub (limit, start)
            sub_result_casted = Cast <to = 1> (sub_result)
            delta_casted = Cast <to = 1> (delta)
            div_result = Div (sub_result_casted, delta_casted)
            ceil_result = Ceil (div_result)
            ceil_result_relu = Relu (ceil_result)
            ceil_result_relu_int = Cast <to = 7> (ceil_result_relu)
            ceil_result_relu_bool = Cast <to = 9> (ceil_result_relu)
            variadic_output, output = Loop (ceil_result_relu_int, ceil_result_relu_bool, start)
              <body = loop_body_attribute (int64 i, bool cond, prev) => (cond_out, current, range) {
                cond_out = Identity (cond)
                current = Add (prev, delta)
                range = Identity (prev)
              }>
          }
        )ONNX")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Type inference
          propagateElemTypeFromInputToOutput(ctx, 0, 0);

          // Shape inference
          const auto start_initializer = ctx.getInputData(0);
          const auto limit_initializer = ctx.getInputData(1);
          const auto delta_initializer = ctx.getInputData(2);

          // Output is always 1-D
          auto output_dim = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim();

          // If any of Range's inputs are not initializers, the output dimension
          // value would remain unknown.
          if (start_initializer != nullptr && limit_initializer != nullptr && delta_initializer != nullptr) {
            // Make sure the input types are homogeneous
            if ((start_initializer->data_type() != limit_initializer->data_type()) ||
                (start_initializer->data_type() != delta_initializer->data_type())) {
              fail_shape_inference("All inputs to 'Range' op must be of the same type");
            }

            // Explicitly compute the output dimension if Range's inputs are
            // stored in initializer list.
            if (start_initializer->data_type() == TensorProto::FLOAT) {
              output_dim->set_dim_value(
                  compute_output_dim_for_range<float>(start_initializer, limit_initializer, delta_initializer));
            } else if (start_initializer->data_type() == TensorProto::INT32) {
              output_dim->set_dim_value(
                  compute_output_dim_for_range<int32_t>(start_initializer, limit_initializer, delta_initializer));
            } else if (start_initializer->data_type() == TensorProto::INT64) {
              output_dim->set_dim_value(
                  compute_output_dim_for_range<int64_t>(start_initializer, limit_initializer, delta_initializer));
            } else if (start_initializer->data_type() == TensorProto::DOUBLE) {
              output_dim->set_dim_value(
                  compute_output_dim_for_range<double>(start_initializer, limit_initializer, delta_initializer));
            } else {
              // 'float16' has no native CPU type -
              // stop with rank inference, no action here
            }

            return;
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Bernoulli,
    22,
    OpSchema()
        .FillUsing(Bernoulli_v22_FillSpec)
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          if (ctx.getAttribute("dtype") != nullptr)
            propagateElemTypeFromAttributeToOutput(ctx, "dtype", 0);
          else
            propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }
          propagateShapeFromInputToOutput(ctx, 0, 0);
        })
        .SetContextDependentFunctionBodyBuilder(
            [](const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) -> bool {
              if (ctx.getInputType(0) == nullptr) {
                // we cannot create a correct function body without knowing the input type
                return false;
              }
              auto input_type = ctx.getInputType(0)->tensor_type().elem_type();
              auto dtype = ctx.getAttribute("dtype") != nullptr
                  ? static_cast<TensorProto_DataType>(ctx.getAttribute("dtype")->i())
                  : input_type;
              FunctionBuilder builder(functionProto);
              builder
                  .Add(
                      "X_random = RandomUniformLike <low = 0.0, high = 1.0, seed = @seed> (input)",
                      "dtype",
                      static_cast<int64_t>(input_type))
                  .Add("X_greater = Greater (X_random, input)")
                  .Add("output = Cast (X_greater)", "to", static_cast<int64_t>(dtype));
              schema.BuildFunction(functionProto);
              return true;
            }));
} // namespace ONNX_NAMESPACE
