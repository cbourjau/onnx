// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/defs/doc_strings.h"
#include "onnx/defs/function.h"
#include "onnx/defs/generated/op_specs_generated.h"
#include "onnx/defs/generator/utils.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

ONNX_OPERATOR_SET_SCHEMA(
    Bernoulli,
    15,
    OpSchema()
        .FillUsing(Bernoulli_v15_FillSpec)
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
                      "X_random = RandomUniformLike <low = 0.0, high = 1.0, seed = @seed> (input)", "dtype", input_type)
                  .Add("X_greater = Greater (X_random, input)")
                  .Add("output = Cast (X_greater)", "to", dtype);
              schema.BuildFunction(functionProto);
              return true;
            }));

ONNX_OPERATOR_SET_SCHEMA(
    Multinomial,
    7,
    OpSchema().FillUsing(Multinomial_v7_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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

ONNX_OPERATOR_SET_SCHEMA(
    RandomNormalLike,
    1,
    OpSchema().FillUsing(RandomNormalLike_v1_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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
    RandomUniformLike,
    1,
    OpSchema().FillUsing(RandomUniformLike_v1_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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
    RandomNormal,
    1,
    OpSchema().FillUsing(RandomNormal_v1_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromAttributeToOutput(ctx, "dtype", 0, TensorProto::FLOAT);
      propagateShapeFromAttributeToOutput(ctx, "shape", 0);
    }));

ONNX_OPERATOR_SET_SCHEMA(
    RandomUniform,
    1,
    OpSchema().FillUsing(RandomUniform_v1_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromAttributeToOutput(ctx, "dtype", 0, TensorProto::FLOAT);
      propagateShapeFromAttributeToOutput(ctx, "shape", 0);
    }));

ONNX_OPERATOR_SET_SCHEMA(
    EyeLike,
    9,
    OpSchema().FillUsing(EyeLike_v9_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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
    Constant,
    24,
    OpSchema().FillUsing(Constant_v24_FillSpec).TypeAndShapeInferenceFunction(ConstantOpInference));

ONNX_OPERATOR_SET_SCHEMA(
    Constant,
    23,
    OpSchema().FillUsing(Constant_v23_FillSpec).TypeAndShapeInferenceFunction(ConstantOpInference));

ONNX_OPERATOR_SET_SCHEMA(
    Constant,
    21,
    OpSchema().FillUsing(Constant_v21_FillSpec).TypeAndShapeInferenceFunction(ConstantOpInference));

ONNX_OPERATOR_SET_SCHEMA(
    Constant,
    19,
    OpSchema().FillUsing(Constant_v19_FillSpec).TypeAndShapeInferenceFunction(ConstantOpInference));

ONNX_OPERATOR_SET_SCHEMA(
    Constant,
    13,
    OpSchema().FillUsing(Constant_v13_FillSpec).TypeAndShapeInferenceFunction(ConstantOpInference));

ONNX_OPERATOR_SET_SCHEMA(
    Constant,
    12,
    OpSchema().FillUsing(Constant_v12_FillSpec).TypeAndShapeInferenceFunction(ConstantOpInference));
ONNX_OPERATOR_SET_SCHEMA(
    Constant,
    1,
    OpSchema().FillUsing(Constant_v1_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      auto attr_proto = ctx.getAttribute("value");
      if (nullptr == attr_proto)
        return; // attribute not present
      if (!attr_proto->has_t())
        return; // attribute has no tensor value
      const TensorProto& tensor_proto = attr_proto->t();
      updateOutputElemType(ctx, 0, tensor_proto.data_type());
      updateOutputShape(ctx, 0, tensor_proto);
    }));
ONNX_OPERATOR_SET_SCHEMA(
    Constant,
    9,
    OpSchema().FillUsing(Constant_v9_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      auto attr_proto = ctx.getAttribute("value");
      if (nullptr == attr_proto || !attr_proto->has_t())
        fail_shape_inference("Attribute 'value' of Constant node must exist with 'Tensor' data.");
      const TensorProto& tensor_proto = attr_proto->t();
      updateOutputElemType(ctx, 0, tensor_proto.data_type());
      updateOutputShape(ctx, 0, tensor_proto);
    }));
ONNX_OPERATOR_SET_SCHEMA(
    Constant,
    11,
    OpSchema().FillUsing(Constant_v11_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      auto value = ctx.getAttribute("value");
      auto sparse_value = ctx.getAttribute("sparse_value");

      if ((nullptr != value) && (nullptr != sparse_value))
        fail_shape_inference(
            "Only one of the attributes 'value' or 'sparse_value' must be specified for a Constant node.");

      if (nullptr != value) {
        // OpSchema::Verify check ensures that the attribute value has_t():
        const TensorProto& tensor_proto = value->t();
        updateOutputElemType(ctx, 0, tensor_proto.data_type());
        updateOutputShape(ctx, 0, tensor_proto);
        return;
      }

      if (nullptr != sparse_value) {
        // OpSchema::Verify check ensures that the attribute value
        // has_sparse_tensor():
        const SparseTensorProto& sparse = sparse_value->sparse_tensor();
        // checker.cc::check_sparse_tensor checks that the sparse-value is
        // well-formed
        updateOutputElemType(ctx, 0, sparse.values().data_type());
        auto output_shape = getOutputShape(ctx, 0);
        for (int i = 0; i < sparse.dims_size(); ++i)
          appendDim(output_shape, sparse.dims(i));
        return;
      }
      fail_shape_inference("One of the attributes 'value' or 'sparse_value' must be specified for a Constant node.");
    }));

ONNX_OPERATOR_SET_SCHEMA(
    ConstantOfShape,
    24,
    OpSchema().FillUsing(ConstantOfShape_v24_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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
    ConstantOfShape,
    23,
    OpSchema().FillUsing(ConstantOfShape_v23_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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
    ConstantOfShape,
    21,
    OpSchema().FillUsing(ConstantOfShape_v21_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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
    ConstantOfShape,
    20,
    OpSchema().FillUsing(ConstantOfShape_v20_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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
    ConstantOfShape,
    9,
    OpSchema().FillUsing(ConstantOfShape_v9_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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

} // namespace ONNX_NAMESPACE
