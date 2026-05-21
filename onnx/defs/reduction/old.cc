// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

#include "onnx/defs/generated/op_specs_generated.h"
#include "onnx/defs/reduction/utils.h"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {
static void reduceShapeInference_opset12(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  int64_t keep_dims = 1;
  const auto* const attr_proto = ctx.getAttribute("keepdims");
  if (attr_proto) {
    keep_dims = attr_proto->i();
  }
  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  int64_t input_ndim = input_shape.dim_size();
  auto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  std::vector<int64_t> axes;
  const auto* const axes_proto = ctx.getAttribute("axes");
  if (axes_proto)
    axes.assign(axes_proto->ints().begin(), axes_proto->ints().end());

  for (int64_t& axe : axes) {
    if (axe < -input_ndim || axe >= input_ndim) {
      fail_shape_inference("axis must be in [-rank, rank-1]. input rank was ", input_ndim);
    }
    if (axe < 0)
      axe += input_ndim;
  }
  for (int i = 0; i < input_ndim; ++i) {
    // axes empty means reduce all dim
    if (!axes.empty() && std::find(axes.begin(), axes.end(), i) == axes.end()) {
      auto* dim = output_shape->add_dim();
      dim->CopyFrom(input_shape.dim(i));
    } else {
      if (keep_dims == 1) {
        auto* dim = output_shape->add_dim();
        dim->set_dim_value(1);
      }
    }
  }
}
ONNX_OPERATOR_SET_SCHEMA(
    ReduceMax,
    12,
    OpSchema().FillUsing(ReduceMax_v12_FillSpec).TypeAndShapeInferenceFunction(reduceShapeInference_opset12));

ONNX_OPERATOR_SET_SCHEMA(
    ReduceMin,
    12,
    OpSchema().FillUsing(ReduceMin_v12_FillSpec).TypeAndShapeInferenceFunction(reduceShapeInference_opset12));

ONNX_OPERATOR_SET_SCHEMA(
    ReduceSum,
    11,
    OpSchema().FillUsing(ReduceSum_v11_FillSpec).TypeAndShapeInferenceFunction(reduceShapeInference_opset12));

ONNX_OPERATOR_SET_SCHEMA(
    ReduceSumSquare,
    11,
    OpSchema().FillUsing(ReduceSumSquare_v11_FillSpec).TypeAndShapeInferenceFunction(reduceShapeInference_opset12));

ONNX_OPERATOR_SET_SCHEMA(
    ReduceMean,
    11,
    OpSchema().FillUsing(ReduceMean_v11_FillSpec).TypeAndShapeInferenceFunction(reduceShapeInference_opset12));

ONNX_OPERATOR_SET_SCHEMA(
    ReduceProd,
    11,
    OpSchema().FillUsing(ReduceProd_v11_FillSpec).TypeAndShapeInferenceFunction(reduceShapeInference_opset12));

ONNX_OPERATOR_SET_SCHEMA(
    ReduceLogSum,
    11,
    OpSchema().FillUsing(ReduceLogSum_v11_FillSpec).TypeAndShapeInferenceFunction(reduceShapeInference_opset12));

ONNX_OPERATOR_SET_SCHEMA(
    ReduceLogSumExp,
    11,
    OpSchema().FillUsing(ReduceLogSumExp_v11_FillSpec).TypeAndShapeInferenceFunction(reduceShapeInference_opset12));

ONNX_OPERATOR_SET_SCHEMA(
    ReduceL1,
    11,
    OpSchema().FillUsing(ReduceL1_v11_FillSpec).TypeAndShapeInferenceFunction(reduceShapeInference_opset12));

ONNX_OPERATOR_SET_SCHEMA(
    ReduceL2,
    11,
    OpSchema().FillUsing(ReduceL2_v11_FillSpec).TypeAndShapeInferenceFunction(reduceShapeInference_opset12));

static void argReduceShapeInference_opset12(InferenceContext& ctx) {
  // set output element type to int64
  updateOutputElemType(ctx, 0, TensorProto_DataType_INT64);

  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  auto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  int64_t input_ndim = input_shape.dim_size();
  int64_t axis = 0; // default to 0
  const auto* const axis_proto = ctx.getAttribute("axis");
  if (axis_proto) {
    axis = axis_proto->i();
    if (axis < -input_ndim || axis >= input_ndim) {
      fail_shape_inference("'axis' must be in [-rank(indices), rank(indices)-1]");
    }
    if (axis < 0)
      axis += input_ndim;
  }

  int64_t keep_dims = 1;
  const auto* const attr_proto = ctx.getAttribute("keepdims");
  if (attr_proto) {
    keep_dims = attr_proto->i();
  }
  // do we need handle negative axis?
  for (int i = 0; i < input_ndim; ++i) {
    if (i != axis) {
      auto* dim = output_shape->add_dim();
      dim->CopyFrom(input_shape.dim(i));
    } else {
      if (keep_dims == 1) {
        auto* dim = output_shape->add_dim();
        dim->set_dim_value(1);
      }
    }
  }
}
// namespace ONNX_NAMESPACE

ONNX_OPERATOR_SET_SCHEMA(
    ArgMax,
    12,
    OpSchema().FillUsing(ArgMax_v12_FillSpec).TypeAndShapeInferenceFunction(argReduceShapeInference_opset12));

ONNX_OPERATOR_SET_SCHEMA(
    ArgMin,
    12,
    OpSchema().FillUsing(ArgMin_v12_FillSpec).TypeAndShapeInferenceFunction(argReduceShapeInference_opset12));

static void reduceShapeInference_opset1(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  int64_t keep_dims = 1;
  const auto* const attr_proto = ctx.getAttribute("keepdims");
  if (attr_proto) {
    keep_dims = attr_proto->i();
  }
  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  int64_t input_ndim = input_shape.dim_size();
  auto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  std::vector<int64_t> axes;
  const auto* const axes_proto = ctx.getAttribute("axes");
  if (axes_proto)
    axes.assign(axes_proto->ints().begin(), axes_proto->ints().end());

  for (int64_t& axe : axes) {
    if (axe < 0)
      axe += input_ndim;
  }
  for (int i = 0; i < input_ndim; ++i) {
    // axes empty means reduce all dim
    if (!axes.empty() && std::find(axes.begin(), axes.end(), i) == axes.end()) {
      auto* dim = output_shape->add_dim();
      dim->CopyFrom(input_shape.dim(i));
    } else {
      if (keep_dims == 1) {
        auto* dim = output_shape->add_dim();
        dim->set_dim_value(1);
      }
    }
  }
}
ONNX_OPERATOR_SET_SCHEMA(
    ReduceMax,
    1,
    OpSchema().FillUsing(ReduceMax_v1_FillSpec).TypeAndShapeInferenceFunction(reduceShapeInference_opset1));

ONNX_OPERATOR_SET_SCHEMA(
    ReduceMin,
    1,
    OpSchema().FillUsing(ReduceMin_v1_FillSpec).TypeAndShapeInferenceFunction(reduceShapeInference_opset1));

ONNX_OPERATOR_SET_SCHEMA(
    ReduceSum,
    1,
    OpSchema().FillUsing(ReduceSum_v1_FillSpec).TypeAndShapeInferenceFunction(reduceShapeInference_opset1));

ONNX_OPERATOR_SET_SCHEMA(
    ReduceSumSquare,
    1,
    OpSchema().FillUsing(ReduceSumSquare_v1_FillSpec).TypeAndShapeInferenceFunction(reduceShapeInference_opset1));

ONNX_OPERATOR_SET_SCHEMA(
    ReduceMean,
    1,
    OpSchema().FillUsing(ReduceMean_v1_FillSpec).TypeAndShapeInferenceFunction(reduceShapeInference_opset1));

ONNX_OPERATOR_SET_SCHEMA(
    ReduceProd,
    1,
    OpSchema().FillUsing(ReduceProd_v1_FillSpec).TypeAndShapeInferenceFunction(reduceShapeInference_opset1));

ONNX_OPERATOR_SET_SCHEMA(
    ReduceLogSum,
    1,
    OpSchema().FillUsing(ReduceLogSum_v1_FillSpec).TypeAndShapeInferenceFunction(reduceShapeInference_opset1));

ONNX_OPERATOR_SET_SCHEMA(
    ReduceLogSumExp,
    1,
    OpSchema().FillUsing(ReduceLogSumExp_v1_FillSpec).TypeAndShapeInferenceFunction(reduceShapeInference_opset1));

ONNX_OPERATOR_SET_SCHEMA(
    ReduceL1,
    1,
    OpSchema().FillUsing(ReduceL1_v1_FillSpec).TypeAndShapeInferenceFunction(reduceShapeInference_opset1));

ONNX_OPERATOR_SET_SCHEMA(
    ReduceL2,
    1,
    OpSchema().FillUsing(ReduceL2_v1_FillSpec).TypeAndShapeInferenceFunction(reduceShapeInference_opset1));

ONNX_OPERATOR_SET_SCHEMA(
    ReduceMax,
    11,
    OpSchema().FillUsing(ReduceMax_v11_FillSpec).TypeAndShapeInferenceFunction(reduceShapeInference_opset1));

ONNX_OPERATOR_SET_SCHEMA(
    ReduceMin,
    11,
    OpSchema().FillUsing(ReduceMin_v11_FillSpec).TypeAndShapeInferenceFunction(reduceShapeInference_opset1));

static void argReduceShapeInference_opset1(InferenceContext& ctx) {
  // set output element type to int64
  updateOutputElemType(ctx, 0, TensorProto_DataType_INT64);

  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  auto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  int64_t input_ndim = input_shape.dim_size();
  int64_t axis = 0; // default to 0
  const auto* const axis_proto = ctx.getAttribute("axis");
  if (axis_proto) {
    axis = axis_proto->i();
    if (axis < 0)
      axis += input_ndim;
  }

  int64_t keep_dims = 1;
  const auto* const attr_proto = ctx.getAttribute("keepdims");
  if (attr_proto) {
    keep_dims = attr_proto->i();
  }
  // do we need handle negative axis?
  for (int i = 0; i < input_ndim; ++i) {
    if (i != axis) {
      auto* dim = output_shape->add_dim();
      dim->CopyFrom(input_shape.dim(i));
    } else {
      if (keep_dims == 1) {
        auto* dim = output_shape->add_dim();
        dim->set_dim_value(1);
      }
    }
  }
}
// namespace ONNX_NAMESPACE

ONNX_OPERATOR_SET_SCHEMA(
    ArgMax,
    1,
    OpSchema().FillUsing(ArgMax_v1_FillSpec).TypeAndShapeInferenceFunction(argReduceShapeInference_opset1));

ONNX_OPERATOR_SET_SCHEMA(
    ArgMin,
    1,
    OpSchema().FillUsing(ArgMin_v1_FillSpec).TypeAndShapeInferenceFunction(argReduceShapeInference_opset1));
// namespace ONNX_NAMESPACE

ONNX_OPERATOR_SET_SCHEMA(
    ArgMax,
    11,
    OpSchema().FillUsing(ArgMax_v11_FillSpec).TypeAndShapeInferenceFunction(argReduceShapeInference_opset12));
ONNX_OPERATOR_SET_SCHEMA(
    ArgMin,
    11,
    OpSchema().FillUsing(ArgMin_v11_FillSpec).TypeAndShapeInferenceFunction(argReduceShapeInference_opset12));

ONNX_OPERATOR_SET_SCHEMA(
    ReduceMax,
    13,
    OpSchema().FillUsing(ReduceMax_v13_FillSpec).TypeAndShapeInferenceFunction(reduceShapeInference));
ONNX_OPERATOR_SET_SCHEMA(
    ReduceMin,
    13,
    OpSchema().FillUsing(ReduceMin_v13_FillSpec).TypeAndShapeInferenceFunction(reduceShapeInference));
ONNX_OPERATOR_SET_SCHEMA(
    ReduceSumSquare,
    13,
    OpSchema().FillUsing(ReduceSumSquare_v13_FillSpec).TypeAndShapeInferenceFunction(reduceShapeInference));
ONNX_OPERATOR_SET_SCHEMA(
    ReduceMean,
    13,
    OpSchema().FillUsing(ReduceMean_v13_FillSpec).TypeAndShapeInferenceFunction(reduceShapeInference));
ONNX_OPERATOR_SET_SCHEMA(
    ReduceProd,
    13,
    OpSchema().FillUsing(ReduceProd_v13_FillSpec).TypeAndShapeInferenceFunction(reduceShapeInference));
ONNX_OPERATOR_SET_SCHEMA(
    ReduceLogSum,
    13,
    OpSchema().FillUsing(ReduceLogSum_v13_FillSpec).TypeAndShapeInferenceFunction(reduceShapeInference));
ONNX_OPERATOR_SET_SCHEMA(
    ReduceLogSumExp,
    13,
    OpSchema().FillUsing(ReduceLogSumExp_v13_FillSpec).TypeAndShapeInferenceFunction(reduceShapeInference));
ONNX_OPERATOR_SET_SCHEMA(
    ReduceL1,
    13,
    OpSchema().FillUsing(ReduceL1_v13_FillSpec).TypeAndShapeInferenceFunction(reduceShapeInference));
ONNX_OPERATOR_SET_SCHEMA(
    ReduceL2,
    13,
    OpSchema().FillUsing(ReduceL2_v13_FillSpec).TypeAndShapeInferenceFunction(reduceShapeInference));

ONNX_OPERATOR_SET_SCHEMA(
    ReduceMax,
    18,
    OpSchema().FillUsing(ReduceMax_v18_FillSpec).TypeAndShapeInferenceFunction(reduceShapeInference));
ONNX_OPERATOR_SET_SCHEMA(
    ReduceMin,
    18,
    OpSchema().FillUsing(ReduceMin_v18_FillSpec).TypeAndShapeInferenceFunction(reduceShapeInference));
} // namespace ONNX_NAMESPACE
