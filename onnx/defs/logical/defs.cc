// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "onnx/defs/generated/op_specs_generated.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

inline static void unaryLogicalOpInference(InferenceContext& ctx) {
  // Type inference
  updateOutputElemType(ctx, 0, TensorProto::BOOL);
  // Shape inference
  if (hasInputShape(ctx, 0)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

// NOLINTNEXTLINE(misc-use-internal-linkage)
void binaryLogicalOpInference(InferenceContext& ctx) {
  // Type inference
  updateOutputElemType(ctx, 0, TensorProto::BOOL);
  // Shape inference
  if (hasNInputShapes(ctx, 2))
    bidirectionalBroadcastShapeInference(
        ctx.getInputType(0)->tensor_type().shape(),
        ctx.getInputType(1)->tensor_type().shape(),
        *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
}
ONNX_OPERATOR_SET_SCHEMA(
    And,
    7,
    OpSchema().FillUsing(And_v7_FillSpec).TypeAndShapeInferenceFunction(binaryLogicalOpInference));

ONNX_OPERATOR_SET_SCHEMA(
    Or,
    7,
    OpSchema().FillUsing(Or_v7_FillSpec).TypeAndShapeInferenceFunction(binaryLogicalOpInference));

ONNX_OPERATOR_SET_SCHEMA(
    Xor,
    7,
    OpSchema().FillUsing(Xor_v7_FillSpec).TypeAndShapeInferenceFunction(binaryLogicalOpInference));

ONNX_OPERATOR_SET_SCHEMA(
    Greater,
    13,
    OpSchema().FillUsing(Greater_v13_FillSpec).TypeAndShapeInferenceFunction(binaryLogicalOpInference));

ONNX_OPERATOR_SET_SCHEMA(
    Less,
    13,
    OpSchema().FillUsing(Less_v13_FillSpec).TypeAndShapeInferenceFunction(binaryLogicalOpInference));

ONNX_OPERATOR_SET_SCHEMA(
    Equal,
    19,
    OpSchema().FillUsing(Equal_v19_FillSpec).TypeAndShapeInferenceFunction(binaryLogicalOpInference));
ONNX_OPERATOR_SET_SCHEMA(
    Not,
    1,
    OpSchema().FillUsing(Not_v1_FillSpec).TypeAndShapeInferenceFunction(unaryLogicalOpInference));
ONNX_OPERATOR_SET_SCHEMA(
    BitShift,
    11,
    OpSchema().FillUsing(BitShift_v11_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      // Type inference
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      // Shape inference
      if (hasNInputShapes(ctx, 2))
        bidirectionalBroadcastShapeInference(
            ctx.getInputType(0)->tensor_type().shape(),
            ctx.getInputType(1)->tensor_type().shape(),
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
    }));

ONNX_OPERATOR_SET_SCHEMA(
    LessOrEqual,
    16,
    OpSchema()
        .FillUsing(LessOrEqual_v16_FillSpec)
        .TypeAndShapeInferenceFunction(binaryLogicalOpInference)
        .FunctionBody(R"ONNX(
        {
            O1 = Less (A, B)
            O2 = Equal (A, B)
            C = Or (O1, O2)
        }
        )ONNX"));

ONNX_OPERATOR_SET_SCHEMA(
    GreaterOrEqual,
    16,
    OpSchema()
        .FillUsing(GreaterOrEqual_v16_FillSpec)
        .TypeAndShapeInferenceFunction(binaryLogicalOpInference)
        .FunctionBody(R"ONNX(
        {
            O1 = Greater (A, B)
            O2 = Equal (A, B)
            C = Or (O1, O2)
        }
        )ONNX"));
ONNX_OPERATOR_SET_SCHEMA(
    BitwiseNot,
    18,
    OpSchema().FillUsing(BitwiseNot_v18_FillSpec).TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static void bitwiseBroadcastShapeInference(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  // Shape inference
  if (hasNInputShapes(ctx, 2))
    bidirectionalBroadcastShapeInference(
        ctx.getInputType(0)->tensor_type().shape(),
        ctx.getInputType(1)->tensor_type().shape(),
        *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
}
ONNX_OPERATOR_SET_SCHEMA(
    BitwiseAnd,
    18,
    OpSchema().FillUsing(BitwiseAnd_v18_FillSpec).TypeAndShapeInferenceFunction(bitwiseBroadcastShapeInference));

ONNX_OPERATOR_SET_SCHEMA(
    BitwiseOr,
    18,
    OpSchema().FillUsing(BitwiseOr_v18_FillSpec).TypeAndShapeInferenceFunction(bitwiseBroadcastShapeInference));

ONNX_OPERATOR_SET_SCHEMA(
    BitwiseXor,
    18,
    OpSchema().FillUsing(BitwiseXor_v18_FillSpec).TypeAndShapeInferenceFunction(bitwiseBroadcastShapeInference));

} // namespace ONNX_NAMESPACE
