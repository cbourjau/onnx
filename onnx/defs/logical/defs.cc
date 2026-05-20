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

// NOLINTNEXTLINE(misc-use-internal-linkage)
std::function<void(OpSchema&)> BinaryLogicDocGenerator(const char* name) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(
        doc = R"DOC(
Returns the tensor resulted from performing the `{name}` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

{broadcast_doc}
)DOC";
        ReplaceAll(doc, "{name}", name);
        ReplaceAll(doc, "{broadcast_doc}", GenerateBroadcastingDocMul().c_str()););
    schema.SetDoc(doc);
    schema.Input(
        0,
        "A",
        "First input operand for the logical operator.",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::NonDifferentiable);
    schema.Input(
        1,
        "B",
        "Second input operand for the logical operator.",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::NonDifferentiable);
    schema.Output(0, "C", "Result tensor.", "T1", OpSchema::Single, true, 1, OpSchema::NonDifferentiable);
    schema.TypeAndShapeInferenceFunction(binaryLogicalOpInference);
  };
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

static constexpr const char* Not_ver1_doc = R"DOC(
Returns the negation of the input tensor element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Not,
    1,
    OpSchema().FillUsing(Not_v1_FillSpec).TypeAndShapeInferenceFunction(unaryLogicalOpInference));

static constexpr const char* BitShift_ver11_doc = R"DOC(
Bitwise shift operator performs element-wise operation. For each input element, if the
attribute "direction" is "RIGHT", this operator moves its binary representation toward
the right side so that the input value is effectively decreased. If the attribute "direction"
is "LEFT", bits of binary representation moves toward the left side, which results the
increase of its actual value. The input X is the tensor to be shifted and another input
Y specifies the amounts of shifting. For example, if "direction" is "Right", X is [1, 4],
and S is [1, 1], the corresponding output Z would be [0, 2]. If "direction" is "LEFT" with
X=[1, 2] and S=[1, 2], the corresponding output Y would be [2, 8].

Because this operator supports Numpy-style broadcasting, X's and Y's shapes are
not necessarily identical.
)DOC";

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

static constexpr const char* BitwiseNot_ver18_doc = R"DOC(
Returns the bitwise not of the input tensor element-wise.
)DOC";

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

static std::function<void(OpSchema&)> BinaryBitwiseDocGenerator(const char* name) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(
        doc = R"DOC(
Returns the tensor resulting from performing the bitwise `{name}` operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

{broadcast_doc}
)DOC";
        ReplaceAll(doc, "{name}", name);
        ReplaceAll(doc, "{broadcast_doc}", GenerateBroadcastingDocMul().c_str()););
    schema.SetDoc(doc);
    schema.Input(
        0,
        "A",
        "First input operand for the bitwise operator.",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::NonDifferentiable);
    schema.Input(
        1,
        "B",
        "Second input operand for the bitwise operator.",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::NonDifferentiable);
    schema.Output(0, "C", "Result tensor.", "T", OpSchema::Single, true, 1, OpSchema::NonDifferentiable);
    schema.TypeAndShapeInferenceFunction(bitwiseBroadcastShapeInference);
  };
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
