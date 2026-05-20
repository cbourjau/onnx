// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
#include <string>

#include "onnx/defs/generated/op_specs_generated.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

static void binaryLogicalOpInference_opset7(InferenceContext& ctx) {
  // Type inference
  updateOutputElemType(ctx, 0, TensorProto::BOOL);
  // Shape inference
  if (hasNInputShapes(ctx, 2))
    bidirectionalBroadcastShapeInference(
        ctx.getInputType(0)->tensor_type().shape(),
        ctx.getInputType(1)->tensor_type().shape(),
        *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
}

static std::function<void(OpSchema&)> BinaryLogicDocGenerator_opset12(const char* name) {
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
    schema.Input(0, "A", "First input operand for the logical operator.", "T");
    schema.Input(1, "B", "Second input operand for the logical operator.", "T");
    schema.Output(0, "C", "Result tensor.", "T1");
    schema.TypeAndShapeInferenceFunction(binaryLogicalOpInference_opset7);
  };
}

ONNX_OPERATOR_SET_SCHEMA(
    Greater,
    9,
    OpSchema().FillUsing(Greater_v9_FillSpec).TypeAndShapeInferenceFunction(binaryLogicalOpInference_opset7));

ONNX_OPERATOR_SET_SCHEMA(
    Less,
    9,
    OpSchema().FillUsing(Less_v9_FillSpec).TypeAndShapeInferenceFunction(binaryLogicalOpInference_opset7));

ONNX_OPERATOR_SET_SCHEMA(
    Equal,
    11,
    OpSchema().FillUsing(Equal_v11_FillSpec).TypeAndShapeInferenceFunction(binaryLogicalOpInference_opset7));

inline static void logicalOpInference_opset1(InferenceContext& ctx) {
  updateOutputElemType(ctx, 0, TensorProto::BOOL);
  if (hasInputShape(ctx, 0)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

static std::function<void(OpSchema&)> BinaryLogicDocGenerator_opset1(const char* name) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(
        doc = R"DOC(
Returns the tensor resulted from performing the `{name}` logical operation
elementwise on the input tensors `A` and `B`.

If broadcasting is enabled, the right-hand-side argument will be broadcasted
to match the shape of left-hand-side argument. See the doc of `Add` for a
detailed description of the broadcasting rules.
)DOC";
        ReplaceAll(doc, "{name}", name););
    schema.SetDoc(doc);
    schema.Attr("broadcast", "Enable broadcasting", AttributeProto::INT, static_cast<int64_t>(0));
    schema.Attr("axis", "If set, defines the broadcast dimensions.", AttributeProto::INT, OPTIONAL_VALUE);
    schema.Input(0, "A", "Left input tensor for the logical operator.", "T");
    schema.Input(1, "B", "Right input tensor for the logical operator.", "T");
    schema.Output(0, "C", "Result tensor.", "T1");
    schema.TypeAndShapeInferenceFunction(logicalOpInference_opset1);
  };
}

static std::function<void(OpSchema&)> BinaryLogicDocGenerator_opset7(const char* name) {
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
    schema.Input(0, "A", "First input operand for the logical operator.", "T");
    schema.Input(1, "B", "Second input operand for the logical operator.", "T");
    schema.Output(0, "C", "Result tensor.", "T1");
    schema.TypeAndShapeInferenceFunction(binaryLogicalOpInference_opset7);
  };
}

ONNX_OPERATOR_SET_SCHEMA(
    And,
    1,
    OpSchema().FillUsing(And_v1_FillSpec).TypeAndShapeInferenceFunction(logicalOpInference_opset1));

ONNX_OPERATOR_SET_SCHEMA(
    Or,
    1,
    OpSchema().FillUsing(Or_v1_FillSpec).TypeAndShapeInferenceFunction(logicalOpInference_opset1));

ONNX_OPERATOR_SET_SCHEMA(
    Xor,
    1,
    OpSchema().FillUsing(Xor_v1_FillSpec).TypeAndShapeInferenceFunction(logicalOpInference_opset1));

ONNX_OPERATOR_SET_SCHEMA(
    Greater,
    1,
    OpSchema().FillUsing(Greater_v1_FillSpec).TypeAndShapeInferenceFunction(logicalOpInference_opset1));

ONNX_OPERATOR_SET_SCHEMA(
    Less,
    1,
    OpSchema().FillUsing(Less_v1_FillSpec).TypeAndShapeInferenceFunction(logicalOpInference_opset1));

ONNX_OPERATOR_SET_SCHEMA(
    Equal,
    1,
    OpSchema().FillUsing(Equal_v1_FillSpec).TypeAndShapeInferenceFunction(logicalOpInference_opset1));

ONNX_OPERATOR_SET_SCHEMA(
    Equal,
    7,
    OpSchema().FillUsing(Equal_v7_FillSpec).TypeAndShapeInferenceFunction(binaryLogicalOpInference_opset7));

ONNX_OPERATOR_SET_SCHEMA(
    Greater,
    7,
    OpSchema().FillUsing(Greater_v7_FillSpec).TypeAndShapeInferenceFunction(binaryLogicalOpInference_opset7));

ONNX_OPERATOR_SET_SCHEMA(
    Less,
    7,
    OpSchema().FillUsing(Less_v7_FillSpec).TypeAndShapeInferenceFunction(binaryLogicalOpInference_opset7));

// Shares same doc generator as newer opset 16 version.
extern std::function<void(OpSchema&)> BinaryLogicDocGenerator(const char* name);

ONNX_OPERATOR_SET_SCHEMA(
    LessOrEqual,
    12,
    OpSchema()
        .FillUsing(LessOrEqual_v12_FillSpec)
        .TypeAndShapeInferenceFunction(InferenceFunction())
        .FunctionBody(R"ONNX(
        {
            O1 = Less (A, B)
            O2 = Equal (A, B)
            C = Or (O1, O2)
        }
        )ONNX"));

ONNX_OPERATOR_SET_SCHEMA(
    GreaterOrEqual,
    12,
    OpSchema()
        .FillUsing(GreaterOrEqual_v12_FillSpec)
        .TypeAndShapeInferenceFunction(InferenceFunction())
        .FunctionBody(R"ONNX(
        {
            O1 = Greater (A, B)
            O2 = Equal (A, B)
            C = Or (O1, O2)
        }
        )ONNX"));

ONNX_OPERATOR_SET_SCHEMA(
    Equal,
    13,
    OpSchema().FillUsing(Equal_v13_FillSpec).TypeAndShapeInferenceFunction(binaryLogicalOpInference));

} // namespace ONNX_NAMESPACE
